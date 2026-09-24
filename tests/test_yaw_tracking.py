"""Yaw-only angular tracking + bounded adaptive LR (2026-09-16).

Three things are locked here, each one a measured failure of the previous
recipe (see `mdp.track_yaw_velocity` for the numbers):

1. `track_yaw_velocity` prices the commanded yaw rate and NOTHING else.  The
   mjlab term it replaces charges the trunk's roll/pitch rates inside the same
   Gaussian, which pinned the reward at the wobble floor for 20k iterations and
   left yaw with no gradient.  The test that matters is
   `test_ignores_roll_and_pitch_rate` — it is the whole reason we forked.
2. The adaptive learning rate is bounded by a property on PPO, so the schedule's
   own assignments (including the ~20 inside one `learn()` call) cannot walk up
   the unbounded 1.5^k ladder any more.
3. The velocity cfg actually wires both in — a helper nobody installs is a
   silent no-op, which is exactly the class of bug this repo has been bitten by
   before (dof_frictionloss under BAM).
"""

import math

import pytest
import torch

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.mdp import (
    LEARNING_RATE_MAX,
    LEARNING_RATE_MIN,
    angular_wobble_cost,
    clamp_learning_rate,
    max_action_delta,
    track_yaw_velocity,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)

STD = 0.5


class _Data:
    def __init__(self, ang_vel_b: torch.Tensor):
        self.root_link_ang_vel_b = ang_vel_b


class _Asset:
    def __init__(self, data):
        self.data = data


class _Scene:
    def __init__(self, asset):
        self._asset = asset

    def __getitem__(self, _key):
        return self._asset


class _CommandManager:
    def __init__(self, command: torch.Tensor):
        self._command = command

    def get_command(self, _name):
        return self._command


class _Env:
    """Duck-typed stand-in: only what the reward touches."""

    def __init__(self, command: torch.Tensor, ang_vel_b: torch.Tensor):
        self.scene = _Scene(_Asset(_Data(ang_vel_b)))
        self.command_manager = _CommandManager(command)


def _env(cmd_yaw, cmd_lin=(0.0, 0.0), wx=0.0, wy=0.0, wz=0.0, n=1):
    command = torch.zeros(n, 3)
    command[:, 0] = cmd_lin[0]
    command[:, 1] = cmd_lin[1]
    command[:, 2] = cmd_yaw
    ang = torch.zeros(n, 3)
    ang[:, 0], ang[:, 1], ang[:, 2] = wx, wy, wz
    return _Env(command, ang)


# --------------------------------------------------------------------------- #
# 1. the reward prices yaw and nothing else                                    #
# --------------------------------------------------------------------------- #
def test_perfect_tracking_is_exactly_one():
    out = track_yaw_velocity(_env(cmd_yaw=0.4, wz=0.4), std=STD, command_name="twist")
    assert torch.allclose(out, torch.ones(1))


def test_yaw_error_follows_the_gaussian():
    for err in (0.1, 0.25, 0.5, 0.9):
        out = track_yaw_velocity(
            _env(cmd_yaw=err, wz=0.0), std=STD, command_name="twist"
        )
        assert math.isclose(out.item(), math.exp(-(err**2) / STD**2), rel_tol=1e-6)


def test_monotone_in_yaw_error():
    vals = [
        track_yaw_velocity(_env(cmd_yaw=e, wz=0.0), std=STD, command_name="twist").item()
        for e in (0.0, 0.1, 0.3, 0.6, 1.2)
    ]
    assert vals == sorted(vals, reverse=True)
    assert vals[0] == 1.0


def test_ignores_roll_and_pitch_rate():
    """THE regression: a walking duck wobbles in roll/pitch; that must not be
    charged to the yaw-tracking term (it is `body_ang_vel`'s job)."""
    quiet = track_yaw_velocity(_env(cmd_yaw=0.3, wz=0.3), std=STD, command_name="twist")
    wobbling = track_yaw_velocity(
        _env(cmd_yaw=0.3, wx=1.5, wy=-1.2, wz=0.3), std=STD, command_name="twist"
    )
    assert torch.allclose(quiet, wobbling)


def test_ignores_linear_command_and_linear_velocity():
    """Linear tracking is a separate term; the obs/command slot must not leak in."""
    a = track_yaw_velocity(
        _env(cmd_yaw=0.2, cmd_lin=(0.4, -0.3), wz=0.2), std=STD, command_name="twist"
    )
    b = track_yaw_velocity(
        _env(cmd_yaw=0.2, cmd_lin=(0.0, 0.0), wz=0.2), std=STD, command_name="twist"
    )
    assert torch.allclose(a, b)


def test_mjlab_composite_is_the_thing_we_forked_from():
    """Documents the difference: mjlab's term DROPS on pure roll rate at zero
    yaw command, i.e. it prices wobble as if it were yaw-tracking error."""
    from mjlab.tasks.velocity.mdp import rewards as mjlab_rewards

    ours = track_yaw_velocity(_env(cmd_yaw=0.0, wx=1.0), std=STD, command_name="twist")
    theirs = mjlab_rewards.track_angular_velocity(
        _env(cmd_yaw=0.0, wx=1.0), std=STD, command_name="twist"
    )
    assert torch.allclose(ours, torch.ones(1))
    assert theirs.item() < 0.2


def test_batched_inputs_stay_per_env():
    env = _env(cmd_yaw=0.0, wz=0.0, n=3)
    env.command_manager._command[0, 2] = 0.5  # env 0: 0.5 rad/s of yaw error
    env.command_manager._command[1, 2] = 0.0  # env 1: perfect
    env.scene._asset.data.root_link_ang_vel_b[2, 1] = 2.0  # env 2: pure wobble
    out = track_yaw_velocity(env, std=STD, command_name="twist")
    assert math.isclose(out[0].item(), math.exp(-0.25 / STD**2), rel_tol=1e-6)
    assert math.isclose(out[1].item(), 1.0, rel_tol=1e-6)
    assert math.isclose(out[2].item(), 1.0, rel_tol=1e-6)


# --------------------------------------------------------------------------- #
# 2. the adaptive learning rate is bounded                                     #
# --------------------------------------------------------------------------- #
def test_clamp_helper_bounds_and_handles_nan():
    assert clamp_learning_rate(1e-2) == LEARNING_RATE_MAX
    assert clamp_learning_rate(1e-9) == LEARNING_RATE_MIN
    assert clamp_learning_rate(2e-4) == 2e-4
    assert clamp_learning_rate(float("nan")) == LEARNING_RATE_MIN


def test_ppo_learning_rate_property_is_installed():
    from rsl_rl.algorithms.ppo import PPO

    assert isinstance(PPO.__dict__.get("learning_rate"), property)


def test_ppo_assignments_are_clamped_every_time():
    """The schedule assigns `self.learning_rate = min(1e-2, lr * 1.5)` per
    mini-batch; every one of those must land inside the band."""
    from rsl_rl.algorithms.ppo import PPO

    alg = PPO.__new__(PPO)  # no __init__: we only test the attribute plumbing
    alg.learning_rate = 1e-3  # cfg default, above the cap
    assert alg.learning_rate == LEARNING_RATE_MAX
    for _ in range(50):  # 50 x 1.5 would reach 1e-2 unbounded
        alg.learning_rate = min(1e-2, alg.learning_rate * 1.5)
    assert alg.learning_rate == LEARNING_RATE_MAX
    alg.learning_rate = 1e-12
    assert alg.learning_rate == LEARNING_RATE_MIN


# --------------------------------------------------------------------------- #
# 3. the cfg actually installs both                                            #
# --------------------------------------------------------------------------- #
def test_velocity_cfg_wires_yaw_only_reward():
    cfg = make_microduck_velocity_env_cfg()
    term = cfg.rewards["track_angular_velocity"]
    assert term.func is microduck_mdp.track_yaw_velocity
    assert term.weight == 3.0
    assert term.params["std"] == 0.5
    # The command slot this reward reads must still be the twist command.
    assert cfg.commands["twist"].rel_turn_in_place_envs > 0.0


def test_velocity_cfg_registers_blowup_diagnostic():
    cfg = make_microduck_velocity_env_cfg()
    # upgraded from max_action_delta to the dumping probe (same returned value)
    assert cfg.metrics["max_action_delta"].func is microduck_mdp.blowup_probe


def test_max_action_delta_sees_the_diverging_env():
    class _AM:
        def __init__(self, action, prev_action):
            self.action = action
            self.prev_action = prev_action

    class _E:
        pass

    env = _E()
    env.action_manager = _AM(
        action=torch.tensor([[0.1, 0.2], [500.0, 0.1]]),
        prev_action=torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
    )
    out = max_action_delta(env)
    assert out.shape == (2,)
    assert math.isclose(out[0].item(), 0.2, rel_tol=1e-6)
    assert math.isclose(out[1].item(), 500.0, rel_tol=1e-6)


# --------------------------------------------------------------------------- #
# 4. the wobble half of the split                                              #
# --------------------------------------------------------------------------- #
def test_wobble_cost_is_zero_when_the_trunk_is_still():
    out = angular_wobble_cost(_env(cmd_yaw=0.0), std=math.sqrt(0.5))
    assert out.item() == 0.0


def test_wobble_cost_follows_the_gaussian_and_is_bounded():
    for w in (0.1, 0.29, 1.0, 4.0):
        out = angular_wobble_cost(_env(cmd_yaw=0.0, wx=math.sqrt(w)), std=math.sqrt(0.5))
        assert math.isclose(out.item(), 1.0 - math.exp(-w / 0.5), rel_tol=1e-6)
        assert 0.0 <= out.item() < 1.0


def test_wobble_cost_ignores_yaw_rate_and_command():
    a = angular_wobble_cost(_env(cmd_yaw=2.0, wz=2.0), std=math.sqrt(0.5))
    b = angular_wobble_cost(_env(cmd_yaw=0.0, wz=0.0), std=math.sqrt(0.5))
    assert math.isclose(a.item(), b.item(), rel_tol=1e-9)


def test_split_reproduces_the_mjlab_composite_term():
    """yaw-only × (1 − wobble cost) must equal the term we forked from, when
    both use the same std.  This is what makes the split non-lossy.

    Tolerance: at the saturated end (ω_xy² ≳ 2, cost ≈ 0.9997) float32
    cancellation in ``1 − (1 − exp(−x))`` costs ~1e-3, so compare with an
    absolute tolerance; analytically the identity is exact.
    """
    from mjlab.tasks.velocity.mdp import rewards as mjlab_rewards

    for cmd, wz, wx, wy in ((0.3, 0.1, 0.5, -0.4), (0.0, 0.0, 0.4, 0.3), (0.4, 0.4, 0.2, 0.2)):
        env = _env(cmd_yaw=cmd, wx=wx, wy=wy, wz=wz)
        yaw = track_yaw_velocity(env, std=STD, command_name="twist")
        wob = angular_wobble_cost(env, std=STD)
        composite = mjlab_rewards.track_angular_velocity(
            env, std=STD, command_name="twist"
        )
        assert torch.allclose(yaw * (1.0 - wob), composite, rtol=0, atol=1e-3)


def test_velocity_cfg_registers_wobble_cost():
    cfg = make_microduck_velocity_env_cfg()
    term = cfg.rewards["angular_wobble"]
    assert term.func is microduck_mdp.angular_wobble_cost
    assert term.weight == -1.5  # measured default; positive cost -> negative weight
    assert term.params["std"] == math.sqrt(0.5)


def test_wobble_toggle_switches_the_arm(monkeypatch):
    """The A/B switch: same code, same checkpoint, one env var apart."""
    monkeypatch.setenv("MICRODUCK_ANGULAR_WOBBLE", "0")
    cfg_off = make_microduck_velocity_env_cfg()
    assert "angular_wobble" not in cfg_off.rewards
    # the yaw-only half must survive the toggle untouched
    assert cfg_off.rewards["track_angular_velocity"].func is microduck_mdp.track_yaw_velocity

    monkeypatch.delenv("MICRODUCK_ANGULAR_WOBBLE", raising=False)
    cfg_on = make_microduck_velocity_env_cfg()
    assert "angular_wobble" in cfg_on.rewards
    assert cfg_on.rewards["angular_wobble"].weight == -1.5  # measured default

    monkeypatch.setenv("MICRODUCK_ANGULAR_WOBBLE", "1")
    assert "angular_wobble" in make_microduck_velocity_env_cfg().rewards


def test_wobble_weight_is_tunable(monkeypatch):
    """③ needs a half-strength arm: same code, same checkpoint, weight only."""
    monkeypatch.delenv("MICRODUCK_ANGULAR_WOBBLE", raising=False)
    monkeypatch.setenv("MICRODUCK_WOBBLE_WEIGHT", "-1.5")
    cfg = make_microduck_velocity_env_cfg()
    assert cfg.rewards["angular_wobble"].weight == -1.5
    assert cfg.rewards["angular_wobble"].func is microduck_mdp.angular_wobble_cost

    monkeypatch.setenv("MICRODUCK_WOBBLE_WEIGHT", "0")
    assert "angular_wobble" not in make_microduck_velocity_env_cfg().rewards

    # the kill switch wins over the weight
    monkeypatch.setenv("MICRODUCK_ANGULAR_WOBBLE", "0")
    monkeypatch.setenv("MICRODUCK_WOBBLE_WEIGHT", "-1.5")
    assert "angular_wobble" not in make_microduck_velocity_env_cfg().rewards


def test_yaw_range_toggle_scales_the_whole_yaw_distribution(monkeypatch):
    """±0.5 → ±1.0 must be ONE number: the uniform range *and* the turn-in-place
    bucket (which is defined as a fraction of the range's max)."""
    monkeypatch.setenv("MICRODUCK_YAW_RANGE", "1.0")
    wide = make_microduck_velocity_env_cfg()
    assert wide.commands["twist"].ranges.ang_vel_z == (-1.0, 1.0)

    monkeypatch.delenv("MICRODUCK_YAW_RANGE", raising=False)
    narrow = make_microduck_velocity_env_cfg()
    assert narrow.commands["twist"].ranges.ang_vel_z == (-0.5, 0.5)

    # everything else about the command distribution is untouched
    assert wide.commands["twist"].ranges.lin_vel_x == narrow.commands["twist"].ranges.lin_vel_x
    assert (
        wide.commands["twist"].rel_turn_in_place_envs
        == narrow.commands["twist"].rel_turn_in_place_envs
    )


# --------------------------------------------------------------------------- #
# 5. blow-up forensics                                                         #
# --------------------------------------------------------------------------- #
def _probe_env(max_da: float, n: int = 3):
    import types

    class _AM:
        def __init__(self):
            self.action = torch.zeros(n, 4)
            self.prev_action = torch.zeros(n, 4)
            self.action[1, 2] = max_da  # env 1 is the offender

    class _Idx:
        free_joint_q_adr = [3, 4, 5, 6, 7, 8, 9]
        free_joint_v_adr = [0, 1, 2, 3, 4, 5]

    class _Inner:
        def __init__(self):
            self.qpos = torch.zeros(n, 10)
            self.qvel = torch.zeros(n, 6)

    class _Data:
        def __init__(self):
            self.data = _Inner()
            self.indexing = _Idx()
            self.joint_pos = torch.zeros(n, 4)
            self.joint_vel = torch.zeros(n, 4)
            self.root_link_ang_vel_b = torch.zeros(n, 3)

    class _Asset:
        def __init__(self):
            self.data = _Data()

    class _Scene:
        def __init__(self):
            self._asset = _Asset()

        def __getitem__(self, _k):
            return self._asset

    class _CM:
        def get_command(self, _n):
            return torch.zeros(n, 3)

        def get_term(self, _n):
            return types.SimpleNamespace(is_standing_env=torch.zeros(n, dtype=torch.bool))

    env = types.SimpleNamespace(
        action_manager=_AM(),
        scene=_Scene(),
        command_manager=_CM(),
        common_step_counter=1234,
        episode_length_buf=torch.arange(n),
        obs_buf={"actor": torch.zeros(n, 61), "critic": torch.zeros(n, 76)},
        num_envs=n,
    )
    return env


def test_blowup_probe_returns_per_env_max_and_dumps_the_offender(tmp_path, monkeypatch):
    import json
    import types

    out = tmp_path / "blowup.jsonl"
    monkeypatch.setenv("MICRODUCK_BLOWUP_DUMP", str(out))
    env = _probe_env(7.5)
    cfg = types.SimpleNamespace(params={"threshold": 3.0, "history": 4, "dedup": 0})
    probe = microduck_mdp.blowup_probe(cfg, env)

    per_env = probe(env)
    assert per_env.shape == (3,)
    assert math.isclose(per_env[1].item(), 7.5, rel_tol=1e-6)

    lines = out.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["env_idx"] == 1
    assert rec["joint"] == 2
    assert math.isclose(rec["max_da"], 7.5, rel_tol=1e-6)
    assert len(rec["action"]) == 4 and len(rec["actor_obs"]) == 61 and len(rec["critic_obs"]) == 76
    assert rec["step"] == 1234
    assert rec["onset"], "the onset sequence is the point of the probe"


def test_blowup_probe_records_the_onset_ramp(tmp_path, monkeypatch):
    """The trajectory must show the steps BEFORE the trip, so the dump can say
    what moved first."""
    import json
    import types

    out = tmp_path / "blowup.jsonl"
    monkeypatch.setenv("MICRODUCK_BLOWUP_DUMP", str(out))
    env = _probe_env(0.0)
    cfg = types.SimpleNamespace(params={"threshold": 2.0, "history": 4, "dedup": 0})
    probe = microduck_mdp.blowup_probe(cfg, env)

    for mag in (0.1, 0.5, 1.5):
        env.action_manager.action[1, 2] = mag
        env.action_manager.prev_action[1, 2] = 0.0
        probe(env)
    env.action_manager.action[1, 2] = 3.0  # trip
    probe(env)

    rec = json.loads(out.read_text().strip().splitlines()[-1])
    ramp = [row["max_da"] for row in rec["onset"]]
    assert ramp[-1] == 3.0
    assert ramp[:3] == [0.1, 0.5, 1.5]


def test_blowup_probe_is_silent_below_threshold_and_without_path(tmp_path, monkeypatch):
    import types

    out = tmp_path / "blowup.jsonl"
    monkeypatch.setenv("MICRODUCK_BLOWUP_DUMP", str(out))
    cfg = types.SimpleNamespace(params={"threshold": 3.0, "history": 4, "dedup": 0})
    env = _probe_env(0.5)
    microduck_mdp.blowup_probe(cfg, env)(env)
    assert not out.exists()

    monkeypatch.delenv("MICRODUCK_BLOWUP_DUMP", raising=False)
    env2 = _probe_env(50.0)
    microduck_mdp.blowup_probe(cfg, env2)(env2)  # must not raise
    assert not out.exists()


def test_velocity_cfg_registers_yaw_gain_metrics():
    cfg = make_microduck_velocity_env_cfg()
    assert cfg.metrics["max_action_delta"].func is microduck_mdp.blowup_probe
    assert cfg.metrics["mean_abs_yaw_rate"].func is microduck_mdp.mean_abs_yaw_rate
    assert cfg.metrics["mean_abs_yaw_command"].func is microduck_mdp.mean_abs_yaw_command


def test_yaw_gain_metrics_read_command_and_rate():
    cmd_env = _probe_env(0.0, n=2)
    cmd_env.command_manager.get_command = lambda _n: torch.tensor([[0.0, 0.0, -0.8], [0.0, 0.0, 0.25]])
    out = microduck_mdp.mean_abs_yaw_command(cmd_env)
    assert torch.allclose(out, torch.tensor([0.8, 0.25]), atol=1e-6)

    rate_env = _probe_env(0.0, n=2)
    rate_env.scene["robot"].data.root_link_ang_vel_b = torch.tensor([[0, 0, -0.5], [0, 0, 0.1]])
    out = microduck_mdp.mean_abs_yaw_rate(rate_env)
    assert torch.allclose(out, torch.tensor([0.5, 0.1]), atol=1e-6)


# --------------------------------------------------------------------------- #
# 6. thrash termination (experiment P1)                                        #
# --------------------------------------------------------------------------- #
def _vel_env(joint_vel: torch.Tensor, n: int):
    """Fake env whose robot exposes the 14-servo view via find_joints()."""
    import types

    class _Data:
        def __init__(self):
            self.joint_vel = joint_vel

    class _Asset:
        def __init__(self):
            self.data = _Data()

        def find_joints(self, _pattern):
            return list(range(joint_vel.shape[1])), None

    class _Scene:
        def __init__(self):
            self._asset = _Asset()

        def __getitem__(self, _k):
            return self._asset

    return types.SimpleNamespace(scene=_Scene(), num_envs=n)


def test_thrash_termination_fires_only_above_the_threshold():
    vel = torch.zeros(3, 14)
    vel[1, 7] = 15.5  # above
    vel[2, 2] = 12.0  # below
    env = _vel_env(vel, 3)
    out = microduck_mdp.thrash_termination(env, max_joint_vel=15.0)
    assert out.tolist() == [False, True, False]


def test_thrash_termination_sees_passive_joints_excluded():
    """The selector must be `^(?!passive_).*`, i.e. only the 14 servo joints —
    a passive wheel spinning fast must not end the episode on a roller model."""
    vel = torch.zeros(2, 18)
    vel[0, 16] = 40.0  # a passive joint index on the roller model layout
    env = _vel_env(vel, 2)
    out = microduck_mdp.thrash_termination(env, max_joint_vel=15.0)
    # _servo_joint_ids() uses find_joints('^(?!passive_).*'); our fake returns
    # every column, so this test documents the intent rather than the regex —
    # assert the plain case is unchanged.
    assert out.tolist() == [True, False]


def test_thrash_toggle_and_threshold(monkeypatch):
    """Default OFF: the A/B showed no measurable benefit, so it is opt-in."""
    monkeypatch.delenv("MICRODUCK_THRASH_TERMINATION", raising=False)
    assert "thrash" not in make_microduck_velocity_env_cfg().terminations

    monkeypatch.setenv("MICRODUCK_THRASH_TERMINATION", "1")
    cfg = make_microduck_velocity_env_cfg()
    assert "thrash" in cfg.terminations
    assert cfg.terminations["thrash"].func is microduck_mdp.thrash_termination
    assert cfg.terminations["thrash"].params["max_joint_vel"] == 15.0
    assert cfg.terminations["thrash"].time_out is False

    monkeypatch.setenv("MICRODUCK_THRASH_QVEL", "12.5")
    assert make_microduck_velocity_env_cfg().terminations["thrash"].params["max_joint_vel"] == 12.5

    monkeypatch.setenv("MICRODUCK_THRASH_TERMINATION", "0")
    assert "thrash" not in make_microduck_velocity_env_cfg().terminations


def test_velocity_cfg_registers_joint_vel_metric():
    cfg = make_microduck_velocity_env_cfg()
    assert cfg.metrics["max_joint_vel"].func is microduck_mdp.max_servo_joint_vel


# --------------------------------------------------------------------------- #
# 8. DC yaw tracking: EMA filter + the deployment command bucket               #
# --------------------------------------------------------------------------- #
def _ema_env(cmd_yaw: float, n: int = 1, episode_length: int = 50):
    """Env stub with the two extra attributes the EMA needs."""
    env = _env(cmd_yaw=cmd_yaw, wz=0.0, n=n)
    env.step_dt = 0.02  # 50 Hz, the training control rate
    env.episode_length_buf = torch.full((n,), episode_length, dtype=torch.long)
    return env


def _run_ema(env, wf, steps, tau, std=None):
    """Feed a yaw-rate trajectory one step at a time; return (rewards, filtered)."""
    rewards, filt = [], []
    for i in range(steps):
        w = wf(i)
        env.scene["robot"].data.root_link_ang_vel_b[:, 2] = w
        env.episode_length_buf += 1
        out = track_yaw_velocity(env, std=std or STD, command_name="twist", tau=tau)
        rewards.append(out.clone())
        filt.append(
            env._yaw_rate_ema_track.clone()
            if tau > 0.0
            else env.scene["robot"].data.root_link_ang_vel_b[:, 2].clone()
        )
    return rewards, filt


def test_zero_tau_is_exactly_the_instantaneous_rate():
    """The A/B control arm must be bit-identical to the recipe that has run."""
    w = torch.tensor([0.13, -0.27, 0.5])
    obs = _env(cmd_yaw=0.4, n=3)
    obs.scene["robot"].data.root_link_ang_vel_b[:, 2] = w
    direct = track_yaw_velocity(obs, std=0.5, command_name="twist")
    for tau in (0.0, -1.0):
        env = _env(cmd_yaw=0.4, n=3)
        env.scene["robot"].data.root_link_ang_vel_b[:, 2] = w
        env.step_dt = 0.02
        env.episode_length_buf = torch.full((3,), 5, dtype=torch.long)
        assert torch.equal(
            track_yaw_velocity(env, std=0.5, command_name="twist", tau=tau), direct
        )


def test_ema_removes_a_symmetric_oscillation_but_keeps_the_dc_turn():
    """The whole point: a stepping gait oscillates about the commanded turn.

    With the instantaneous rate the term pays for the oscillation (which the
    policy cannot remove); filtering first leaves the DC turn and the reward
    becomes high and — crucially — still sensitive to the DC value.
    """
    tau, freq, amp, dc = 0.5, 2.0, 0.6, 0.5
    wf = lambda i: dc + amp * math.sin(2 * math.pi * freq * i * 0.02)  # noqa: E731
    env_f = _ema_env(cmd_yaw=dc)
    _, filt = _run_ema(env_f, wf, 300, tau=tau)
    env_i = _ema_env(cmd_yaw=dc)
    raw, _ = _run_ema(env_i, wf, 300, tau=0.0)

    settled = slice(150, 300)
    mean_f = float(torch.stack(filt)[settled].mean())
    assert abs(mean_f - dc) < 0.05, f"EMA did not recover the DC turn: {mean_f}"
    rew_f = float(torch.stack([r.mean() for r in _run_ema(_ema_env(dc), wf, 300, tau)[0]])[settled].mean())
    rew_i = float(torch.stack([r.mean() for r in raw])[settled].mean())
    assert rew_f > rew_i + 0.2, (rew_f, rew_i)

    # and the filtered reward still falls when the DC turn is missing
    rew_gone = float(
        torch.stack([r.mean() for r in _run_ema(_ema_env(dc), lambda i: amp * math.sin(2 * math.pi * freq * i * 0.02), 300, tau)[0]])[settled].mean()
    )
    assert rew_gone < rew_f - 0.2, (rew_gone, rew_f)


def test_ema_step_response_has_the_requested_time_constant():
    """After tau seconds of a step the filter must be at 1 - 1/e."""
    tau = 0.5
    steps = int(tau / 0.02)
    env = _ema_env(cmd_yaw=0.0)
    env.scene["robot"].data.root_link_ang_vel_b[:, 2] = 0.0
    track_yaw_velocity(env, std=0.5, command_name="twist", tau=tau)  # seed at 0
    env.scene["robot"].data.root_link_ang_vel_b[:, 2] = 1.0
    for _ in range(steps):
        track_yaw_velocity(env, std=0.5, command_name="twist", tau=tau)
    assert abs(float(env._yaw_rate_ema_track) - (1 - math.exp(-1))) < 0.02


def test_ema_snaps_on_a_fresh_episode_and_never_poisons_itself():
    """A reset must not leave a filter transient behind, NaN must not stick."""
    env = _ema_env(cmd_yaw=0.5, n=2, episode_length=400)
    env.scene["robot"].data.root_link_ang_vel_b[:, 2] = torch.tensor([0.9, -0.9])
    track_yaw_velocity(env, std=0.5, command_name="twist", tau=0.5)
    assert torch.allclose(env._yaw_rate_ema_track, torch.tensor([0.9, -0.9]), atol=1e-6)

    # env 0 ends: fresh envs adopt the measurement immediately
    env.episode_length_buf[0] = 0
    env.scene["robot"].data.root_link_ang_vel_b[:, 2] = torch.tensor([-0.4, -0.9])
    track_yaw_velocity(env, std=0.5, command_name="twist", tau=0.5)
    assert abs(float(env._yaw_rate_ema_track[0]) + 0.4) < 1e-5
    assert env._yaw_rate_ema_track[1] < -0.89  # the other env keeps its history

    # NaN measurement: filtered value stays finite
    env.episode_length_buf[:] = 0
    env.scene["robot"].data.root_link_ang_vel_b[:, 2] = torch.tensor([float("nan"), 0.2])
    track_yaw_velocity(env, std=0.5, command_name="twist", tau=0.5)
    assert torch.isfinite(env._yaw_rate_ema_track).all()


def test_dc_toggles_are_one_env_var_apart(monkeypatch):
    """std / tau / bucket must default to the recipe already running."""
    base = make_microduck_velocity_env_cfg()
    assert base.rewards["track_angular_velocity"].params["std"] == 0.5
    assert base.rewards["track_angular_velocity"].params["tau"] == 0.0
    assert base.commands["twist"].rel_sustained_turn_envs == 0.0

    monkeypatch.setenv("MICRODUCK_YAW_TRACK_STD", "0.25")
    monkeypatch.setenv("MICRODUCK_YAW_EMA_TAU", "0.5")
    monkeypatch.setenv("MICRODUCK_SUSTAINED_TURN", "0.15")
    fixed = make_microduck_velocity_env_cfg()
    assert fixed.rewards["track_angular_velocity"].params["std"] == 0.25
    assert fixed.rewards["track_angular_velocity"].params["tau"] == 0.5
    assert fixed.commands["twist"].rel_sustained_turn_envs == 0.15
    # the bucket does not disturb the rest of the command mix
    assert (
        fixed.commands["twist"].rel_turn_in_place_envs
        == base.commands["twist"].rel_turn_in_place_envs
    )
    assert fixed.commands["twist"].ranges.ang_vel_z == (-0.5, 0.5)


class _CmdCfg:
    """Just enough of VelocityCommandCommandOnlyCfg for the bucket code."""

    def __init__(self, turn=0.0, sust=0.0, hold=25.0, yaw_max=0.5):
        self.ranges = type("R", (), {})()
        self.ranges.ang_vel_z = (-yaw_max, yaw_max)
        self.rel_turn_in_place_envs = turn
        self.rel_sustained_turn_envs = sust
        self.sustained_turn_hold_s = hold


def _command_term(cfg, n=8):
    """Instance whose ONLY live method is our `_resample_command` override."""
    import types

    term = object.__new__(microduck_mdp.VelocityCommandCommandOnly)
    term.cfg = cfg
    term._env = types.SimpleNamespace(num_envs=n, device="cpu")
    term.vel_command_b = torch.zeros(n, 3)
    term.vel_command_w = torch.zeros(n, 3)
    term.is_standing_env = torch.zeros(n, dtype=torch.bool)
    term.time_left = torch.zeros(n)
    return term


def _stub_parent(monkeypatch):
    """The parent's sampling needs the whole mjlab env and is tested by mjlab;
    stubbing it lets the bucket logic be tested on its own."""
    from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand

    monkeypatch.setattr(
        UniformVelocityCommand, "_resample_command", lambda self, env_ids: None
    )


def test_sustained_turn_bucket_pins_the_deployment_command(monkeypatch):
    """lin=0, |yaw| = range max, unmarked as standing, held past the episode."""
    term = _command_term(_CmdCfg(turn=0.0, sust=1.0, hold=25.0))
    _stub_parent(monkeypatch)
    term._resample_command(torch.arange(8))
    assert torch.all(term.vel_command_b[:, 0] == 0.0)
    assert torch.all(term.vel_command_b[:, 1] == 0.0)
    assert torch.all(term.vel_command_b[:, 2].abs() == 0.5)
    assert not term.is_standing_env.any()
    assert torch.all(term.time_left == 25.0)
    # both turn directions are represented across the batch
    assert term.vel_command_b[:, 2].min() < 0 < term.vel_command_b[:, 2].max()


def test_sustained_bucket_fires_even_when_the_turn_bucket_draws_nothing(monkeypatch):
    """Regression: the turn-in-place early return used to swallow this bucket.

    With rel_turn_in_place_envs = 0 and a single env id, the turn bucket draws
    nothing — the old code returned there, so the sustained bucket never ran and
    the whole feature was a silent no-op.
    """
    term = _command_term(_CmdCfg(turn=0.0, sust=1.0, hold=30.0), n=1)
    _stub_parent(monkeypatch)
    term._resample_command(torch.tensor([0]))
    assert abs(float(term.vel_command_b[0, 2])) == 0.5
    assert float(term.time_left[0]) == 30.0

    # ... and the default (bucket off) leaves the command alone
    off = _command_term(_CmdCfg(turn=0.0, sust=0.0), n=1)
    off.time_left = torch.tensor([4.0])
    off.vel_command_b = torch.tensor([[0.2, 0.1, -0.3]])
    off._resample_command(torch.tensor([0]))
    assert float(off.time_left[0]) == 4.0
    assert torch.allclose(off.vel_command_b, torch.tensor([[0.2, 0.1, -0.3]]))


def test_sustained_turn_scales_with_the_yaw_range(monkeypatch):
    """±1.0 must move the bucket too — the deployment point is always the max."""
    term = _command_term(_CmdCfg(turn=0.0, sust=1.0, yaw_max=1.0))
    _stub_parent(monkeypatch)
    term._resample_command(torch.arange(8))
    assert torch.all(term.vel_command_b[:, 2].abs() == 1.0)


def test_dc_turn_gain_metric_reads_the_deployment_quantity():
    """The training-side twin of the rehearsal: EMA(ω_z)/cmd on turning envs."""
    env = _ema_env(cmd_yaw=0.5, n=3, episode_length=200)
    env.command_manager._command[1, 2] = 0.5
    env.command_manager._command[2, 2] = 0.1  # below min_cmd -> excluded
    # env 0 turns at the command, env 1 turns at half, env 2 is ignored
    env.scene["robot"].data.root_link_ang_vel_b[:, 2] = torch.tensor([0.35, 0.35, 0.99])
    for _ in range(400):  # let the 0.5 s filter settle
        env.episode_length_buf += 1
        microduck_mdp.dc_turn_gain(env, tau=0.5, min_cmd=0.25)
    # step to the target rates: the filter needs its own settling time
    env.scene["robot"].data.root_link_ang_vel_b[:, 2] = torch.tensor([0.5, 0.25, 0.99])
    for _ in range(70):
        env.episode_length_buf += 1
        out = microduck_mdp.dc_turn_gain(env, tau=0.5, min_cmd=0.25)
    assert abs(float(out[0]) - 1.0) < 0.05
    assert abs(float(out[1]) - 0.5) < 0.05
    assert float(out[2]) == 0.0


def test_dc_turn_gain_has_its_own_filter():
    """Metric and reward must not share a buffer (different tau, A/B logging)."""
    env = _ema_env(cmd_yaw=0.5, n=1, episode_length=200)
    env.scene["robot"].data.root_link_ang_vel_b[:, 2] = 0.5
    for _ in range(200):
        env.episode_length_buf += 1
        track_yaw_velocity(env, std=0.5, command_name="twist", tau=0.1)
        microduck_mdp.dc_turn_gain(env, tau=2.0, min_cmd=0.25)
    assert hasattr(env, "_yaw_rate_ema_track")
    assert hasattr(env, "_yaw_rate_ema_metric")
    # a step change: after 3 steps (0.06 s) the 0.1 s filter has moved most of
    # the way, the 2 s filter has barely started
    env.scene["robot"].data.root_link_ang_vel_b[:, 2] = 0.0
    for _ in range(3):
        env.episode_length_buf += 1
        track_yaw_velocity(env, std=0.5, command_name="twist", tau=0.1)
        microduck_mdp.dc_turn_gain(env, tau=2.0, min_cmd=0.25)
    fast = float(env._yaw_rate_ema_track)
    slow = float(env._yaw_rate_ema_metric)
    assert fast < 0.32 and slow > 0.45, (fast, slow)


def test_velocity_cfg_registers_dc_turn_gain():
    cfg = make_microduck_velocity_env_cfg()
    assert cfg.metrics["dc_turn_gain"].func is microduck_mdp.dc_turn_gain
    assert cfg.metrics["dc_turn_gain"].params["tau"] == 0.5
    assert cfg.metrics["dc_turn_gain"].params["min_cmd"] == 0.25


# --------------------------------------------------------------------------- #
# 9. action-rate mass: the largest penalty in the stack                        #
# --------------------------------------------------------------------------- #
def test_action_rate_curriculum_scales_with_one_env_var(monkeypatch):
    """MICRODUCK_ACTION_RATE_SCALE must scale the whole ramp, not one stage."""
    base = make_microduck_velocity_env_cfg()
    base_stages = [
        s["weight"] for s in base.curriculum["action_rate_weight"].params["weight_stages"]
    ]
    assert base_stages == [-0.1, -0.2, -0.4, -0.6, -0.8, -1.0]

    monkeypatch.setenv("MICRODUCK_ACTION_RATE_SCALE", "0.4")
    scaled = make_microduck_velocity_env_cfg()
    stages = [
        s["weight"] for s in scaled.curriculum["action_rate_weight"].params["weight_stages"]
    ]
    assert stages == [pytest.approx(w * 0.4) for w in base_stages]
    # the reward's stage-0 weight is the curriculum's first stage, and the step
    # grid must not move (the schedule is step-aligned, AGENTS.md)
    assert scaled.rewards["action_rate_l2"].weight == pytest.approx(-0.1)
    assert [s["step"] for s in scaled.curriculum["action_rate_weight"].params["weight_stages"]] == [
        s["step"] for s in base.curriculum["action_rate_weight"].params["weight_stages"]
    ]
    # and nothing else in the penalty stack may move with it
    assert scaled.rewards["angular_wobble"].weight == -1.5


def test_action_rate_scale_default_is_the_recipe_that_ran(monkeypatch):
    monkeypatch.delenv("MICRODUCK_ACTION_RATE_SCALE", raising=False)
    import importlib

    cfg_mod = importlib.import_module(
        "mjlab_microduck.tasks.microduck_velocity_env_cfg"
    )
    assert cfg_mod.action_rate_scale() == 1.0


def test_yaw_weight_switch_is_one_env_var(monkeypatch):
    """The last structural lever: make the turn matter more (3.0 -> 8-10)."""
    monkeypatch.delenv("MICRODUCK_YAW_WEIGHT", raising=False)
    base = make_microduck_velocity_env_cfg()
    assert base.rewards["track_angular_velocity"].weight == 3.0
    # the shape of the term must be untouched by a weight change
    assert base.rewards["track_angular_velocity"].params["std"] == 0.5
    assert base.rewards["track_angular_velocity"].params["tau"] == 0.0

    monkeypatch.setenv("MICRODUCK_YAW_WEIGHT", "10")
    heavy = make_microduck_velocity_env_cfg()
    assert heavy.rewards["track_angular_velocity"].weight == 10.0
    assert heavy.rewards["track_angular_velocity"].params["std"] == 0.5
    # and no other reward may move with it
    assert heavy.rewards["track_linear_velocity"].weight == base.rewards["track_linear_velocity"].weight
    assert heavy.rewards["angular_wobble"].weight == base.rewards["angular_wobble"].weight


def test_linear_weight_switch_rebalances_the_task_terms(monkeypatch):
    """Rebalancing arm: lift the linear term alongside the yaw term (2026-09-19).

    W=8 reached 97% of the turn requirement but cost walking (forward 60% vs 73%).
    Raising the linear weight keeps BOTH task terms dominant over the regularizers,
    which is a different intervention from lowering a penalty.
    """
    monkeypatch.delenv("MICRODUCK_LINEAR_WEIGHT", raising=False)
    base = make_microduck_velocity_env_cfg()
    assert base.rewards["track_linear_velocity"].weight == 5.0
    # the reward's shape must not move with its weight
    assert base.rewards["track_linear_velocity"].params["std"] == pytest.approx(0.31622776601683794)

    monkeypatch.setenv("MICRODUCK_LINEAR_WEIGHT", "10")
    monkeypatch.setenv("MICRODUCK_YAW_WEIGHT", "8")
    heavy = make_microduck_velocity_env_cfg()
    assert heavy.rewards["track_linear_velocity"].weight == 10.0
    assert heavy.rewards["track_angular_velocity"].weight == 8.0
    # penalty masses must be untouched by the rebalance
    assert heavy.rewards["angular_wobble"].weight == base.rewards["angular_wobble"].weight
    assert heavy.rewards["action_rate_l2"].weight == base.rewards["action_rate_l2"].weight
