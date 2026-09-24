"""A3: the sustained-forward-command bucket (`rel_sustained_walk_envs`).

Evidence that motivated it (logs/seed_sweep_results.txt): velocity walk passes only **12/25**
demonstration rounds, and every failing round reads `v = 0.05-0.12 m/s` with `g = -1.00` - upright
and simply not walking. The training distribution explains it: `rel_standing_envs` is curriculum
ramped to **0.50**, so half of all experience is "command = 0, stand still", while the deployment
and every evaluation episode start with a non-zero forward command. This bucket pins a fraction of
envs on that operating point for longer than an episode, exactly as `rel_sustained_turn_envs` does
for the turn.

The behaviour test matters more than the cfg test here: the failure mode this guards against is the
standing mask silently zeroing the command (which is what `rel_turn_in_place_envs` had to work
around with `is_standing_env[ids] = False`).
"""

import types

import torch

import mjlab_microduck.tasks.mdp as microduck_mdp


def _cmd_obj(n=8, **cfg_attrs):
    """A VelocityCommandCommandOnly with the parent's random sampling neutralised."""
    obj = object.__new__(microduck_mdp.VelocityCommandCommandOnly)
    obj.cfg = types.SimpleNamespace(**cfg_attrs)
    # `device` / `num_envs` are read-only properties that dereference `self._env`, so the fake env
    # is what has to exist (setting obj.device / obj.num_envs raises AttributeError).
    obj._env = types.SimpleNamespace(device="cpu", num_envs=n)
    obj.vel_command_b = torch.zeros(n, 3)
    obj.vel_command_w = torch.zeros(n, 3)
    obj.is_standing_env = torch.ones(n, dtype=torch.bool)     # every env starts "standing"
    obj.time_left = torch.zeros(n)
    return obj


def test_bucket_is_a_no_op_at_zero():
    obj = _cmd_obj(rel_sustained_walk_envs=0.0)
    obj._sustained_walk_bucket(torch.arange(8))
    assert torch.all(obj.vel_command_b == 0.0)
    assert torch.all(obj.is_standing_env), "nothing may be un-marked when the bucket is off"


def test_bucket_pins_a_forward_command_and_clears_standing():
    obj = _cmd_obj(rel_sustained_walk_envs=1.0, sustained_walk_speed=0.3,
                   sustained_walk_hold_s=25.0)
    obj._sustained_walk_bucket(torch.arange(8))
    assert torch.allclose(obj.vel_command_b[:, 0], torch.full((8,), 0.3))
    assert torch.allclose(obj.vel_command_b[:, 1], torch.zeros(8))
    assert torch.allclose(obj.vel_command_b[:, 2], torch.zeros(8))
    assert not obj.is_standing_env.any(), (
        "a pinned env left marked standing would have its command zeroed by the standing mask"
    )
    assert torch.allclose(obj.time_left, torch.full((8,), 25.0)), (
        "the hold must outlast a 20 s episode, or the command changes mid-episode"
    )
    # the world-frame reference is refreshed from the body-frame command
    assert torch.allclose(obj.vel_command_w, obj.vel_command_b)


def test_bucket_selects_a_fraction_not_all():
    obj = _cmd_obj(n=2000, rel_sustained_walk_envs=0.2, sustained_walk_speed=0.3)
    obj._sustained_walk_bucket(torch.arange(2000))
    frac = float((obj.vel_command_b[:, 0] > 0.0).float().mean())
    assert 0.16 < frac < 0.24, f"expected ~20 % pinned, got {frac:.3f}"


def test_switch_defaults_off_and_is_wired(monkeypatch):
    import mjlab_microduck.tasks.microduck_velocity_env_cfg as cfg_mod

    monkeypatch.delenv("MICRODUCK_SUSTAINED_WALK", raising=False)
    assert cfg_mod.sustained_walk_fraction() == 0.0
    cfg = cfg_mod.make_microduck_velocity_env_cfg(play=False)
    assert cfg.commands["twist"].rel_sustained_walk_envs == 0.0

    monkeypatch.setenv("MICRODUCK_SUSTAINED_WALK", "0.2")
    monkeypatch.setenv("MICRODUCK_SUSTAINED_WALK_SPEED", "0.25")
    cfg2 = cfg_mod.make_microduck_velocity_env_cfg(play=False)
    assert cfg2.commands["twist"].rel_sustained_walk_envs == 0.2
    assert cfg2.commands["twist"].sustained_walk_speed == 0.25


# ── A1 remaining lever: the capability-paced spin target ──────────────────────────────────────────
# Measured (5-round demo): the spin policy reaches 0.60-1.22 rad/s against a 3.0 rad/s target with a
# std-1.5 Gaussian, and does so at ~70 deg of tilt. A target far beyond capability makes "spin fast"
# and "stay upright" non-complementary - being down is the more stable way to spin fast - so the arm
# asks for a rate the robot can hold.


def test_spin_rate_max_defaults_to_the_canonical_envelope(monkeypatch):
    import mjlab_microduck.tasks.microduck_spin_env_cfg as cfg_mod

    monkeypatch.delenv("MICRODUCK_SPIN_RATE_MAX", raising=False)
    assert cfg_mod._spin_rate_max() == 3.0
    cfg = cfg_mod.make_microduck_spin_env_cfg(play=False)
    assert cfg.rewards["spin_rate_track"].params["rate_max"] == 3.0


def test_spin_rate_max_arm_is_read_at_build_time(monkeypatch):
    """The envelope must NOT be frozen at import (that silently ignored the switch)."""
    import mjlab_microduck.tasks.microduck_spin_env_cfg as cfg_mod

    monkeypatch.setenv("MICRODUCK_SPIN_RATE_MAX", "1.0")
    cfg = cfg_mod.make_microduck_spin_env_cfg(play=False)
    # both the Gaussian tracker and its L1 bootstrap must agree, or the bootstrap pulls toward a
    # target the tracker no longer asks for
    assert cfg.rewards["spin_rate_track"].params["rate_max"] == 1.0
    assert cfg.rewards["spin_rate_l1"].params["rate_max"] == 1.0
    # the phase fractions are canonical and must not move with the rate
    assert cfg.rewards["spin_rate_track"].params["hold_end"] == 0.525


# ── A4: roller_crouch return-leg pricing ──────────────────────────────────────────────────────────
# The crouch rewards are per-step Gaussians on a phase-interpolated height target, i.e. an ANNUITY:
# a round that never completes the return still banks most of the cycle. Measured: 9/15 rounds pass
# over 3 seeds and the failures end at ~79 mm / g -0.40, i.e. on the return leg. A potential-based
# delta is unfarmable by parking, which is the property this test pins.


def _crouch_env():
    """Stub env for the crouch delta. The asset is built ONCE and shared: a fresh object per
    `scene[...]` access silently discards the pose the test sets (the same trap the earlier demo
    stubs hit)."""
    asset = types.SimpleNamespace(data=types.SimpleNamespace(root_link_pos_w=torch.zeros(1, 3)))

    class _Scene:
        terrain = types.SimpleNamespace(env_origins=torch.zeros(1, 3))

        def __getitem__(self, _name):
            return asset

    class _CM:
        def __init__(self):
            self.command = torch.zeros(1, 3)

        def get_command(self, _n):
            return self.command

    return types.SimpleNamespace(scene=_Scene(), command_manager=_CM(),
                                 episode_length_buf=torch.tensor([50]))


def test_crouch_delta_pays_for_progress_and_not_for_holding():
    """Rising toward the phase target pays; holding the same state pays exactly zero."""
    import math

    env = _crouch_env()
    kwargs = dict(command_name="twist", height_low=0.075, height_high=0.11, std=0.02)
    # phase 0.25 => cmd = (cos, sin) = (0, 1) => target = height_low (deep crouch)
    env.command_manager.command[0, 0] = 0.0
    env.command_manager.command[0, 1] = 1.0

    first = float(microduck_mdp.crouch_phase_progress_delta(env, **kwargs))
    assert first == 0.0, "the first call only seeds the state"

    # hold the same (wrong) pose: no progress -> zero
    for _ in range(5):
        held = float(microduck_mdp.crouch_phase_progress_delta(env, **kwargs))
    assert held == 0.0, "holding must pay nothing (otherwise it is farmable by parking)"

    # move the trunk to the commanded target: progress -> positive
    env.scene["robot"].data.root_link_pos_w[0, 2] = 0.075
    gained = float(microduck_mdp.crouch_phase_progress_delta(env, **kwargs))
    assert gained > 0.5, f"reaching the commanded height must pay, got {gained}"
    assert math.isfinite(gained)


def test_crouch_delta_switch_defaults_off(monkeypatch):
    import mjlab_microduck.tasks.microduck_roller_crouch_env_cfg as cfg_mod

    monkeypatch.delenv("MICRODUCK_CROUCH_DELTA", raising=False)
    assert cfg_mod._crouch_delta_weight() == 0.0
    cfg = cfg_mod.make_microduck_roller_crouch_env_cfg(play=False)
    assert "crouch_phase_delta" not in cfg.rewards

    monkeypatch.setenv("MICRODUCK_CROUCH_DELTA", "4.0")
    cfg2 = cfg_mod.make_microduck_roller_crouch_env_cfg(play=False)
    assert cfg2.rewards["crouch_phase_delta"].weight == 4.0
    # the existing crouch terms must survive, or the arm tests more than one thing
    for term in ("crouch_glide_pose", "crouch_glide_pose_l1", "upright", "body_ang_vel"):
        assert term in cfg2.rewards


# ── A4: the sitstand sustained-posture bucket ─────────────────────────────────────────────────────
# Audit (logs/sitstand_cmd_audit.txt): 4 of 5 failing demo rounds had a CONSTANT posture flag that the
# robot ignored. The dwell (3.5-6.5 s in a 20 s episode) leaves "flag constant AND opposite to spawn"
# at a few percent of episodes, so the bucket makes it common - and half of the pinned envs are
# commanded the posture OPPOSITE to their spawn, which is the case the failures actually show.


def _sitstand_cmd(n=2000, spawn_z=0.115, **cfg_attrs):
    obj = object.__new__(microduck_mdp.SitStandCommand)
    obj.cfg = types.SimpleNamespace(**cfg_attrs)
    obj._env = types.SimpleNamespace(device="cpu", num_envs=n)
    obj.vel_command_b = torch.zeros(n, 3)
    obj.vel_command_w = torch.zeros(n, 3)
    obj.vel_command_b[:, 0] = 0.0
    obj.time_left = torch.zeros(n)
    obj._sit_prob = 0.5
    obj._ramp_s = 2.0
    obj._sit_z = 0.060
    obj._stand_z = 0.115
    obj._alpha = torch.zeros(n)

    class _Asset:
        data = types.SimpleNamespace(
            root_link_pos_w=torch.tensor([[0.0, 0.0, spawn_z]] * n)
        )

    obj.robot = _Asset()
    obj._env_ref = types.SimpleNamespace(
        scene=types.SimpleNamespace(terrain=types.SimpleNamespace(env_origins=torch.zeros(n, 3)))
    )
    return obj


def test_sitstand_hold_bucket_is_off_by_default(monkeypatch):
    import mjlab_microduck.tasks.microduck_sitstand_env_cfg as cfg_mod

    monkeypatch.delenv("MICRODUCK_SITSTAND_HOLD", raising=False)
    assert cfg_mod._hold_env_fraction() == 0.0
    cfg = cfg_mod.make_microduck_sitstand_env_cfg(play=False)
    assert cfg.commands["twist"].rel_hold_envs == 0.0


def test_sitstand_hold_bucket_switch_is_wired(monkeypatch):
    import mjlab_microduck.tasks.microduck_sitstand_env_cfg as cfg_mod

    monkeypatch.setenv("MICRODUCK_SITSTAND_HOLD", "0.3")
    cfg = cfg_mod.make_microduck_sitstand_env_cfg(play=False)
    assert cfg.commands["twist"].rel_hold_envs == 0.3
    assert cfg.commands["twist"].hold_s >= 20.0, "the hold must outlast a 20 s episode"


def test_sitstand_hold_bucket_pins_one_command_and_often_the_opposite():
    obj = _sitstand_cmd(rel_hold_envs=1.0, hold_s=25.0, spawn_z=0.115)   # spawned STANDING
    obj._resample_command(torch.arange(2000))
    assert torch.allclose(obj.time_left, torch.full((2000,), 25.0)), "no mid-episode resample"
    assert set(obj.vel_command_b[:, 0].unique().tolist()) <= {0.0, 1.0}
    frac_opposite = float((obj.vel_command_b[:, 0] > 0.5).float().mean())
    assert 0.4 < frac_opposite < 0.6, (
        f"half the pinned envs must be commanded the OPPOSITE posture, got {frac_opposite:.2f}"
    )

    obj2 = _sitstand_cmd(rel_hold_envs=1.0, hold_s=25.0, spawn_z=0.060)  # spawned SITTING
    obj2._resample_command(torch.arange(2000))
    frac_stand = float((obj2.vel_command_b[:, 0] < 0.5).float().mean())
    assert 0.4 < frac_stand < 0.6, "and the polarity must follow the spawn state"


def test_sitstand_hold_bucket_off_leaves_the_dwell_alone():
    obj = _sitstand_cmd(rel_hold_envs=0.0, hold_s=25.0)
    obj.time_left = torch.full((2000,), 4.0)
    obj._resample_command(torch.arange(2000))
    assert torch.allclose(obj.time_left, torch.full((2000,), 4.0)), (
        "with the bucket off the dwell timer must not be overwritten"
    )
