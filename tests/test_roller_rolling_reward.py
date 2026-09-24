"""Rolling-consistency reward (`mdp.wheel_rolling_reward`) — the A/B that replaced wheel-spin.

`wheel_speed_reward` pays for forward wheel ROTATION whatever the body does, so "spin the wheels
with the body parked" is its argmax. `wheel_rolling_reward` pays for translation *coupled* to that
rotation — a distance with a slip penalty. These tests lock the behaviours the distinction rests on:
parking pays 0, free-spinning pays 0, genuine rolling pays and scales with speed, the cap saturates,
the command gate closes, and the whole thing is NaN-safe.

They also lock the env-var A/B switch (`MICRODUCK_WHEEL_ROLLING`), because an A/B whose arms are not
actually different is worse than no A/B: the first version of this experiment paid ~0.004 of raw
reward because `slip_std` was 0.06 against a measured slip of 0.09 (invisible gradient), which is
why the calibrated default is asserted here.
"""

import types

import torch

import mjlab_microduck.tasks.mdp as microduck_mdp

_WHEEL_JOINT_IDS = {
    "passive_LF_?wheel": 2,
    "passive_LR_?wheel": 5,
    "passive_RF_?wheel": 9,
    "passive_RR_?wheel": 12,
}
_WHEEL_RADIUS = 0.0175


def _env(cmd_x=0.3, v_fwd=0.0, omega=0.0, nan_wheel=False):
    """Minimal stand-in for the manager env: only what the reward touches."""
    joint_vel = torch.zeros(1, 14)
    for j in _WHEEL_JOINT_IDS.values():
        joint_vel[:, j] = omega
    if nan_wheel:
        joint_vel[:, 5] = float("nan")

    class _Asset:
        data = types.SimpleNamespace(
            joint_vel=joint_vel,
            root_link_lin_vel_b=torch.tensor([[v_fwd, 0.0, 0.0]]),
        )

        def find_joints(self, name):
            return [_WHEEL_JOINT_IDS[name]], None

    class _Scene:
        def __getitem__(self, _key):
            return _Asset()

    class _CommandManager:
        def get_command(self, _name):
            return torch.tensor([[cmd_x, 0.0, 0.0]])

    return types.SimpleNamespace(scene=_Scene(), command_manager=_CommandManager())


def _reward(env, **kwargs):
    kwargs.setdefault("command_name", "twist")
    return microduck_mdp.wheel_rolling_reward(env, **kwargs).item()


def _omega_for(surface_speed):
    return surface_speed / _WHEEL_RADIUS


def test_parked_pays_nothing():
    """v = 0 and wheels still: the wheel/ground agreement is perfect, and it must still pay 0."""
    assert _reward(_env(v_fwd=0.0, omega=0.0)) == 0.0


def test_free_spinning_wheels_pay_nothing():
    """Wheels at 0.35 m/s of surface speed with the body parked: the burnout case -> 0."""
    assert _reward(_env(v_fwd=0.0, omega=_omega_for(0.35))) == 0.0


def test_genuine_rolling_pays_the_distance():
    """Zero slip at 0.30 m/s against a 0.35 cap -> cmd_x * v/cap."""
    value = _reward(_env(cmd_x=0.3, v_fwd=0.30, omega=_omega_for(0.30)))
    assert abs(value - 0.3 * (0.30 / 0.35)) < 1e-5


def test_speed_scales_the_reward_monotonically():
    slow = _reward(_env(v_fwd=0.10, omega=_omega_for(0.10)))
    fast = _reward(_env(v_fwd=0.30, omega=_omega_for(0.30)))
    assert 0.0 < slow < fast


def test_slip_is_penalised_but_not_cliffed():
    """A 0.25 m/s mismatch (the free-spin signature) must collapse, not merely dip."""
    clean = _reward(_env(v_fwd=0.30, omega=_omega_for(0.30)))
    slipping = _reward(_env(v_fwd=0.30, omega=_omega_for(0.55)))
    assert slipping < 0.25 * clean


def test_cap_saturates_so_overspeed_is_not_farmed():
    at_cap = _reward(_env(cmd_x=0.3, v_fwd=0.35, omega=_omega_for(0.35)))
    beyond = _reward(_env(cmd_x=0.3, v_fwd=1.20, omega=_omega_for(1.20)))
    assert abs(at_cap - beyond) < 1e-5


def test_deadband_and_command_gate_close():
    """Below the deadband the env is parked, not rolling; a non-forward command pays nothing."""
    assert _reward(_env(v_fwd=0.005, omega=_omega_for(0.005))) == 0.0
    assert _reward(_env(cmd_x=0.0, v_fwd=0.30, omega=_omega_for(0.30))) == 0.0
    assert _reward(_env(cmd_x=-0.3, v_fwd=0.30, omega=_omega_for(0.30))) == 0.0


def test_nan_safe():
    value = _reward(_env(v_fwd=0.30, omega=_omega_for(0.30), nan_wheel=True))
    assert torch.isfinite(torch.tensor(value))


def test_switch_is_off_by_default():
    import mjlab_microduck.tasks.microduck_velocity_rollers_env_cfg as cfg_mod

    assert cfg_mod._wheel_rolling() is False
    assert cfg_mod._rolling_slip_std() == 0.18


def test_switch_on_swaps_wheel_speed_for_wheel_rolling(monkeypatch):
    import mjlab_microduck.tasks.microduck_velocity_rollers_env_cfg as cfg_mod

    monkeypatch.setenv("MICRODUCK_WHEEL_ROLLING", "1")
    cfg = cfg_mod.make_microduck_velocity_rollers_env_cfg(play=False)
    assert cfg.rewards["wheel_speed"].weight == 0.0, "the spin term must be silent in this arm"
    assert cfg.rewards["wheel_rolling"].weight == 10.0, "same weight, different objective"
    assert cfg.rewards["wheel_rolling"].params["slip_std"] == 0.18
    assert cfg.rewards["wheel_rolling"].params["cap_speed"] == 0.35


def test_switch_off_keeps_the_original_recipe(monkeypatch):
    import mjlab_microduck.tasks.microduck_velocity_rollers_env_cfg as cfg_mod

    monkeypatch.delenv("MICRODUCK_WHEEL_ROLLING", raising=False)
    cfg = cfg_mod.make_microduck_velocity_rollers_env_cfg(play=False)
    assert cfg.rewards["wheel_speed"].weight == 10.0
    assert "wheel_rolling" not in cfg.rewards


def test_swizzle_inherits_the_switch(monkeypatch):
    """swizzle derives from the rollers base, so the A/B must reach it (it is arm 2)."""
    import mjlab_microduck.tasks.microduck_velocity_swizzle_env_cfg as sw_mod

    monkeypatch.setenv("MICRODUCK_WHEEL_ROLLING", "1")
    cfg = sw_mod.make_microduck_velocity_swizzle_env_cfg(play=False)
    assert "wheel_rolling" in cfg.rewards
    assert cfg.rewards["wheel_speed"].weight == 0.0


# ── posture arm: the trunk-height band ────────────────────────────────────────────────────────────
# `com_height_target` reads `root_link_pos_w` (trunk), NOT the CoM, and the original band
# (0.0935-0.1235) tops out 14.5 mm below the measured 138 mm roller stand - so it charged the very
# posture the family is named for, and inside the band the cheapest end won. These tests pin the
# default (so the baseline stays reproducible) and the arm's override.


def test_height_band_defaults_to_the_original_recipe(monkeypatch):
    import mjlab_microduck.tasks.microduck_velocity_rollers_env_cfg as cfg_mod

    monkeypatch.delenv("MICRODUCK_ROLLER_HEIGHT_BAND", raising=False)
    assert cfg_mod._height_band() == (0.0935, 0.1235)
    cfg = cfg_mod.make_microduck_velocity_rollers_env_cfg(play=False)
    assert cfg.rewards["com_height_target"].params["target_height_min"] == 0.0935


def test_height_band_arm_brackets_the_measured_roller_stand(monkeypatch):
    import mjlab_microduck.tasks.microduck_velocity_rollers_env_cfg as cfg_mod

    monkeypatch.setenv("MICRODUCK_ROLLER_HEIGHT_BAND", "0.130,0.145")
    cfg = cfg_mod.make_microduck_velocity_rollers_env_cfg(play=False)
    params = cfg.rewards["com_height_target"].params
    assert params["target_height_min"] == 0.130
    assert params["target_height_max"] == 0.145
    # the measured roller stand (ROLLER_STAND_Z = 0.138) must be INSIDE the band, not above it
    assert params["target_height_min"] < 0.138 < params["target_height_max"]


def test_com_height_target_measures_the_trunk_not_the_com():
    """The name lies; the test documents which quantity the band actually constrains."""
    import inspect

    src = inspect.getsource(microduck_mdp.com_height_target)
    body = src.split('"""')[2] if src.count('"""') >= 2 else src   # drop the (explanatory) docstring
    assert "root_link_pos_w" in body
    assert "root_com_pos_w" not in body


# ── no-skate arm: retire the blade-skating terms ──────────────────────────────────────────────────


def test_noskate_switch_removes_the_three_skating_terms(monkeypatch):
    import mjlab_microduck.tasks.microduck_velocity_rollers_env_cfg as cfg_mod

    monkeypatch.delenv("MICRODUCK_ROLLERS_NOSKATE", raising=False)
    assert cfg_mod._rollers_noskate() is False
    base = cfg_mod.make_microduck_velocity_rollers_env_cfg(play=False)
    for term in ("single_support", "skating_air_time", "glide"):
        assert term in base.rewards, f"{term} must exist in the baseline recipe"

    monkeypatch.setenv("MICRODUCK_ROLLERS_NOSKATE", "1")
    arm = cfg_mod.make_microduck_velocity_rollers_env_cfg(play=False)
    for term in ("single_support", "skating_air_time", "glide"):
        assert term not in arm.rewards
    # everything that is NOT a skating term must survive, or the arm tests more than one thing
    for term in ("upright", "pose", "wheel_speed", "braking", "gait_symmetry", "forward_lean",
                 "com_height_target", "heading_hold"):
        assert term in arm.rewards, f"{term} must survive the no-skate arm"
