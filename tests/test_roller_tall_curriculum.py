"""Two-stage roller curriculum (`mdp.height_band_curriculum`) — optimization_plan.md step 4/B.

The wheeled family has two mutually exclusive gaits (measured, `logs/family_rollers.md` round 4):
crouched + rolling travels at 117 % of the commanded push but sits at 116 mm, while the named roller
stand (138.9 mm) needs the skating cycle and only reaches 70 %. Swapping the recipe at step 0
destroys both (15 %). The two-stage recipe therefore keeps the crouched band with the skating terms
muted until the rolling gait exists, then LERPS the band up to the measured stand while fading the
skating terms back in.

These tests pin the default (off, so the baseline recipes stay reproducible), the arm's schedule, and
the interpolation - including the scalar return, which mjlab requires (`CurriculumManager` calls
`.item()` on the returned state and raises on a 2-element tensor).
"""

import types

import torch

import mjlab_microduck.tasks.mdp as microduck_mdp

BAND_STAGES = [
    {"step": 0, "band": (0.0935, 0.1235)},
    {"step": 48000, "band": (0.130, 0.145)},
]


def _curric_env(step):
    params = {"target_height_min": 0.0, "target_height_max": 0.0}

    class _RM:
        def get_term_cfg(self, _name):
            return types.SimpleNamespace(params=params)

    return types.SimpleNamespace(common_step_counter=step, reward_manager=_RM()), params


def _run(step):
    env, params = _curric_env(step)
    state = microduck_mdp.height_band_curriculum(
        env, None, reward_name="com_height_target", band_stages=BAND_STAGES
    )
    return state, params


def test_returns_a_scalar_state():
    state, _ = _run(0)
    assert isinstance(state, torch.Tensor) and state.numel() == 1, (
        "CurriculumManager does term_state.item() - a 2-element tensor raises at reset"
    )


def test_stage_one_is_the_original_crouched_band():
    _, params = _run(0)
    assert params["target_height_min"] == 0.0935
    assert params["target_height_max"] == 0.1235


def test_stage_two_brackets_the_measured_roller_stand():
    _, params = _run(48000)
    assert params["target_height_min"] == 0.130
    assert params["target_height_max"] == 0.145
    assert params["target_height_min"] < 0.138 < params["target_height_max"], (
        "the measured stand (ROLLER_STAND_Z = 0.138) must end up INSIDE the band"
    )


def test_the_band_interpolates_rather_than_steps():
    """Halfway through the ramp the band must be halfway - a step would be a moving discontinuity."""
    _, params = _run(24000)
    assert abs(params["target_height_min"] - 0.11175) < 1e-6
    assert abs(params["target_height_max"] - 0.13425) < 1e-6
    # and strictly between the two stages
    assert 0.0935 < params["target_height_min"] < 0.130
    assert 0.1235 < params["target_height_max"] < 0.145


def test_switch_is_off_by_default(monkeypatch):
    import mjlab_microduck.tasks.microduck_velocity_rollers_env_cfg as cfg_mod

    monkeypatch.delenv("MICRODUCK_ROLLER_TALL_STAGE2", raising=False)
    assert cfg_mod._tall_curriculum_stage2() == 0.0
    cfg = cfg_mod.make_microduck_velocity_rollers_env_cfg(play=False)
    assert "com_height_band" not in cfg.curriculum, "the default recipe must not gain a band ramp"
    assert "glide_weight" not in cfg.curriculum


def test_arm_schedules_the_ramp_and_the_skating_fade_in(monkeypatch):
    import mjlab_microduck.tasks.microduck_velocity_rollers_env_cfg as cfg_mod

    monkeypatch.setenv("MICRODUCK_ROLLER_TALL_STAGE2", "2000")
    cfg = cfg_mod.make_microduck_velocity_rollers_env_cfg(play=False)

    band = cfg.curriculum["com_height_band"].params["band_stages"]
    assert band[0]["step"] == 0 and band[0]["band"] == (0.0935, 0.1235)
    assert band[1]["step"] == 2000 * 24 and band[1]["band"] == (0.130, 0.145)

    for name, full in (("single_support", 3.0), ("skating_air_time", 1.5), ("glide", 4.0)):
        stages = cfg.curriculum[f"{name}_weight"].params["weight_stages"]
        assert stages[0]["weight"] == 0.0, f"{name} must be muted in stage 1"
        assert stages[1] == {"step": 2000 * 24, "weight": 0.0}
        assert stages[2]["weight"] == full, f"{name} must reach its full weight in stage 2"
        assert stages[2]["step"] == 2500 * 24, "the fade-in must lag the band ramp"


def test_the_two_stage_switch_is_independent_of_the_other_roller_switches(monkeypatch):
    import mjlab_microduck.tasks.microduck_velocity_rollers_env_cfg as cfg_mod

    monkeypatch.setenv("MICRODUCK_ROLLER_TALL_STAGE2", "1200")
    monkeypatch.delenv("MICRODUCK_WHEEL_ROLLING", raising=False)
    monkeypatch.delenv("MICRODUCK_ROLLERS_NOSKATE", raising=False)
    cfg = cfg_mod.make_microduck_velocity_rollers_env_cfg(play=False)
    assert "com_height_band" in cfg.curriculum
    assert cfg.rewards["wheel_speed"].weight == 10.0, "the rolling switch must stay independent"
    assert "glide" in cfg.rewards, "the skating terms must stay PRESENT (ramped), not deleted"


# ── step 5b: the roller_standup structural backstop ───────────────────────────────────────────────
# This family had NO termination that recycles a parked episode (only nan_state was left after
# fell_over was popped), so a round that never leaves the ground farms the whole 6 s - measured in
# the 5-round demo, one round stalled at 80 mm for its entire episode.


def test_roller_standup_stall_defaults_off(monkeypatch):
    import mjlab_microduck.tasks.microduck_roller_standup_env_cfg as cfg_mod

    monkeypatch.delenv("MICRODUCK_STALL_TILT_G", raising=False)
    cfg = cfg_mod.make_microduck_roller_standup_env_cfg(play=False)
    assert cfg.terminations["recovery_stall"].params["tilt_clause_g"] is None
    assert cfg.terminations["recovery_stall"].params["threshold_z"] == 0.10


def test_roller_standup_stall_takes_the_shared_clause(monkeypatch):
    import mjlab_microduck.tasks.microduck_roller_standup_env_cfg as cfg_mod

    monkeypatch.setenv("MICRODUCK_STALL_TILT_G", "-0.9")
    cfg = cfg_mod.make_microduck_roller_standup_env_cfg(play=False)
    params = cfg.terminations["recovery_stall"].params
    assert params["tilt_clause_g"] == -0.9
    # the z threshold must sit BELOW the roller stand (0.138) and above a lying robot, or the rule
    # either punishes standing or never fires
    assert 0.06 < params["threshold_z"] < 0.138


def test_roller_standup_keeps_nan_state_and_drops_fell_over():
    """fell_over must stay popped (the robot STARTS fallen) - the new rule is the only backstop."""
    import mjlab_microduck.tasks.microduck_roller_standup_env_cfg as cfg_mod

    cfg = cfg_mod.make_microduck_roller_standup_env_cfg(play=False)
    assert "fell_over" not in cfg.terminations
    assert "nan_state" in cfg.terminations
    assert "recovery_stall" in cfg.terminations


# ── step 5b: the near-wall spawn port ─────────────────────────────────────────────────────────────
# StandUp spreads its face-up starts +/-90 deg about the long axis, i.e. partway along the roll - a
# built-in reverse curriculum for the direction it cannot otherwise reach. RollerStandUp had no such
# spread, and the demo showed a round that never left the ground.


def test_roller_standup_roll_spawn_defaults_off(monkeypatch):
    import mjlab_microduck.tasks.microduck_roller_standup_env_cfg as cfg_mod

    monkeypatch.delenv("MICRODUCK_ROLLER_STANDUP_FACEUP_ROLL", raising=False)
    cfg = cfg_mod.make_microduck_roller_standup_env_cfg(play=False)
    assert cfg.events["set_ground_state"].params["face_up_roll_max"] == 0.0


def test_roller_standup_roll_spawn_arm_spreads_partway(monkeypatch):
    import math

    import mjlab_microduck.tasks.microduck_roller_standup_env_cfg as cfg_mod

    monkeypatch.setenv("MICRODUCK_ROLLER_STANDUP_FACEUP_ROLL", "90")
    cfg = cfg_mod.make_microduck_roller_standup_env_cfg(play=False)
    spread = cfg.events["set_ground_state"].params["face_up_roll_max"]
    assert abs(math.degrees(spread) - 90.0) < 1e-6
    # the port must not disturb the rest of the mix
    params = cfg.events["set_ground_state"].params
    assert params["face_down_prob"] == 0.50 and params["standing_prob"] == 0.50
