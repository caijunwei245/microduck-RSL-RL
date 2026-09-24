"""BallKick cfg invariants — including the reward economy that silently drifted.

The kick reward is an economy, not a sum: a capped linear payoff that peaks at
``BALL_TARGET_SPEED`` plus an overshoot penalty, sized so that

* the at-target payoff is ≈ +3/step — enough to pay for the swing's transient
  pose/upright cost against the ~8/step standing stack (the pre-taming weight
  3.0 gave 0.75/step and was too weak to justify the swing);
* the landscape peaks at the target while erring hard stays much cheaper than
  not kicking: net reward reaches 0 only at 4× the target speed.

Commit 2a0b1b0 ("left or right handed ball kick") raised ``BALL_TARGET_SPEED``
0.25 -> 1.0 **without rescaling the weights**, although its own comment three
lines below said to: the shipped economy became a 12/step payoff (4× the
documented one) with the net-zero strike at ~4 m/s instead of ~1 m/s. Nothing
caught it, because nothing tested the *relationship* between the constants and
the weights. These tests do, so that edit class cannot pass again.
"""

import importlib.util
from pathlib import Path

import mujoco
import pytest

from mjlab_microduck.robot.microduck_constants import MICRODUCK_BALL_CFG
from mjlab_microduck.tasks import microduck_ball_kick_env_cfg as kick_cfg
from mjlab_microduck.tasks.microduck_ball_kick_env_cfg import (
    BALL_AT_TARGET_PAYOFF,
    BALL_FORWARD_WEIGHT,
    BALL_OFFSET_ABS_Y,
    BALL_OFFSET_X,
    BALL_POS_NOISE_XY,
    BALL_RADIUS,
    BALL_TARGET_SPEED,
    ENABLE_SYMMETRY,
    KICK_FOOT,
    make_microduck_ball_kick_env_cfg,
)

REPO = Path(__file__).resolve().parents[1]
BALL_XML = REPO / "src" / "mjlab_microduck" / "robot" / "microduck" / "ball.xml"
# Foot geometry measured at HOME (see the constants block: foot centers at
# (0, ±0.042), toe tip x ≈ 0.034).
TOE_TIP_X = 0.034


def _kick_env():
    return make_microduck_ball_kick_env_cfg()


# ── The reward economy ────────────────────────────────────────────────────────
def test_kick_at_target_payoff_is_the_documented_three_per_step():
    """The failure mode of 2a0b1b0: a target change that no weight followed."""
    cfg = _kick_env()
    fwd = cfg.rewards["ball_forward_velocity"]
    target = BALL_TARGET_SPEED

    # Capped term saturates exactly at the target, so its payoff is weight*target.
    assert fwd.params["max_speed"] == target
    assert fwd.weight == pytest.approx(BALL_FORWARD_WEIGHT)
    assert fwd.weight == pytest.approx(BALL_AT_TARGET_PAYOFF / target)

    payoff = fwd.weight * target
    assert payoff == pytest.approx(BALL_AT_TARGET_PAYOFF, rel=1e-6)
    assert 2.0 <= payoff <= 4.0, (
        f"at-target payoff is {payoff:.2f}/step; the design is ≈+3/step. "
        "Changing BALL_TARGET_SPEED requires rescaling the weights with it."
    )


def test_kick_payoff_is_sized_against_the_standing_stack():
    """~+3/step is 30-60% of the standing stack — weaker never justified the
    swing, stronger makes 'kick as hard as possible' the argmax again."""
    cfg = _kick_env()
    ball_terms = {"ball_forward_velocity", "ball_speed_overshoot"}
    standing_stack = sum(
        t.weight for n, t in cfg.rewards.items() if n not in ball_terms and t.weight > 0
    )
    assert standing_stack > 0
    share = (cfg.rewards["ball_forward_velocity"].weight * BALL_TARGET_SPEED) / standing_stack
    assert 0.30 <= share <= 0.60, (
        f"at-target payoff is {share:.0%} of the {standing_stack:.1f}/step standing stack"
    )


def test_kick_overshoot_landscape_peaks_at_target_with_a_gentler_far_side():
    cfg = _kick_env()
    over = cfg.rewards["ball_speed_overshoot"]
    assert over.weight < 0, "the overshoot term is a penalty: negative weight"
    assert over.params["target_speed"] == BALL_TARGET_SPEED

    payoff = cfg.rewards["ball_forward_velocity"].weight * BALL_TARGET_SPEED
    # Net (payoff - |w_over| * (v - target)) hits zero here: the speed at which
    # kicking too hard finally costs as much as not kicking at all.
    zero_speed = BALL_TARGET_SPEED + payoff / abs(over.weight)
    assert zero_speed == pytest.approx(4.0 * BALL_TARGET_SPEED, rel=0.02)
    # Asymmetric slopes: the optimum is the target, not zero and not max speed.
    assert abs(over.weight) < cfg.rewards["ball_forward_velocity"].weight


def test_kick_reward_stack_signs_and_walking_terms_removed():
    cfg = _kick_env()
    for name in (
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "pose",
        "soft_landing",
    ):
        assert name not in cfg.rewards, f"{name} is walking-specific, dropped by this task"
    # House convention: a penalty carries a negative weight (microduck_*_penalty
    # functions return <= 0), a positive term a positive one.
    for name, term in cfg.rewards.items():
        if name == "ball_speed_overshoot":
            continue
        if term.weight < 0:
            assert any(k in name for k in ("rate", "limit", "collision", "momentum", "ang_vel")), name
    assert cfg.rewards["ball_forward_velocity"].weight > 0
    assert cfg.rewards["support_foot_grounded"].weight > 0


# ── Actor/critic asymmetry: the actor must stay blind to the ball ─────────────
def test_kick_actor_is_ball_blind_and_critic_sees_the_ball():
    cfg = _kick_env()
    actor_terms = cfg.observations["actor"].terms
    critic_terms = cfg.observations["critic"].terms

    assert not [n for n in actor_terms if "ball" in n], (
        "the real robot has no ball sensing — a ball obs in the actor is untransferable"
    )
    for term in ("ball_position", "ball_velocity"):
        assert term in critic_terms
        assert critic_terms[term].params["asset_name"] == "ball"
    assert "ball_position" not in actor_terms


def test_kick_command_slots_keep_the_unified_layout():
    """head/body slots are zero-padded but ALIVE (obs-width parity + live weights)."""
    cfg = _kick_env()
    for group in ("actor", "critic"):
        assert cfg.observations[group].terms["head_command"].params["dim"] == 4
        assert cfg.observations[group].terms["body_command"].params["dim"] == 6

    cmd = cfg.commands["twist"]
    assert cmd.heading_command is False and cmd.ranges.heading is None
    # Tiny but non-zero: dead inputs never come back for a later curriculum.
    for rng in (cmd.ranges.lin_vel_x, cmd.ranges.lin_vel_y, cmd.ranges.ang_vel_z):
        assert rng is not None and rng[0] < 0 < rng[1]
    assert cmd.resampling_time_range == (kick_cfg.EPISODE_LENGTH_S, 2 * kick_cfg.EPISODE_LENGTH_S)


# ── Footedness: the flag must flip spawn AND support sensor together ─────────
@pytest.mark.parametrize(("foot", "sign"), [("right", -1), ("left", +1)])
def test_kick_foot_flag_flips_ball_spawn_and_support_sensor(foot, sign):
    cfg = make_microduck_ball_kick_env_cfg(kick_foot=foot)
    offset = cfg.events["reset_ball"].params["offset"]
    assert offset == (BALL_OFFSET_X, sign * BALL_OFFSET_ABS_Y)

    support = "left" if foot == "right" else "right"
    sensor = next(s for s in cfg.scene.sensors if s.name == "support_foot_ground_contact")
    assert sensor.primary.pattern == rf"^{support}_foot_collision$"


def test_kick_rejects_an_unknown_foot():
    with pytest.raises(AssertionError):
        make_microduck_ball_kick_env_cfg(kick_foot="middle")


def test_kick_default_foot_is_right():
    assert KICK_FOOT == "right"


# ── Ball asset: constants, XML and the rehearsal must agree ──────────────────
def test_kick_ball_asset_matches_the_constants():
    model = mujoco.MjModel.from_xml_path(str(BALL_XML))
    assert model.nq == 7, "the ball is a free-floating prop (freejoint)"
    assert model.nu == 0, "the ball is unactuated"
    assert model.geom_size[0][0] == pytest.approx(BALL_RADIUS)
    assert model.body_mass[1] == pytest.approx(0.015)
    assert MICRODUCK_BALL_CFG.init_state.pos[2] == pytest.approx(BALL_RADIUS)


def test_kick_spawn_never_penetrates_the_toe():
    """The bug the constants block documents: at 0.08 ± 0.02 the solver ejected
    the ball at reset, paying a 'kick' reward for no kick. Worst case here is
    BALL_OFFSET_X - noise_x - radius >= toe tip + clearance."""
    noise_x = BALL_POS_NOISE_XY[0] if isinstance(BALL_POS_NOISE_XY, (tuple, list)) else BALL_POS_NOISE_XY
    worst_rear_surface = BALL_OFFSET_X - noise_x - BALL_RADIUS
    assert worst_rear_surface - TOE_TIP_X >= 0.005


def test_kick_placement_distribution_demands_a_real_strike():
    """2026-09-19: the strike used to graze the ball at the edge of its reach —
    a 15 mm shift took the deployment rehearsal from 0.109 to 0.216 m/s, and a
    40 ms delay change from 0.000 to 0.290. The placement distribution is the
    lever: x noise is tight (depth must be reached) while y keeps the aiming
    error the blind actor must survive."""
    noise_x, noise_y = BALL_POS_NOISE_XY
    assert noise_x < noise_y, "x is depth, y is aiming error — do not merge them"
    # The band must reach the deepest safe contact (clearance rule above) ...
    assert BALL_OFFSET_X - noise_x <= 0.074 + 1e-9
    # ... and still cover the placement the old recipe used, so nothing that used
    # to be in-distribution has been dropped from the far side.
    assert BALL_OFFSET_X + noise_x >= 0.090 - 1e-9


def test_kick_ball_placement_matches_the_deployment_rehearsal():
    """infer_policy.py duplicates these three numbers to place the ball when a
    kick is triggered ('must match ... reset_ball_in_front_of_foot'); a drift
    would rehearse a kick the policy never trained for."""
    spec = importlib.util.spec_from_file_location(
        "infer_policy", REPO / "scripts" / "infer_policy.py"
    )
    ip = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ip)
    assert ip.BALL_OFFSET_X == BALL_OFFSET_X
    assert ip.BALL_OFFSET_ABS_Y == BALL_OFFSET_ABS_Y
    assert ip.BALL_RADIUS == BALL_RADIUS


# ── Wiring that the task depends on ──────────────────────────────────────────
def test_kick_actuator_lag_is_coherent_not_dithered():
    """mjlab's DelayBuffer re-draws the lag EVERY step when delay_update_period is 0
    (with hold_prob 0), so the shared actuator cfg's "3-6 step delay" is a dither
    around ~4.5 steps — never a coherent latency. The robot has a coherent one, and
    for a 0.2 s one-shot swing that difference is the whole ball speed: the same
    checkpoint in the same training env scores 0.2433/0.2448 m/s with the dither and
    0.1215/0.1404 with the lag held (logs/kick_r1_report.md §34). This task must
    hold each env's lag for an episode; other tasks stay as they are."""
    cfg = _kick_env()
    act = cfg.scene.entities["robot"].articulation.actuators[0]
    assert (act.delay_min_lag, act.delay_max_lag) == (3, 6)
    assert act.delay_update_period >= 50 * kick_cfg.EPISODE_LENGTH_S, (
        "the actuator lag must be held for at least one episode, otherwise training "
        "dithered its own latency and the deployed (coherent) latency is untrained"
    )
    from mjlab_microduck.robot.microduck_constants import actuators as shared

    assert shared.delay_update_period == 0, (
        "this fix is for the kick task only — do not silently change every task's "
        "actuator latency model"
    )


def test_kick_ball_spawn_event_runs_after_the_robot_pose_is_set():
    """Events run in insertion order and the ball position derives from the
    robot's final pose — reset_ball must come after set_ground_state."""
    events = list(_kick_env().events)
    assert events.index("reset_ball") > events.index("set_ground_state")


def test_kick_trains_from_the_deployment_hand_off_stand():
    """The kick is triggered by an ONNX hand-off mid-episode, so the policy starts
    from the walking policy's settled stand, not from HOME+noise. That stand is
    0.13 rad off HOME (2.6x the ±0.05 reset noise); without this event the trained
    policy stood still in the deployment rehearsal (0.0005 m/s) while kicking
    normally from a HOME reset (0.244 m/s) — see logs/kick_r1_report.md §7."""
    cfg = _kick_env()
    assert "settled_stand" in cfg.events
    term = cfg.events["settled_stand"]
    assert term.func is kick_cfg.microduck_mdp.reset_joints_to_settled_stand
    assert term.mode == "reset"

    p = term.params
    assert len(p["offset"]) == 14, "one deviation per actuated joint, in model order"
    assert 0.0 < p["prob"] < 1.0, "keep a share of plain HOME resets in the data"
    lo, hi = p["scale_range"]
    assert 0.0 < lo < 1.0 <= hi, "the range must span HOME-ish through the measured stand"
    assert p["position_range"] == (-0.05, 0.05)

    # The point of the event: an initial state far outside the noise it trained with
    # (measured max 0.132 rad = 2.6x the ±0.05 uniform reset noise).
    worst = max(abs(v) for v in p["offset"])
    assert worst > 2.5 * p["position_range"][1]
    # It is a stance shift, not a head wobble: support-leg hip pitch (2) and
    # kicking-leg hip roll (10) carry it. Re-measure logs/kick_obs_walk.csv if either
    # stops dominating — it means the walking policy or the hand-off changed.
    assert abs(p["offset"][2]) > 0.10 and abs(p["offset"][10]) > 0.10

    # Order matters: after the upright ground state writes the joints, before the
    # ball is placed from the final pose.
    events = list(cfg.events)
    assert events.index("set_ground_state") < events.index("settled_stand")
    assert events.index("settled_stand") < events.index("reset_ball")


def test_kick_robot_is_the_full_collision_model_and_stays_first():
    cfg = _kick_env()
    entities = list(cfg.scene.entities)
    assert entities[0] == "robot", "reset events write the robot root at qpos[:, 0:7]"
    assert entities == ["robot", "ball"]
    # The kick robot is the standup (full-collision) MODEL with one override: the
    # coherent per-episode actuator lag (see the dedicated test below).
    robot = cfg.scene.entities["robot"]
    assert robot.spec_fn is kick_cfg.MICRODUCK_STANDUP_ROBOT_CFG.spec_fn
    assert robot.init_state is kick_cfg.MICRODUCK_STANDUP_ROBOT_CFG.init_state
    assert cfg.sim.nconmax >= 50, "ball-terrain + ball-robot contacts need headroom"


def test_kick_standalone_env_keeps_the_bam_friction_event():
    """BAM computes friction in the actuator: without this startup event the
    friction fields never expand and the DR is a silent no-op."""
    assert "expand_bam_friction_fields" in _kick_env().events


def test_kick_episode_is_a_standing_start_and_nan_guarded():
    cfg = _kick_env()
    assert cfg.episode_length_s == 5.0
    ground = cfg.events["set_ground_state"].params
    assert ground["standing_prob"] == 1.0
    assert ground["face_down_prob"] == ground["face_up_prob"] == ground["sitting_prob"] == 0.0
    assert cfg.events["reset_robot_joints"].params["position_range"] == (-0.05, 0.05)
    assert "nan_state" in cfg.terminations and "fell_over" in cfg.terminations


def test_kick_curriculum_action_rate_ramp_is_the_velocity_one():
    """The swing is a fast one-shot: at 1000 iterations the penalty sits at -0.6,
    not the converged -1.0 (the documented first knob if the kick comes out weak)."""
    stages = _kick_env().curriculum["action_rate_weight"].params["weight_stages"]
    assert [(s["step"] // 24, s["weight"]) for s in stages] == [
        (0, -0.1), (500, -0.2), (750, -0.4), (1000, -0.6), (1250, -0.8), (1500, -1.0),
    ]


def test_kick_symmetry_stays_off():
    """The task is inherently one-footed: mirroring the 61D obs would ask for a
    left-footed kick from a right-footed run."""
    assert ENABLE_SYMMETRY is False
    # The mirror loss lives in the runner cfg, not the env cfg.
    assert kick_cfg.MicroduckBallKickRlCfg.algorithm.symmetry_cfg is None


def test_kick_task_and_backlash_twin_are_registered():
    import mjlab_microduck.tasks  # noqa: F401  (populates the registry)
    from mjlab.tasks.registry import list_tasks

    tasks = list_tasks()
    assert "Mjlab-BallKick-Flat-MicroDuck" in tasks
    assert "Mjlab-BallKick-Flat-Backlash-MicroDuck" in tasks
    assert kick_cfg.MicroduckBallKickRlCfg.experiment_name == f"ball_kick_{KICK_FOOT}"
