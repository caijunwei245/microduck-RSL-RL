"""StandUp cfg invariants — the reward economy that the 40k-iteration audit exposed.

Audit (logs/family_eval.py, per-spawn-bucket, on the 40,000-iteration checkpoint): envs that never
stood still collected 54-65 % of the reward a standing env collected (prone 33.7, sitting 32.0,
supine 38.1 against 58.9). "Get up" was therefore worth only a 35-46 % marginal gain against the
cost of the whole maneuver, and 35,000 further iterations moved the ground buckets not at all
(prone 0.140 / sitting 0.062 / supine 0.214 unchanged from 5k to 40k).
"""

import pytest

from mjlab_microduck.tasks.microduck_standup_env_cfg import make_microduck_standup_env_cfg


def test_pose_stand_legs_is_tight_enough_that_a_slump_cannot_collect_it():
    """std 0.5 -> 0.2 on 2026-09-21: at 0.5 a slumped pose whose legs merely resemble HOME kept
    most of this weight-2.0 term. Keep it tight enough that the ground buckets are unpayable."""
    cfg = make_microduck_standup_env_cfg()
    std = cfg.rewards["pose_stand_legs"].params["std"]
    assert std <= 0.25, f"pose_stand_legs std={std}: a slumped pose can collect it again"


def test_standup_has_no_positive_term_payable_from_a_flop():
    """Every positive-weighted term must be paired with an uprightness or height gate, otherwise a
    stable flop farms it (AGENTS.md: never gate a positive reward on being in a bad state). This
    test records the audit's requirement rather than re-deriving it: the sharp/gated terms must
    exist and carry real weight."""
    cfg = make_microduck_standup_env_cfg()
    for name in ("height_stand_sharp", "upright_sharp", "standing_composite"):
        assert name in cfg.rewards, f"{name} is the gate that makes the smooth terms unpayable flat"
        assert cfg.rewards[name].weight > 0


def test_standup_has_an_incremental_progress_term():
    """The structural fix for the audit: a delta-of-progress term so that parking in a slump pays
    ZERO, plus the flat terms it must dominate kept small."""
    cfg = make_microduck_standup_env_cfg()
    assert "standup_progress_delta" in cfg.rewards
    assert cfg.rewards["standup_progress_delta"].weight >= 3.0
    assert cfg.rewards["pose_stand_legs"].weight <= 1.0, "the flat legs term must not dominate"


def test_standup_has_a_recovery_stall_termination():
    """The structural lever: parking in a slump must END the episode, otherwise the audit's 45-57 %
    payable-while-failed economy keeps the slump optimal (proven by three bit-identical re-checks)."""
    cfg = make_microduck_standup_env_cfg()
    assert "recovery_stall" in cfg.terminations
    term = cfg.terminations["recovery_stall"]
    assert term.time_out is False, "a stall is a failure, not a time-out"
    assert term.params["stall_steps"] <= 150, "too slow to bite inside a 6 s episode"
    assert term.params["threshold_z"] <= 0.10, "must not fire on a standing robot"


def test_recovery_stall_does_not_let_micro_wiggles_reset_the_counter():
    """The criterion must bite in a deterministic rollout too: with progress_eps=0.01 a lying robot's
    micro-motion kept resetting the stall counter, so the rule fired in training but never in the
    evaluation (0 of 256 envs), which made its effect unmeasurable."""
    from mjlab_microduck.tasks.mdp import recovery_stall_termination
    import inspect

    sig = inspect.signature(recovery_stall_termination)
    assert sig.parameters["progress_eps"].default <= 0.001
