"""The two floor-flip levers for the stand-up family — mechanism tests, not config cosmetics.

Context (measured 2026-09-22, `logs/family_plan.md` part 3): from a forced floor spawn the policy
rises to 113 mm and PARKS at 34 deg of tilt, giving a floor-flip rate of 0/184. Two independent
causes were found and each gets one arm:

* STRUCTURE — `recovery_stall_termination`'s only clause was `z < 0.09`, and the park is at 0.113 m,
  so the rule never fired (1/256) and a 3,000-iteration retrain with the fixed `progress_eps`
  changed the rate by exactly nothing. The tilt clause is what makes the rule able to see it.
* REWARD — the two terms that can price the final 30 deg are curriculum-ANNEALED (1.0 -> 0.2 and
  1.5 -> 0.5), i.e. the goal price is weakest exactly where the policy gets stuck.

These tests pin both the defaults (so the baseline stays reproducible) and the arms, and they
exercise the tilt clause on the ACTUAL parked state rather than on a generic low state.
"""

import types

import torch

import mjlab_microduck.tasks.mdp as microduck_mdp


def _stall_env(z_m, g_z, n=1):
    """Minimal env for `recovery_stall_termination`: pose + episode counter.

    The asset is built ONCE and shared, so a test can mutate the pose between calls and see the
    progress logic respond (a fresh tensor per access would silently freeze the state).
    """
    asset = types.SimpleNamespace(
        data=types.SimpleNamespace(
            root_link_pos_w=torch.tensor([[0.0, 0.0, z_m]] * n),
            projected_gravity_b=torch.tensor([[0.0, 0.0, g_z]] * n),
        )
    )

    class _Scene:
        def __getitem__(self, _name):
            return asset

    return types.SimpleNamespace(scene=_Scene(), episode_length_buf=torch.full((n,), 50))


def test_parked_half_stand_is_invisible_to_the_original_rule():
    """The measured park: z = 113 mm, g = -0.83 (34 deg). Height-only rule must never fire."""
    env = _stall_env(0.113, -0.83)
    fired = False
    for _ in range(300):
        fired = fired or bool(
            microduck_mdp.recovery_stall_termination(env, threshold_z=0.09, stall_steps=100).any()
        )
    assert not fired, "the height-only rule cannot see the park - that was the whole bug"


def test_tilt_clause_fires_on_the_parked_half_stand():
    """Same state, tilt clause on: it must terminate once the stall counter fills."""
    env = _stall_env(0.113, -0.83)
    fires = [
        bool(
            microduck_mdp.recovery_stall_termination(
                env, threshold_z=0.09, stall_steps=100, tilt_clause_g=-0.9
            ).any()
        )
        for _ in range(120)
    ]
    assert not any(fires[:99]), "must not fire before stall_steps"
    first = fires.index(True)
    assert first in (99, 100), f"expected to fire at the stall_steps boundary, fired at {first}"


def test_tilt_clause_leaves_a_standing_env_alone():
    """A genuinely upright env (g = -1, z = 113 mm) must never be terminated by this rule."""
    env = _stall_env(0.113, -1.0)
    for _ in range(300):
        assert not microduck_mdp.recovery_stall_termination(
            env, threshold_z=0.09, stall_steps=100, tilt_clause_g=-0.9
        ).any()


def test_tilt_clause_does_not_fire_while_the_robot_is_still_rising():
    """Rising resets the counter: a slow rise must survive the tilt clause."""
    env = _stall_env(0.09, -0.5)
    fired = False
    for i in range(300):
        # keep making progress: z climbs 0.5 mm and the tilt improves every 40 steps
        env.scene["robot"].data.root_link_pos_w[:, 2] = 0.09 + i * 0.0005
        env.scene["robot"].data.projected_gravity_b[:, 2] = -0.5 - i * 0.001
        fired = fired or bool(
            microduck_mdp.recovery_stall_termination(
                env, threshold_z=0.09, stall_steps=100, tilt_clause_g=-0.9
            ).any()
        )
    assert not fired, "a making-progress episode must never be stalled out"


# ── arm switches ──────────────────────────────────────────────────────────────────────────────────


def test_stall_tilt_clause_defaults_off(monkeypatch):
    import mjlab_microduck.tasks.microduck_standup_env_cfg as cfg_mod

    monkeypatch.delenv("MICRODUCK_STALL_TILT_G", raising=False)
    assert cfg_mod._stall_tilt_g() is None
    cfg = cfg_mod.make_microduck_standup_env_cfg(play=False)
    assert cfg.terminations["recovery_stall"].params["tilt_clause_g"] is None


def test_structure_arm_sets_the_tilt_clause(monkeypatch):
    import mjlab_microduck.tasks.microduck_standup_env_cfg as cfg_mod

    monkeypatch.setenv("MICRODUCK_STALL_TILT_G", "-0.9")
    cfg = cfg_mod.make_microduck_standup_env_cfg(play=False)
    assert cfg.terminations["recovery_stall"].params["tilt_clause_g"] == -0.9


def _sharp_last_stage(cfg, name):
    return cfg.curriculum[f"{name}_weight"].params["weight_stages"][-1]["weight"]


def test_sharp_weights_are_annealed_by_default(monkeypatch):
    """The bug: the last-mile price SHRINKS over training (1.0 -> 0.2 and 1.5 -> 0.5)."""
    import mjlab_microduck.tasks.microduck_standup_env_cfg as cfg_mod

    monkeypatch.delenv("MICRODUCK_STANDUP_SHARP_HOLD", raising=False)
    cfg = cfg_mod.make_microduck_standup_env_cfg(play=False)
    assert _sharp_last_stage(cfg, "height_stand_sharp") == 0.2
    assert _sharp_last_stage(cfg, "upright_sharp") == 0.5


def test_reward_arm_inverts_the_anneal(monkeypatch):
    import mjlab_microduck.tasks.microduck_standup_env_cfg as cfg_mod

    monkeypatch.setenv("MICRODUCK_STANDUP_SHARP_HOLD", "1")
    cfg = cfg_mod.make_microduck_standup_env_cfg(play=False)
    for name, stages_expected in (("height_stand_sharp", (1.0, 1.5, 2.0)),
                                  ("upright_sharp", (1.5, 2.0, 3.0))):
        got = tuple(s["weight"] for s in cfg.curriculum[f"{name}_weight"].params["weight_stages"])
        assert got == stages_expected, f"{name}: expected a rising schedule, got {got}"


def test_the_two_arms_are_independent(monkeypatch):
    """Each arm must change exactly one thing, or the A/B is confounded."""
    import mjlab_microduck.tasks.microduck_standup_env_cfg as cfg_mod

    monkeypatch.setenv("MICRODUCK_STALL_TILT_G", "-0.9")
    monkeypatch.delenv("MICRODUCK_STANDUP_SHARP_HOLD", raising=False)
    structure = cfg_mod.make_microduck_standup_env_cfg(play=False)
    assert structure.terminations["recovery_stall"].params["tilt_clause_g"] == -0.9
    assert _sharp_last_stage(structure, "upright_sharp") == 0.5, "structure arm must not touch reward"

    monkeypatch.delenv("MICRODUCK_STALL_TILT_G", raising=False)
    monkeypatch.setenv("MICRODUCK_STANDUP_SHARP_HOLD", "1")
    reward = cfg_mod.make_microduck_standup_env_cfg(play=False)
    assert reward.terminations["recovery_stall"].params["tilt_clause_g"] is None
    assert _sharp_last_stage(reward, "upright_sharp") == 3.0, "reward arm must not touch structure"


# ── port to VelStand (2026-09-23) ─────────────────────────────────────────────────────────────────
# Same switch, same rule: VelStand's `recovery_stall` had the identical height-only blind spot, and
# its economy has an explicit 25-40 deg dead zone (fallen = tilt > 40, up = tilt < 25, no recycling
# in between) which is exactly where a recovery policy parks.


def test_velstand_stall_tilt_clause_defaults_off(monkeypatch):
    import mjlab_microduck.tasks.microduck_velstand_env_cfg as cfg_mod

    monkeypatch.delenv("MICRODUCK_STALL_TILT_G", raising=False)
    assert cfg_mod._stall_tilt_g() is None
    cfg = cfg_mod.make_microduck_velstand_env_cfg(play=False)
    assert cfg.terminations["recovery_stall"].params["tilt_clause_g"] is None


def test_velstand_port_sets_the_same_clause(monkeypatch):
    import mjlab_microduck.tasks.microduck_velstand_env_cfg as cfg_mod

    monkeypatch.setenv("MICRODUCK_STALL_TILT_G", "-0.9")
    cfg = cfg_mod.make_microduck_velstand_env_cfg(play=False)
    assert cfg.terminations["recovery_stall"].params["tilt_clause_g"] == -0.9
    # the clause must be strictly between "fallen" (>40 deg) and "recovered" (<25 deg), i.e. it has
    # to cover the dead zone on BOTH sides of it
    import math

    clause_tilt = math.degrees(math.acos(min(1.0, -(-0.9))))
    assert cfg_mod.RECOVERED_UP_TILT_DEG < clause_tilt < cfg_mod.REWARD_GATE_TILT_DEG


def test_velstand_dead_zone_is_what_the_clause_covers():
    """Documents the audit: `fallen` and `recovered` do not overlap, so a 33 deg park is neither."""
    import mjlab_microduck.tasks.microduck_velstand_env_cfg as cfg_mod

    assert cfg_mod.RECOVERED_UP_TILT_DEG == 25.0
    assert cfg_mod.REWARD_GATE_TILT_DEG == 40.0
    assert cfg_mod.TERM_GATE_TILT_DEG == 40.0
    assert cfg_mod.RECOVERED_UP_TILT_DEG < 33.0 < cfg_mod.REWARD_GATE_TILT_DEG, (
        "the measured park (33 deg) sits inside the dead zone - that is the whole finding"
    )
