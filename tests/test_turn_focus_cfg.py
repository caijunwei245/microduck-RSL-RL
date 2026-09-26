"""The in-place-turn bonus (`MICRODUCK_TURN_FOCUS`) and the tail it targets.

Motivating measurement (`logs/turn_tail_findings.md`, 2026-09-26): with the wobble tax removed the
turn policy reaches gain ~1.15 at a commanded 0.3 rad/s but in a fraction of episodes **stands
still** - upright, never falling, simply not rotating. Instrumenting every episode showed the cause
is not a bad draw: pinning the spawn pose, or every per-episode DR group, MOVES the failure rather
than removing it, because the two behaviours pay almost the same. Over matched episodes the
stand-still basin scored **240.401** against the turning policy's **241.184** (-0.32 %): not moving
already collects the full `track_linear_velocity` reward (the commanded linear velocity is zero),
pays 4.6 less `angular_wobble`, and is slightly more upright. Only the yaw term separates them.

So `mdp.turn_focus_bonus` pays the yaw Gaussian a second time, gated to envs that are in place with
a real turn commanded. These tests lock: the switch default (off), the gate's mask, and that a
gated-off env contributes exactly zero (a weight-0 term must not perturb the running recipe).
"""

import types

import torch

import mjlab_microduck.tasks.mdp as microduck_mdp


def _stub_env(cmd: torch.Tensor, w_z: torch.Tensor, name: str = "robot"):
    """Minimal env: only `scene[<name>].data.root_link_ang_vel_b` and the command manager."""
    data = types.SimpleNamespace(root_link_ang_vel_b=torch.stack(
        [torch.zeros_like(w_z), torch.zeros_like(w_z), w_z], dim=1))
    entity = types.SimpleNamespace(data=data)
    cm = types.SimpleNamespace(get_command=lambda _n: cmd)
    return types.SimpleNamespace(scene={name: entity}, command_manager=cm)


def _cmd(vx, vy, wz):
    return torch.tensor([[vx, vy, wz]], dtype=torch.float32)


def test_switch_defaults_to_off(monkeypatch):
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import turn_focus_weight
    monkeypatch.delenv("MICRODUCK_TURN_FOCUS", raising=False)
    assert turn_focus_weight() == 0.0, "default must leave the running recipe untouched"
    monkeypatch.setenv("MICRODUCK_TURN_FOCUS", "3.5")
    assert turn_focus_weight() == 3.5


def test_term_is_registered_with_zero_weight_by_default():
    from mjlab.tasks.registry import load_env_cfg
    import mjlab_microduck.tasks  # noqa: F401
    cfg = load_env_cfg("Mjlab-Velocity-Flat-MicroDuck", play=False)
    assert "turn_focus" in cfg.rewards
    assert cfg.rewards["turn_focus"].weight == 0.0
    assert cfg.rewards["turn_focus"].func is microduck_mdp.turn_focus_bonus


def test_gate_pays_an_in_place_turn():
    env = _stub_env(_cmd(0.0, 0.0, 0.3), torch.tensor([0.3]))
    out = microduck_mdp.turn_focus_bonus(env, std=0.5)
    assert float(out) > 0.99, "in place, turning exactly as commanded -> full payment"


def test_gate_is_silent_while_walking_or_standing():
    # walking with a small yaw correction must NOT be paid by this term (that is what
    # track_angular_velocity is for), and neither must standing still
    for cmd in (_cmd(0.3, 0.0, 0.3), _cmd(0.0, 0.0, 0.0), _cmd(0.0, 0.0, 0.2)):
        env = _stub_env(cmd, torch.tensor([0.3]))
        assert float(microduck_mdp.turn_focus_bonus(env, std=0.5)) == 0.0, cmd


def test_error_still_costs_like_the_tracking_term():
    # same Gaussian as track_yaw_velocity: a policy that is not turning gets almost nothing
    env = _stub_env(_cmd(0.0, 0.0, 0.3), torch.tensor([0.0]))
    out = float(microduck_mdp.turn_focus_bonus(env, std=0.5))
    assert out < 0.75, f"commanded 0.3, achieved 0.0 -> exp(-0.36) = {out:.3f}"
