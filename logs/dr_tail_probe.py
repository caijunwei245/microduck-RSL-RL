"""Which DR draw makes the turn policy stand still? Per-episode instrumentation.

The tail found in `logs/turn_ship_verdict.md`: the wobble-free turn policy tracks a commanded
0.3 rad/s at gain ~1.15 in most episodes but in some **stands still** (upright, g -1.00, never
falls), and an independently trained wobble-free policy fails the SAME episode indices with nearly
the same numbers. Two policies agreeing on which episodes are bad points at the episodes rather
than at the weights - so this records, for every episode, both the OUTCOME and the DR values it was
played under, then ranks the factors by how well they separate the failures.

**It drives `skill_demo.run_round` itself** (via its `probe` hook) rather than re-implementing the
rollout loop. That is not a style choice: a hand-rolled copy of the loop reproduced a completely
different result for episodes 5-6 (0.232/0.233 instead of 0.073/0.042 rad/s), i.e. it hid the very
tail it existed to explain. Same code, or no conclusion.

Reading the velocity cfg's event table first narrows the search:

    startup (drawn ONCE per env, so they cannot explain episode-to-episode differences)
        foot_friction           feet geom friction, abs 0.7-1.3
        encoder_bias            joint_pos bias, +-0.015 rad
        base_com                trunk CoM offset, +-25/25/30 mm
        randomize_mass_inertia  trunk pseudo-inertia, alpha +-0.025
    reset (per episode)
        reset_base              x,y +-0.5 m, z 0.12-0.13 m, yaw +-pi
        randomize_com           trunk CoM +-3 mm, head CoM +-3 mm
        randomize_joint_friction  BAM friction scale 0.9-1.1
        randomize_armature      dof armature scale 0.9-1.1
    interval                    push_robot, +-0.3 m/s in x,y

Usage:  uv run python logs/dr_tail_probe.py [--rounds 24] [--seed 0] [--cmd 0.3] [--ckpts a,b]
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys

import numpy as np
import torch

import mjlab_microduck.tasks  # noqa: F401  (registrations + patches)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from skill_demo import Actor, run_round  # noqa: E402  (same rollout loop as the demos)

DEVICE = "cuda:0"
TASK = "Mjlab-Velocity-Flat-MicroDuck"
FEET_GEOMS = (26, 72)          # left/right foot collision geoms (from the cfg's foot_friction event)
TRUNK_BODY = 0

PROBE_KEYS = ("steps", "yaw_mean", "z_end", "g_end", "upright_frac", "fell",
              "r_track_lin", "r_track_yaw", "r_upright", "r_wobble", "r_total")


def _to_np(x):
    """warp array / torch tensor -> numpy (warp arrays reject scalar indexing)."""
    t = x.to("cpu") if hasattr(x, "to") else x
    if hasattr(t, "numpy"):
        t = t.numpy()
    return np.asarray(t)


def dr_vector(env) -> dict[str, float]:
    """Everything readable that varies per episode (plus the startup ones, to prove they do not)."""
    out: dict[str, float] = {}
    m = env.sim.model.struct
    r = env.scene["robot"]
    fr = np.asarray(_to_np(m.geom_friction))[0][list(FEET_GEOMS)]
    for i, v in enumerate(np.asarray(fr).reshape(len(FEET_GEOMS), -1).mean(axis=1)):
        out[f"foot_friction[{i}]"] = float(v)
    ipos = np.asarray(_to_np(m.body_ipos)[TRUNK_BODY]).reshape(-1)
    for ax, v in zip("xyz", ipos[:3]):
        out[f"trunk_com_{ax}"] = float(v)
    arm = np.asarray(_to_np(m.dof_armature)[0]).reshape(-1)
    out["armature_mean"] = float(arm.mean()) if arm.size else float("nan")
    damp = np.asarray(_to_np(m.dof_damping)[0]).reshape(-1)
    out["damping_mean"] = float(damp.mean()) if damp.size else float("nan")
    # BAM friction scale lives on the custom actuators (mdp.randomize_bam_friction writes it)
    scales = []
    for a in getattr(r, "actuators", []) or []:
        for name in ("friction_scale", "_friction_scale"):
            s = getattr(a, name, None)
            if s is not None:
                vals = np.asarray(_to_np(s)).reshape(-1)
                if vals.size:
                    scales.extend(float(v) for v in vals)
                break
    out["friction_scale_mean"] = float(np.mean(scales)) if scales else float("nan")
    # firmware PD gain scales (mdp.randomize_delayed_actuator_gains, mode="reset" -> EVERY episode).
    # These live on the actuator, not the model, which is exactly why the first --pin-all missed them.
    for attr, key in (("kp_scale", "kp_scale"), ("kd_scale", "kd_scale"),
                      ("_kp_scale", "kp_scale"), ("_kd_scale", "kd_scale")):
        if key in out:
            continue
        vals = []
        for a in getattr(r, "actuators", []) or []:
            v = getattr(a, attr, None)
            if v is not None:
                arr = np.asarray(_to_np(v)).reshape(-1)
                if arr.size:
                    vals.extend(float(x) for x in arr)
        if vals:
            out[key] = float(np.mean(vals))
    # spawn state: what the episode starts from
    rp = np.asarray(_to_np(r.data.root_link_pos_w[0])).reshape(-1)
    for ax, v in zip("xyz", rp[:3]):
        out[f"spawn_{ax}"] = float(v)
    if hasattr(r.data, "root_link_quat_w"):
        q = np.asarray(_to_np(r.data.root_link_quat_w[0])).reshape(-1)
        if q.size >= 4:
            w, x, y, z = (float(v) for v in q[:4])
            out["spawn_yaw_deg"] = float(np.degrees(np.arctan2(2 * (w * z + x * y),
                                                               1 - 2 * (y * y + z * z))))
    jp = np.asarray(_to_np(r.data.joint_pos[0])).reshape(-1)
    out["joint_pos_abs_mean"] = float(np.abs(jp).mean()) if jp.size else float("nan")
    # what the policy SEES vs the truth: the encoder-bias view is an obs, so under bias they differ
    try:
        jb = np.asarray(_to_np(r.data.joint_pos_biased[0])).reshape(-1)
        if jb.size and jp.size == jb.size:
            out["encoder_bias_abs_mean"] = float(np.abs(jb - jp).mean())
    except Exception:  # noqa: BLE001
        pass
    return out


def state_snapshot(env) -> dict:
    """Everything the FIRST observation of an episode is built from, plus the actuator internals.

    With every DR draw pinned the inputs are identical, so any spread across episodes in here IS the
    carried state - that is the whole point of the measurement.
    """
    out: dict = {}
    r = env.scene["robot"]
    for name in ("joint_pos", "joint_pos_biased", "joint_vel", "root_link_pos_w", "root_link_quat_w",
                 "root_link_ang_vel_b", "projected_gravity_b", "joint_pos_target",
                 "joint_effort_target"):
        v = getattr(r.data, name, None)
        if hasattr(v, "detach"):
            out[f"data.{name}"] = v.detach().float().cpu().numpy().reshape(-1).copy()
    am = getattr(env, "action_manager", None)
    for name in ("action", "_action", "prev_action", "action_history"):
        v = getattr(am, name, None) if am is not None else None
        if hasattr(v, "detach"):
            out[f"action_manager.{name}"] = v.detach().float().cpu().numpy().reshape(-1).copy()
    for i, a in enumerate(getattr(r, "actuators", []) or []):
        for name, v in vars(a).items():
            if hasattr(v, "detach") and getattr(v, "dtype", None) is not None:
                try:
                    arr = v.detach().float().cpu().numpy().reshape(-1).copy()
                except Exception:  # noqa: BLE001
                    continue
                if arr.size <= 4096:
                    out[f"act{i}.{name}"] = arr
        for name in ("_delay_buffer", "delay_buffer"):
            buf = getattr(a, name, None)
            inner = getattr(buf, "_buffer", None) if buf is not None else None
            if hasattr(inner, "detach"):
                out[f"act{i}.delay_buffer_contents"] = (
                    inner.detach().float().cpu().numpy().reshape(-1).copy())
    return out


def reset_actuator_delay(env) -> int:
    """Re-initialise the BAM actuator's action-delay buffer (the suspected carried state).

    `FrictionDRBamActuator` keeps `_delay_buffer` to emulate the command chain's lag, and no
    reset-mode event touches it - so after an in-place episode reset the new episode replays up to
    `max_lag` control steps of the PREVIOUS episode's commands.
    """
    n = 0
    for a in getattr(env.scene["robot"], "actuators", []) or []:
        for name in ("_init_delay_buffer", "reset"):
            fn = getattr(a, name, None)
            if callable(fn) and name == "_init_delay_buffer":
                try:
                    fn()
                    n += 1
                except Exception:  # noqa: BLE001
                    pass
                break
    return n


def spearman(xs, ys):
    """Rank correlation without scipy; ties get their average rank."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out
    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


# which event terms each --pin group switches off (midpoint-pinned / dropped)
PIN_GROUPS: dict[str, tuple[str, ...]] = {
    "spawn": ("reset_base",),
    "com": ("randomize_com", "randomize_head_com", "base_com"),
    "friction": ("randomize_joint_friction",),
    "armature": ("randomize_armature",),
    "damping": ("randomize_joint_damping",),
    "gains": ("randomize_motor_gains",),
    "orientation": ("randomize_base_orientation",),
    "mass": ("randomize_mass_inertia",),
}
PIN_KEYS = ("ranges", "scale_range", "bias_range", "pose_range", "velocity_range",
            "position_range", "alpha_range", "kp_range", "kd_range")


def _midpoint(v):
    """Pin a sampled range to its midpoint; pass anything else through unchanged."""
    if isinstance(v, dict):
        return {k: _midpoint(x) for k, x in v.items()}
    if isinstance(v, (tuple, list)) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
        m = (float(v[0]) + float(v[1])) / 2.0
        return (m, m)
    return v


def apply_pins(env_cfg, groups: set[str]) -> None:
    """Pin the DR draws BEFORE the env is built (the managers deepcopy their cfg at init)."""
    if not groups:
        return
    targeted: set[str] = set()
    for g in groups:
        if g == "all":
            targeted.update(PIN_GROUPS)
            continue
        targeted.add(g)
    for g in sorted(targeted):
        for term in PIN_GROUPS.get(g, ()):
            if term not in env_cfg.events:
                continue
            cfg = env_cfg.events[term]
            for k in PIN_KEYS:
                if k in cfg.params:
                    cfg.params[k] = _midpoint(cfg.params[k])
    if "all" in groups or "push" in groups:
        env_cfg.events.pop("push_robot", None)


def make_env(pins: str):
    env_cfg = load_env_cfg(TASK, play=False)
    env_cfg.scene.num_envs = 1
    groups = {p.strip() for p in pins.split(",") if p.strip() and p.strip() != "none"}
    if "push" in groups:
        groups.add("push")
    apply_pins(env_cfg, groups)
    return env_cfg


def run_policy(ckpt: str, rounds: int, seed: int, cmd: float, pins: str = "none",
               fresh_env: bool = False, reset_delay: bool = False,
               dump_state: bool = False) -> tuple[list[dict], list[dict]]:
    torch.manual_seed(seed)          # identical DR stream for every policy compared
    env_cfg = make_env(pins)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=DEVICE)
    max_steps = int(round(env_cfg.episode_length_s /
                          (env_cfg.sim.mujoco.timestep * env_cfg.decimation)))
    agent_cfg = load_rl_cfg(TASK)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    policy = Actor(ckpt, clip=getattr(agent_cfg, "clip_actions", None))
    spec = ("velocity turn", TASK, ckpt, "turn", 0.0, cmd, None, None)
    out = []
    states: list[dict] = []
    for _ in range(rounds):
        if fresh_env:
            # DECISIVE CONTROL for "carried state": a brand-new env per episode. Nothing can survive
            # a reset that is not written by the reset events themselves (actuator delay queues,
            # action history, solver warm-start data are the usual suspects).
            del env, wrapped
            env = ManagerBasedRlEnv(cfg=make_env(pins), device=DEVICE)
            wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        def _probe(e, _reset=reset_delay):
            if _reset:
                reset_actuator_delay(e)
            if dump_state:
                states.append(state_snapshot(e))
            return dr_vector(e)

        res = run_round(env, wrapped, policy, spec, max_steps, probe=_probe)
        # episode reward sums (bare term names) - the two basins' PAYOFFS, which is what decides
        # between them. `_episode_sums` is keyed by bare term name, not "Episode_Reward/<term>".
        sums = res.get("sums") or {}
        if fresh_env:
            pass
        out.append({**res.get("probe", {}),
                    "yaw_mean": res["yaw_abs_mean"], "z_end": res["z_last"], "g_end": res["g_last"],
                    "upright_frac": res["upright_frac"], "fell": res["fell"], "steps": res["steps"],
                    "r_track_lin": sums.get("track_linear_velocity", float("nan")),
                    "r_track_yaw": sums.get("track_angular_velocity", float("nan")),
                    "r_upright": sums.get("upright", float("nan")),
                    "r_wobble": sums.get("angular_wobble", float("nan")),
                    "r_total": sum(sums.values()) if sums else float("nan")})
    del env
    return out, states


def report_state_spread(states: list[dict], weak_idx: list[int]) -> None:
    """Which snapshot field differs between episodes? With DR pinned, any spread is carried state."""
    if not states:
        return
    keys = sorted({k for st in states for k in st})
    print()
    print("############ initial-state spread across episodes (with DR pinned this IS the residue) ############")
    print(f"{'field':34s} {'max|v|':>10s} {'spread':>10s} {'weak eps differ?':>18s}")
    for k in keys:
        vals = [st.get(k) for st in states]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        import numpy as _np
        stack = _np.stack([v.reshape(-1) for v in vals])
        spread = float(_np.abs(stack - stack[0]).max())
        if spread <= 1e-9:
            continue
        weakdev = float(_np.abs(_np.stack([states[i][k].reshape(-1) for i in weak_idx]) -
                               stack[0]).max()) if weak_idx else 0.0
        print(f"{k:34s} {float(_np.abs(stack).max()):10.4f} {spread:10.4f} {weakdev:18.4f}")


def report(rows: dict[str, list[dict]], cmd: float, seed: int, pins: str = "none",
           fresh_env: bool = False, states: list[dict] | None = None) -> None:
    tags = list(rows)
    print("############ per-episode DR vs turn outcome (skill_demo.run_round driving) ############")
    print(f"cmd {cmd} rad/s in place, seed {seed}, {len(rows[tags[0]])} episodes per policy, "
          f"pinned DR groups: {pins}, fresh env per episode: {fresh_env}")
    print("policies: " + " | ".join(tags))
    print()
    print(f"{'ep':>3s} " + " ".join(f"{t[-20:]:>22s}" for t in tags))
    for i in range(len(rows[tags[0]])):
        cells = [f"{rows[t][i]['yaw_mean']:6.3f} / z{rows[t][i]['z_end']:3.0f} / "
                 f"g{rows[t][i]['g_end']:+.2f}" for t in tags]
        bad = "   <-- stand-still" if any(rows[t][i]["yaw_mean"] < 0.15 for t in tags) else ""
        print(f"{i + 1:>3d} " + " ".join(f"{c:>22s}" for c in cells) + bad)
    print("    (cell = |yaw| rad/s, steady-state window / terminal trunk z mm / terminal gravity z)")

    for t in tags:
        rs = rows[t]
        weak = [r for r in rs if r["yaw_mean"] < 0.15]
        strong = [r for r in rs if r["yaw_mean"] >= 0.15]
        print()
        print(f"=== {t}")
        print(f"    {len(weak)}/{len(rs)} episodes below 0.15 rad/s; median "
              f"{statistics.median(r['yaw_mean'] for r in rs):.3f}; "
              f"min upright_frac {min(r['upright_frac'] for r in rs):.2f}; "
              f"any fell {any(r['fell'] for r in rs)}")
        if not weak or not strong:
            print("    no weak/strong split to report - the table below is the per-episode spread")
        else:
            print(f"    {'payoff (episode sum)':24s} {'weak':>10s} {'strong':>10s}   delta")
            for k in ("r_total", "r_track_lin", "r_track_yaw", "r_upright", "r_wobble"):
                wv = [r[k] for r in weak if r.get(k) == r.get(k)]
                sv = [r[k] for r in strong if r.get(k) == r.get(k)]
                if wv and sv:
                    print(f"    {k:24s} {statistics.fmean(wv):10.3f} {statistics.fmean(sv):10.3f}   "
                          f"{statistics.fmean(wv) - statistics.fmean(sv):+.3f}")
        keys = [k for k in rs[0] if k not in PROBE_KEYS]
        print(f"    {'factor':24s} {'weak':>10s} {'strong':>10s} {'min':>10s} {'max':>10s} {'rho':>6s}")
        for k in keys:
            allv = [r[k] for r in rs if r.get(k) == r.get(k)]
            allp = [r["yaw_mean"] for r in rs if r.get(k) == r.get(k)]
            wv = [r[k] for r in weak if r.get(k) == r.get(k)]
            sv = [r[k] for r in strong if r.get(k) == r.get(k)]
            if not allv:
                continue
            rho = spearman(allv, allp) if len(allv) > 3 else float("nan")
            wm = statistics.fmean(wv) if wv else float("nan")
            sm = statistics.fmean(sv) if sv else float("nan")
            flag = ""
            if wv and sv and (wm < min(sv) - 1e-9 or wm > max(sv) + 1e-9):
                flag = "  <== separates"
            const = "  (constant)" if max(allv) - min(allv) < 1e-9 else ""
            print(f"    {k:24s} {wm:10.5f} {sm:10.5f} {min(allv):10.5f} {max(allv):10.5f} "
                  f"{rho:6.2f}{flag}{const}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", default="logs/rsl_rl/velocity/2026-09-26_19-46-52_turn03_woboff/"
                                         "model_4998.pt,logs/rsl_rl/velocity/"
                                         "2026-09-17_00-03-38_yawonly_long/model_63998.pt",
                    help="comma-separated checkpoints; each gets the SAME DR stream")
    ap.add_argument("--rounds", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cmd", type=float, default=0.3)
    ap.add_argument("--dump-state", action="store_true",
                    help="snapshot the initial state of every episode and print which fields differ "
                         "between episodes (the residue, with DR pinned)")
    ap.add_argument("--reset-delay", action="store_true",
                    help="re-init the BAM actuator action-delay buffer at every episode start "
                         "(the suspected carried state)")
    ap.add_argument("--fresh-env", action="store_true",
                    help="build a NEW env for every episode - the control that separates carried "
                         "state from the policy")
    ap.add_argument("--pin", default="none",
                    help="comma list of per-episode DR groups to pin to their midpoints "
                         "(spawn,com,friction,armature,damping,gains,orientation,mass,push,all) - "
                         "the control that separates 'the draw' from 'the env state'")
    args = ap.parse_args()

    rows = {}
    states: list[dict] = []
    for ckpt in [c.strip() for c in args.ckpts.split(",") if c.strip()]:
        tag = os.path.basename(os.path.dirname(ckpt))
        rows[tag], states = run_policy(ckpt, args.rounds, args.seed, args.cmd, args.pin,
                                       args.fresh_env, args.reset_delay, args.dump_state)
    report(rows, args.cmd, args.seed, args.pin, args.fresh_env, states)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
