"""Is the environment deterministic across resets? (2026-09-27)

`logs/dr_tail_findings.md` left one thing unexplained: with every DR draw pinned, six episodes in one
env start from *identical* states (a 20+ field snapshot showed zero spread) and still diverge - most
turn at 0.33-0.37 rad/s, one stands still at 0.047. Before blaming "carried state" one more time, this
removes the policy from the equation entirely:

    reset -> apply the SAME fixed action sequence -> record the state;  repeat N times;  diff.

Any difference between repetitions is the simulator (or the env's own plumbing) being nondeterministic
across resets. Two runs, one with DR and one with the play cfg (no DR at all).

Usage:  uv run python logs/sim_determinism_test.py [--steps 200] [--trials 4] [--task ...]
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

import mjlab_microduck.tasks  # noqa: F401  (registrations + patches)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

DEVICE = "cuda:0"


def _np(x):
    t = x.detach().cpu() if hasattr(x, "detach") else x
    return np.asarray(t.numpy() if hasattr(t, "numpy") else t)


def trial(env, wrapped, steps: int, action: torch.Tensor) -> dict:
    wrapped.reset()
    r0 = env.scene["robot"]
    init = {
        "spawn_pos": _np(r0.data.root_link_pos_w[0]).reshape(-1).copy(),
        "spawn_quat": (_np(r0.data.root_link_quat_w[0]).reshape(-1).copy()
                       if hasattr(r0.data, "root_link_quat_w") else np.zeros(4)),
        "joint_pos": _np(r0.data.joint_pos[0]).reshape(-1).copy(),
    }
    for _ in range(steps):
        env.step(action)
    r = env.scene["robot"]
    ds = env.sim.data.struct
    return {
        **init,
        "qpos": _np(ds.qpos[0]).reshape(-1).copy(),
        "qvel": _np(ds.qvel[0]).reshape(-1).copy(),
        "root_z": float(r.data.root_link_pos_w[0, 2]),
        "gravity_z": float(r.data.projected_gravity_b[0, 2]),
    }


TRACE_FIELDS = ("ctrl", "qpos", "qvel", "actuator_force", "sensordata", "qacc", "nefc")


def traced_trial(env, wrapped, steps: int, action: torch.Tensor) -> dict[str, list]:
    """Per-step trace of everything that could carry the divergence."""
    wrapped.reset()
    ds = env.sim.data.struct
    rec: dict[str, list] = {k: [] for k in TRACE_FIELDS}
    rec["joint_pos_target"] = []
    for _ in range(steps):
        env.step(action)
        for k in TRACE_FIELDS:
            v = getattr(ds, k, None)
            if v is None:
                rec[k].append(np.zeros(1))
            else:
                arr = _np(v)                     # warp arrays reject item indexing: convert first
                rec[k].append(np.asarray(arr).reshape(-1).copy())
        tgt = getattr(env.scene["robot"].data, "joint_pos_target", None)
        rec["joint_pos_target"].append(_np(tgt[0]).reshape(-1).copy() if tgt is not None
                                       else np.zeros(1))
    return rec


def first_divergence(a: dict[str, list], b: dict[str, list]) -> None:
    print(f"{'field':20s} {'first step that differs':>24s} {'max|diff| there':>18s} "
          f"{'max|diff| overall':>18s}")
    for k in a:
        first, mag, worst = None, 0.0, 0.0
        for i, (x, y) in enumerate(zip(a[k], b[k])):
            d = float(np.abs(np.asarray(x) - np.asarray(y)).max()) if x.shape == y.shape else float("inf")
            worst = max(worst, d)
            if d > 0.0 and first is None:
                first, mag = i + 1, d
        print(f"{k:20s} {('-' if first is None else str(first)):>24s} {mag:18.3e} {worst:18.3e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Mjlab-Velocity-Flat-MicroDuck")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--play-cfg", action="store_true", help="no domain randomization at all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trace", action="store_true",
                    help="two trials with a per-step trace, reporting the FIRST field that differs "
                         "and at which step")
    ap.add_argument("--pin-spawn", action="store_true",
                    help="zero the spawn pose range (otherwise the trials start from different poses "
                         "and the comparison says nothing)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    env_cfg = load_env_cfg(args.task, play=args.play_cfg)
    env_cfg.scene.num_envs = 1
    if args.pin_spawn and "reset_base" in env_cfg.events:
        pr = dict(env_cfg.events["reset_base"].params.get("pose_range", {}))
        for k in ("x", "y", "z", "yaw"):
            if k in pr:
                m = (float(pr[k][0]) + float(pr[k][1])) / 2.0
                pr[k] = (m, m)
        env_cfg.events["reset_base"].params["pose_range"] = pr
        print("spawn pose pinned:", pr)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=DEVICE)
    agent_cfg = load_rl_cfg(args.task)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    action = torch.zeros(1, env.action_manager.total_action_dim, device=DEVICE)

    if args.trace:
        for act in getattr(env.scene["robot"], "actuators", []) or []:
            cfg = getattr(act, "cfg", None)
            print(f"actuator {type(act).__name__}: delay_min_lag={getattr(cfg, 'delay_min_lag', None)} "
                  f"delay_max_lag={getattr(cfg, 'delay_max_lag', None)} "
                  f"delay_update_period={getattr(cfg, 'delay_update_period', None)} "
                  f"delay_hold_prob={getattr(cfg, 'delay_hold_prob', None)} "
                  f"per_env_phase={getattr(cfg, 'delay_per_env_phase', None)}")
        a = traced_trial(env, wrapped, args.steps, action)
        b = traced_trial(env, wrapped, args.steps, action)
        print(f"per-step trace, two trials, {args.steps} steps each")
        print("(a difference in `ctrl` at step 1 means the ACTUATOR/command path is stochastic; "
              "identical ctrl with diverging qpos means the PHYSICS is)")
        first_divergence(a, b)
        return 0

    runs = [trial(env, wrapped, args.steps, action) for _ in range(args.trials)]
    print(f"task={args.task}  steps={args.steps}  trials={args.trials}  "
          f"play_cfg={args.play_cfg}  seed={args.seed}")
    print(f"{'trial':>5s} {'spawn xyz':>26s} {'init|dq| vs t1':>15s} {'root_z':>9s} "
          f"{'gravity_z':>10s} {'d qpos':>10s} {'d qvel':>10s}")
    base = runs[0]
    worst = 0.0
    for i, r in enumerate(runs):
        dq = float(np.abs(r["qpos"] - base["qpos"]).max())
        dv = float(np.abs(r["qvel"] - base["qvel"]).max())
        worst = max(worst, dq, dv)
        di = float(np.abs(r["joint_pos"] - base["joint_pos"]).max())
        sp = r["spawn_pos"]
        print(f"{i + 1:>5d} ({sp[0]:+.4f},{sp[1]:+.4f},{sp[2]:.4f}) {di:15.3e} "
              f"{r['root_z']:9.5f} {r['gravity_z']:10.5f} {dq:10.2e} {dv:10.2e}")
    print()
    if worst == 0.0:
        print("DETERMINISTIC: identical state + identical actions -> identical state, bit for bit.")
    else:
        print(f"NONDETERMINISTIC: the same reset + the same {args.steps} actions diverge by up to "
              f"{worst:.3e}. Nothing about the policy, the reward or the evaluation harness can "
              f"explain that; it is the environment itself.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
