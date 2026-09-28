"""Why does a kick miss? Foot-to-ball geometry, per round.

The ledger has the kick at 13/25 (52 %) for weeks with the spawn-noise explanation ruled out
(1.0x / 0.5x / 0.25x ball noise -> 52/48/48 %), and the constants block in the cfg warns that the
strike lands at the very edge of reach: "40 ms / 1.5 cm decided between a kick and a miss". A miss is
therefore either

    SHORT   the foot never gets within contact range of the ball, or
    SOFT    it arrives but too slowly / too glancing to send the ball anywhere,

and the two need opposite fixes (a geometry/approach change vs a strike-quality reward). This probe
separates them by recording, per step, the distance from the kicking foot's collision geom to the
ball and the ball's speed around the closest approach.

It drives `skill_demo.build_agent` + `run_round` (same construction and rollout as every demo).

Usage:  uv run python logs/kick_contact_probe.py [--rounds 12] [--seed 0] [--ckpt path.pt]
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import skill_demo  # noqa: E402
from skill_demo import build_agent  # noqa: E402

HIT_M = 0.05          # the demo's criterion: the ball moved >= 5 cm forward


def foot_geom_ids(robot, foot: str):
    """`find_geoms` returns (indices, names) - unpacking it wrong silently fell back to the trunk
    position and reported 110-120 mm for every round, hit or miss."""
    try:
        ids, names = robot.find_geoms(f"^{foot}_foot_collision$")
        if ids is not None and len(ids):
            return [int(i) for i in ids]
    except Exception as e:  # noqa: BLE001
        print(f"  (foot geom lookup failed: {e!r})")
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    spec = None
    for row in skill_demo.SKILLS:
        if row[0] == "ball_kick right":
            spec = row
    ckpt = args.ckpt or skill_demo.newest(spec[2])
    spec = (spec[0], spec[1], ckpt, spec[3], spec[4], spec[5], spec[6], spec[7])
    print(f"checkpoint: {ckpt}")
    torch.manual_seed(args.seed)
    env, wrapped, policy, max_steps, _mk = build_agent(spec[1], ckpt, quiet=True)
    robot = env.scene["robot"]
    ball = env.scene["ball"]
    ids = foot_geom_ids(robot, "right")
    print(f"kicking-foot collision geoms: {ids or '<name lookup failed, using body pos>'}")
    step = env.step

    print(f"\n{'round':>5s} {'min_dist_mm':>11s} {'at_step':>8s} {'ball_v_at_contact':>17s} "
          f"{'ball_final_m':>12s}  outcome")
    hits, shorts, softs, v0s = 0, 0, 0, []
    for r_i in range(args.rounds):
        obs = wrapped.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        ball0 = float(ball.data.root_link_pos_w[0, 0])
        v0 = float(np.linalg.norm(np.asarray(
            ball.data.root_link_lin_vel_w[0].detach().cpu()).reshape(-1)[:2]))
        dmin, dstep, v_at, vs = 1e9, -1, 0.0, []
        v0s.append(v0)
        with torch.no_grad():
            for i in range(max_steps):
                bp = np.asarray(ball.data.root_link_pos_w[0].detach().cpu()).reshape(-1)[:3]
                if ids:
                    gp = np.asarray(robot.data.geom_pos_w[0].detach().cpu()).reshape(-1, 3)
                    fp = np.stack([gp[j] for j in ids]).mean(axis=0)
                else:
                    fp = np.asarray(robot.data.body_link_pos_w[0].detach().cpu()).reshape(-1, 3)[:1][0]
                d = float(np.linalg.norm(bp - fp))
                v = float(np.linalg.norm(np.asarray(
                    ball.data.root_link_lin_vel_w[0].detach().cpu()).reshape(-1)[:2]))
                vs.append(v)
                if d < dmin:
                    dmin, dstep = d, i
                out = env.step(policy(skill_demo.as_actor(obs)))
                obs = out[0]
                term_b, trunc_b = out[2], out[3]
                if bool((term_b | trunc_b)[0]):
                    break
        v_at = max(vs[dstep:dstep + 4]) if vs else 0.0
        ball_final = float(ball.data.root_link_pos_w[0, 0]) - ball0
        hit = ball_final >= HIT_M
        hits += hit
        if hit:
            outcome = "hit"
        elif dmin < 0.04:                      # got inside contact range but the ball stayed put
            softs += 1
            outcome = "SOFT (arrived, no speed)"
        else:
            shorts += 1
            outcome = "SHORT (never in range)"
        print(f"{r_i + 1:>5d} {dmin * 1000:>11.1f} {dstep:>8d} {v_at:>17.3f} "
              f"{ball_final:>12.3f}  {outcome}")
    del env
    n = args.rounds
    print(f"\nhits {hits}/{n} ({hits * 100 // n} %) | misses: SHORT {shorts}, SOFT {softs}")
    print(f"ball speed at step 0: median {statistics.median(v0s):.3f} m/s (max {max(v0s):.3f}) - "
          f"anything above ~0.1 here is the ball being EJECTED at reset, not kicked")
    print("SHORT -> the approach/timing never reaches the ball (geometry or swing change);")
    print("SOFT  -> contact happens too gently to matter (strike-quality reward, not geometry).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
