"""How long does one deployable cycle take? Duration sweep for the episodic skills.

The training tasks run 20 s episodes (several cycles, so PPO sees the transition repeatedly), but a
published skill is a BUTTON: `duration_s` should be the length of one useful cycle, and `end_phase`
the phase at which it completes. The official set's ground_pick, for instance, runs 2.8 s of its 4 s
cycle (end_phase 0.7). Guessing those numbers is how a policy ships that stops halfway.

This measures the shortest duration at which the row's own criterion still passes, by running
`skill_demo.run_round` with a truncated step budget (the env is built normally; only the rollout is
cut short, which is exactly what a `duration_s` button does).

Usage:  uv run python logs/publish_duration_sweep.py --row ground_pick --durations 2.0 2.5 3.0 4.0
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import skill_demo  # noqa: E402
from skill_demo import build_agent, run_round  # noqa: E402

CONTROL_HZ = 50


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", required=True, help='demo row name, e.g. "ground_pick"')
    ap.add_argument("--durations", type=float, nargs="+", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--detail", action="store_true",
                    help="print every round's readout, not just the medians (a truncated window can "
                         "fail for reasons the median hides - see the spin row)")
    args = ap.parse_args()

    spec = None
    for row in skill_demo.SKILLS:
        if args.row.lower() in row[0].lower():
            spec = row
    assert spec is not None, f"no row matches {args.row!r}"
    ckpt = args.ckpt or skill_demo.newest(spec[2])
    spec = (spec[0], spec[1], ckpt, spec[3], spec[4], spec[5], spec[6], spec[7])
    print(f"row: {spec[0]}  task: {spec[1]}  kind: {spec[3]}")
    print(f"checkpoint: {ckpt}\n")

    torch.manual_seed(args.seed)
    env, wrapped, policy, _max_steps, _mk = build_agent(spec[1], ckpt, quiet=True)
    print(f"{'duration_s':>10s} {'steps':>6s} {'passed':>7s}   readouts")
    for dur in args.durations:
        steps = int(round(dur * CONTROL_HZ))
        res = [run_round(env, wrapped, policy, spec, steps) for _ in range(args.rounds)]
        ok = sum(1 for r in res if r["ok"])
        spans = [r["z_max"] - r["z_min"] for r in res]
        zlast = [r["z_last"] for r in res]
        run_metric = ""
        if spec[3] == "turn" or spec[3] == "spin":
            run_metric = f"|yaw|={statistics.fmean(r['yaw_abs_mean'] for r in res):.3f} "
        if spec[3] == "spin":
            # The button question, not the demo question: how much rotation does this window deliver?
            run_metric += (f"turned={statistics.median(r['yaw_abs_rad'] for r in res):.1f} rad "
                           f"(net {statistics.median(r['yaw_net_rad'] for r in res):+.1f}) ")
        if spec[3] == "ball":
            # A kick's own number: how far the ball went. It keeps rolling after the robot stops, so the
            # published `duration_s` has to cover the TRAVEL, not just the strike (contact is at step
            # 5-8, see logs/kick_frame_fix.txt) - the sweep shows where the distance plateaus.
            run_metric += (f"ball={statistics.median(r['ball'] for r in res):.2f} m "
                           f"(min {min(r['ball'] for r in res):.2f}) ")
        if spec[3] == "roller_stand":
            run_metric += (f"z_max={statistics.median(r['z_max'] for r in res):.0f} mm "
                           f"held_high={int(statistics.median(r['held_high'] for r in res))} steps "
                           f"upright={statistics.median(r['upright_frac'] for r in res):.2f} ")
        print(f"{dur:>10.1f} {steps:>6d} {ok:>3d}/{len(res):<3d}   {run_metric}"
              f"span median {statistics.median(spans):.0f} mm, z_last median {statistics.median(zlast):.0f} mm")
        if args.detail:
            for i, r in enumerate(res):
                print(f"{'':>10s} round {i + 1}: ok={str(r['ok']):5s} {skill_demo.detail(spec[3], r)}")
    del env
    print("\nThe shortest duration that still passes is the one to put in the manifest; `end_phase` is")
    print("that duration divided by the task's own phase period (4.0 s ground_pick, 5.0 s crouch).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
