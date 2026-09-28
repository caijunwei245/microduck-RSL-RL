"""Paired A/B evaluation: same seeds, two checkpoints, seed as the unit of analysis.

Why this exists (`logs/optimization_space_2026-09-28.md` §1): the simulator is chaotic and not
bit-reproducible across resets, so episode outcomes carry a large per-seed component. At the gate's
episode counts a round-pass rate has a 95 % interval of ±14-19 points, and detecting a 10-point
improvement on a 0.60 row needs ~188 rounds (38 seeds). Re-running the SAME seeds for both arms
removes the between-seed component from the comparison, so a ~1-round-per-seed shift becomes visible
at 5 seeds instead of ~30 - the seed-level delta is the statistic, not the pooled rate.

It reuses `skill_demo.build_agent` (environment construction) and `skill_demo.run_round` (the rollout
loop) rather than re-implementing either: a hand-rolled copy of that loop has already hidden a real
effect in this project (`logs/turn_tail_findings.md`).

Usage:
    uv run python logs/paired_eval.py --skill "velocity turn" \\
        --a logs/rsl_rl/velocity/<run>/model_1999.pt \\
        --b logs/rsl_rl/velocity/<run>/model_4998.pt \\
        [--seeds 5] [--rounds 5] [--task TASK] [--turn 0.5] [--label-a before --label-b after]

Output: per-seed table (rounds passed and the row's primary continuous metric, both arms, with the
delta), then the paired summary - mean seed-level delta ± SE, a sign test, how many seeds moved - and
for contrast the pooled rates with the UNPAIRED interval, which is what the pairing buys.
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import skill_demo  # noqa: E402
from skill_demo import build_agent, run_round  # noqa: E402

# The row's primary continuous readout, per kind. Pass rates are derived from noisy booleans;
# these numbers are what the booleans are computed from.
PRIMARY = {
    "walk": "v_mean",
    "turn": "yaw_abs_mean",
    "ball": "ball",
    "spin": "yaw_abs_mean",
    "floor_flip": "hold",
    "slope_descent": "descent",
    "roller_stand": "z_max",
    "roulade": "z_last",
    "cycle_low": "z_max",
    # sitstand: the terminal trunk height, which per command pattern is ~53 mm (held sit) or ~116 mm
    # (held stand / after a rise) - the row's per-round detail line carries the pattern.
    "posture_cycle": "z_last",
    "crouch_cycle": "z_max",
}


def spec_for(skill: str, ckpt: str, task: str | None, turn_cmd: float):
    """Find the rotation's row for this skill and swap in the checkpoint.

    The criteria, commands and pin all come FROM the rotation's table, so a paired comparison cannot
    silently judge a different thing than the demos do.
    """
    for row in skill_demo.SKILLS:
        name, row_task, _glob, kind, cx, cy, floor, low = row
        if skill.lower() in name.lower():
            return (name, task or row_task, ckpt, kind, cx, (turn_cmd if kind == "turn" else cy),
                    floor, low)
    if "turn" in skill.lower():          # these two are inserted in skill_demo.main()
        cx = 0.3 if skill.lower().startswith("walk") else 0.0
        return (skill, task or "Mjlab-Velocity-Flat-MicroDuck", ckpt, "turn", cx, turn_cmd, None, None)
    raise SystemExit(f"unknown skill {skill!r} - see the SKILLS table in logs/skill_demo.py")


def run_arm(spec, ckpt: str, seed: int, rounds: int, play_cfg: bool, fresh_env: bool,
            turn_cmd: float) -> list[dict]:
    """One arm's `rounds` episodes at one seed, through the rotation's own loop."""
    torch.manual_seed(seed)
    env, wrapped, policy, max_steps, mk_env = build_agent(
        spec[1], ckpt, play_cfg=play_cfg, pin_floor=bool(spec[6]), quiet=True)
    out = []
    for r_i in range(rounds):
        if fresh_env and r_i > 0:
            del env, wrapped
            env, wrapped, policy, max_steps, mk_env = build_agent(
                spec[1], ckpt, play_cfg=play_cfg, pin_floor=bool(spec[6]), quiet=True)
        res = run_round(env, wrapped, policy, spec, max_steps)
        out.append(res)
    del env
    return out


def sign_test(deltas: list[float]) -> float:
    """Two-sided exact binomial p on the non-tied deltas (n is small by construction)."""
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def pooled_ci(passes: int, n: int) -> str:
    if n == 0:
        return "n/a"
    p = passes / n
    hw = 1.96 * math.sqrt(p * (1 - p) / n)
    return f"{p * 100:.0f}% +/-{hw * 100:.0f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill", required=True, help='e.g. "velocity turn", "sitstand", "spin"')
    ap.add_argument("--a", required=True, help="checkpoint A (the baseline)")
    ap.add_argument("--b", required=True, help="checkpoint B (the candidate)")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--task", default=None, help="override the task id (default: the row's own)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--turn", type=float, default=0.5)
    ap.add_argument("--play-cfg", action="store_true", help="no domain randomization")
    ap.add_argument("--fresh-env", action="store_true", help="new env per round (see --help of skill_demo)")
    args = ap.parse_args()

    spec_a = spec_for(args.skill, args.a, args.task, args.turn)
    spec_b = spec_for(args.skill, args.b, args.task, args.turn)
    metric = PRIMARY.get(spec_a[3], "yaw_abs_mean")
    print(f"############ paired A/B: {spec_a[0]} ############")
    print(f"  {args.label_a:>10s}: {args.a}")
    print(f"  {args.label_b:>10s}: {args.b}")
    print(f"  task {spec_a[1]} | kind {spec_a[3]} | cmd_x {spec_a[4]} cmd_yaw {spec_a[5]} "
          f"| pin {spec_a[6]} | primary metric '{metric}'")
    print(f"  {args.seeds} seeds x {args.rounds} rounds per arm; the SEED is the unit of analysis")
    print()

    per_seed = []
    for s in range(args.seeds):
        ra = run_arm(spec_a, args.a, s, args.rounds, args.play_cfg, args.fresh_env, args.turn)
        rb = run_arm(spec_b, args.b, s, args.rounds, args.play_cfg, args.fresh_env, args.turn)
        pa = sum(1 for r in ra if r["ok"])
        pb = sum(1 for r in rb if r["ok"])
        ma = statistics.fmean(r[metric] for r in ra)
        mb = statistics.fmean(r[metric] for r in rb)
        per_seed.append(dict(seed=s, pa=pa, pb=pb, d=pb - pa, ma=ma, mb=mb, dm=mb - ma))
        print(f"  seed {s}: rounds {args.label_a} {pa}/{args.rounds}  {args.label_b} "
              f"{pb}/{args.rounds}  delta {pb - pa:+d}   | {metric}: {ma:.4g} -> {mb:.4g} "
              f"({mb - ma:+.4g})")

    n = len(per_seed)
    d_rounds = [p["d"] for p in per_seed]
    d_metric = [p["dm"] for p in per_seed]
    print()
    print(f"  PAIRED (seed-level, n={n}):")
    if n > 1:
        sd = statistics.stdev(d_rounds)
        print(f"    rounds/seed delta: mean {statistics.fmean(d_rounds):+.2f} "
              f"+/-{1.96 * sd / math.sqrt(n):.2f} (95%), SD {sd:.2f}, "
              f"sign test p={sign_test(d_rounds):.3f}")
        sdm = statistics.stdev(d_metric)
        print(f"    '{metric}' delta : mean {statistics.fmean(d_metric):+.4g} "
              f"+/-{1.96 * sdm / math.sqrt(n):.4g} (95%), SD {sdm:.4g}")
    improved = sum(1 for d in d_rounds if d > 0)
    worse = sum(1 for d in d_rounds if d < 0)
    print(f"    seeds improved {improved}, regressed {worse}, tied {n - improved - worse}")
    tot_a = sum(p["pa"] for p in per_seed)
    tot_b = sum(p["pb"] for p in per_seed)
    rounds_all = n * args.rounds
    print()
    print(f"  UNPAIRED contrast (what a pooled gate would say):")
    print(f"    {args.label_a}: {tot_a}/{rounds_all} = {pooled_ci(tot_a, rounds_all)}")
    print(f"    {args.label_b}: {tot_b}/{rounds_all} = {pooled_ci(tot_b, rounds_all)}")
    print("    -> the pooled intervals overlap for any effect smaller than ~15 points; the paired")
    print("       column above is the one that can see a one-round-per-seed shift.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
