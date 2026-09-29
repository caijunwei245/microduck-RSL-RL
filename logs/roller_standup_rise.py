"""How long does RollerStandUp take to get up, from the spawn that actually needs it?

`logs/publish_duration_sweep.py` says every window from 0.6 s up passes 15/15 with the trunk ending at
139 mm — which cannot be the rise time, because the task spawns **50 % belly / 50 % already standing**
(`set_ground_state`, and the standing bucket is deliberate: it is what teaches the policy to HOLD, not
just to rise). Half the rounds therefore pass a short window for free, and a pass rate over the mixed
spawn says nothing about the rise. Same class as the ball_kick frame and the spin rest phase: the
number was being read against a criterion the episode had not been asked to satisfy.

This pins the spawn through the event manager (the only way to reach a reset event's params — writes to
`env.cfg` are silent no-ops, see AGENTS.md) and reports the rise time per round:

    belly    face_down_prob=1.0 -> the shortest window that covers the worst spawn sets duration_s
    standing standing_prob=1.0  -> the same window must NOT make it fall over

Both buckets matter for a published button: the daemon may call the skill from either pose and the
manifest declares one `duration_s`.

Usage:  uv run python logs/roller_standup_rise.py [--rounds 15] [--window 3.0] [--row roller_standup]
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


def pin_spawn(env, bucket: str) -> dict:
    """Force one bucket of `set_ground_state` — through the CURRICULUM, not the event.

    Writing `event_manager.get_term_cfg("set_ground_state").params` looks right and does nothing:
    `ground_state_mix` is an `event_param_curriculum` that runs BEFORE the reset events and rewrites
    those four probabilities on every reset (the env cfg says so in a comment, for the play path where
    it deletes the term outright). Measured here: with the event pinned to face_down=1.0 the spawn mix
    stayed ~50/50, i.e. the pin was silently reverted - the same silent-no-op shape as AGENTS.md's
    "writes to env.cfg are ignored". Rewriting the curriculum's own stage is what actually pins it.
    """
    probs = {
        "belly":    {"face_down_prob": 1.0, "face_up_prob": 0.0, "sitting_prob": 0.0, "standing_prob": 0.0},
        "back":     {"face_down_prob": 0.0, "face_up_prob": 1.0, "sitting_prob": 0.0, "standing_prob": 0.0},
        "standing": {"face_down_prob": 0.0, "face_up_prob": 0.0, "sitting_prob": 0.0, "standing_prob": 1.0},
    }[bucket]
    cfg = env.curriculum_manager.get_term_cfg("ground_state_mix")
    cfg.params["param_stages"] = [{"step": 0, "params": dict(probs)}]
    # The event gets the same numbers so the two cannot disagree if the curriculum is ever skipped.
    env.event_manager.get_term_cfg("set_ground_state").params.update(probs)
    return probs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", default="roller_standup")
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--window", type=float, default=3.0, help="rollout length in seconds")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()

    spec = next((r for r in skill_demo.SKILLS if args.row.lower() in r[0].lower()), None)
    assert spec is not None, f"no row matches {args.row!r}"
    ckpt = args.ckpt or skill_demo.newest(spec[2])
    spec = (spec[0], spec[1], ckpt, spec[3], spec[4], spec[5], spec[6], spec[7])
    steps = int(round(args.window * CONTROL_HZ))
    print(f"row: {spec[0]}  task: {spec[1]}")
    print(f"checkpoint: {ckpt}")
    print(f"window: {args.window} s ({steps} steps), {args.rounds} rounds per bucket "
          f"(belly = the rise case, back = the hardest direction, standing = the HOLD case)\n")

    torch.manual_seed(args.seed)
    env, wrapped, policy, _max_steps, _mk = build_agent(spec[1], ckpt, quiet=True)

    for bucket in ("belly", "back", "standing"):
        wrote = pin_spawn(env, bucket)
        print(f"--- spawn pinned to {bucket}: {wrote} ---")
        res = [run_round(env, wrapped, policy, spec, steps) for _ in range(args.rounds)]
        for i, r in enumerate(res):
            print(f"  round {i + 1:2d}: z0={r['z0']:.0f} mm rise="
                  f"{'%.2f s' % r['z130_s'] if r['z130_s'] is not None else ' never '} "
                  f"z_last={r['z_last']:.0f} mm max={r['z_max']:.0f} held_high={r['held_high']:3d} "
                  f"upright={r['upright_frac']:.2f} fell={r['fell']}")
        rises = [r["z130_s"] for r in res if r["z130_s"] is not None]
        holds = [r["held_high"] for r in res]
        print(f"  {bucket}: reached 130 mm in {len(rises)}/{len(res)} rounds; "
              f"rise median {statistics.median(rises):.2f} s "
              f"(range {min(rises):.2f}-{max(rises):.2f})" if rises else
              f"  {bucket}: never reached 130 mm")
        print(f"  z_last median {statistics.median(r['z_last'] for r in res):.0f} mm, "
              f"held_high median {statistics.median(holds):.0f} steps, "
              f"upright median {statistics.median(r['upright_frac'] for r in res):.2f}, "
              f"fell {sum(1 for r in res if r['fell'])}/{len(res)}\n")

    del env
    print("duration_s for the manifest = the belly rise median plus a hold margin; the standing bucket")
    print("must survive the same window (it is the bucket that tests holding, not rising).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
