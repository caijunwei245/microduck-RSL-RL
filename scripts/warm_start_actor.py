"""Warm-start a run's ACTOR from another family's checkpoint (critic kept, optimizer reset).

The obs layout is shared across the whole policy family (61D actor), so an actor trained for one
skill is a valid initialisation for another — the earlier `velstand_warm` run proved it
(velocity -> velstand). What blocks a cross-MODEL transfer is the CRITIC: privileged obs differ per
task (74D standup vs 76D velstand, measured 2026-09-23), so a straight resume dies with

    size mismatch for mlp.0.weight: copying [512, 74] into [512, 76]

The critic is task-specific and has to be relearned anyway, so only the actor is worth carrying over.
This tool builds a resume-able checkpoint by taking the BASE checkpoint (right critic shape, right
normalizer for the target task) and replacing its `actor_state_dict` with the donor's. The optimizer
state is dropped, because its Adam moments belong to the donor's actor parameters.

Usage::

    uv run python scripts/warm_start_actor.py \
        --donor logs/rsl_rl/microduck_stand/<run>/model_54994.pt \
        --base  logs/rsl_rl/velstand/<run>/model_75247.pt \
        --out-dir logs/rsl_rl/velstand/_warm_actor [--out-name model_0.pt]

Then resume with::

    --agent.resume True --agent.load-run _warm_actor --agent.load-checkpoint model_0.pt
"""

import argparse
import os
import shutil

import torch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--donor", required=True, help="checkpoint whose ACTOR is carried over")
    ap.add_argument("--base", required=True, help="checkpoint that fixes the critic/normalizer shape")
    ap.add_argument("--out-dir", required=True, help="run dir to write (params/ copied from --base)")
    ap.add_argument("--out-name", default="model_0.pt")
    args = ap.parse_args()

    donor = torch.load(args.donor, map_location="cpu", weights_only=False)
    base = torch.load(args.base, map_location="cpu", weights_only=False)
    if "actor_state_dict" not in donor or "actor_state_dict" not in base:
        raise SystemExit("both checkpoints must carry actor_state_dict")

    d_actor, b_actor = donor["actor_state_dict"], base["actor_state_dict"]
    if set(d_actor) != set(b_actor):
        raise SystemExit(
            f"actor keys differ:\n  donor only: {sorted(set(d_actor) - set(b_actor))}\n"
            f"  base only:  {sorted(set(b_actor) - set(d_actor))}"
        )
    mismatched = {
        k: (tuple(d_actor[k].shape), tuple(b_actor[k].shape))
        for k in d_actor
        if tuple(d_actor[k].shape) != tuple(b_actor[k].shape)
    }
    if mismatched:
        raise SystemExit(f"actor shapes differ (dims must match to transfer): {mismatched}")

    out = dict(base)
    out["actor_state_dict"] = d_actor
    out["iter"] = 0
    # rsl_rl's loader REQUIRES optimizer_state_dict (it indexes it directly, no .get()), so the key
    # has to stay. The base's Adam moments belong to the base actor's parameters though, so zero them
    # and reset the step counters: fresh Adam on the donor's weights, no stale momentum.
    opt = out.get("optimizer_state_dict")
    zeroed = 0
    if isinstance(opt, dict) and isinstance(opt.get("state"), dict):
        for st in opt["state"].values():
            if not isinstance(st, dict):
                continue
            for key in ("exp_avg", "exp_avg_sq"):
                if key in st and torch.is_tensor(st[key]):
                    st[key] = torch.zeros_like(st[key])
                    zeroed += 1
            if "step" in st:
                st["step"] = (
                    torch.zeros_like(st["step"]) if torch.is_tensor(st["step"]) else 0
                )
    infos = dict(out.get("infos") or {})
    infos["warm_start_from"] = os.path.abspath(args.donor)
    out["infos"] = infos

    os.makedirs(args.out_dir, exist_ok=True)
    base_params = os.path.join(os.path.dirname(os.path.abspath(args.base)), "params")
    if os.path.isdir(base_params):
        shutil.copytree(base_params, os.path.join(args.out_dir, "params"), dirs_exist_ok=True)

    dest = os.path.join(args.out_dir, args.out_name)
    torch.save(out, dest)
    n = len(d_actor)
    print(f"wrote {dest}: {n} actor tensors from {args.donor}")
    print(f"  critic/normalizer kept from {args.base}; {zeroed} Adam moments zeroed; iter reset to 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
