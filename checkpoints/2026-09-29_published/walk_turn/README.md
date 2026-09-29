---
tags:
- microduck
- robotics
- reinforcement-learning
- onnx
library_name: onnx
---

# walk_turn

Walk, and turn in place at the commanded rate (wobble tax off: 0.5 rad/s at gain 1.17-1.23). Loads into the walk slot; the factory walk policy is a different repo.

A **perpetual** policy for the [microduck](https://github.com/pollen-robotics/microduck) (61-D observation, 14 actions, 50 Hz). Runs until told otherwise — a gait for the `walk` slot.

## Run it on a robot

```bash
sudo robotctl policy load walk <user>/microduck-walk_turn
```

The observation normalizer is baked into `policy.onnx`; feed raw observations.
`manifest.json` follows schema 2 of the microduck policy manifest (`docs/policy-manifest.md` in the daemon repo).

## Training

- **task_id**: `Mjlab-Velocity-Flat-MicroDuck`
- **repo**: `pollen-robotics/microduck_rl`
- **branch**: `develop`
- **commit**: `c10f9a0ce`
- **checkpoint**: `4998`
- exported from a checkout with uncommitted changes
