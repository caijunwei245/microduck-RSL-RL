---
tags:
- microduck
- robotics
- reinforcement-learning
- onnx
library_name: onnx
---

# roller_standup

Get up onto the passive wheels from the belly or the back, then hold the roller stand.

A **episodic** policy for the [microduck](https://github.com/pollen-robotics/microduck) (61-D observation, 14 actions, 50 Hz). Runs 1.0 s and returns itself to a standing pose.

## Run it on a robot

```bash
sudo robotctl policy add roller_standup <user>/microduck-roller_standup
robotctl robot do roller_standup
```

The observation normalizer is baked into `policy.onnx`; feed raw observations.
`manifest.json` follows schema 2 of the microduck policy manifest (`docs/policy-manifest.md` in the daemon repo).

## Training

- **task_id**: `Mjlab-RollerStandUp-Flat-MicroDuck`
- **repo**: `pollen-robotics/microduck_rl`
- **branch**: `develop`
- **commit**: `c10f9a0ce`
- **checkpoint**: `1999`
- exported from a checkout with uncommitted changes
