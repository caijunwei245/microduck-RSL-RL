---
tags:
- microduck
- robotics
- reinforcement-learning
- onnx
library_name: onnx
---

# ball_kick_left

Kick the ball forward with the left foot, then recover to the stand.

A **episodic** policy for the [microduck](https://github.com/pollen-robotics/microduck) (61-D observation, 14 actions, 50 Hz). Runs 1.5 s and returns itself to a standing pose.

## Run it on a robot

```bash
sudo robotctl policy add ball_kick_left <user>/microduck-ball_kick_left
robotctl robot do ball_kick_left
```

The observation normalizer is baked into `policy.onnx`; feed raw observations.
`manifest.json` follows schema 2 of the microduck policy manifest (`docs/policy-manifest.md` in the daemon repo).

## Training

- **task_id**: `Mjlab-BallKick-Flat-MicroDuck`
- **repo**: `pollen-robotics/microduck_rl`
- **branch**: `develop`
- **commit**: `c10f9a0ce`
- **checkpoint**: `1499`
- exported from a checkout with uncommitted changes
