---
tags:
- microduck
- robotics
- reinforcement-learning
- onnx
library_name: onnx
---

# ground_pick

Mjlab-GroundPick-Flat-MicroDuck

A **episodic** policy for the [microduck](https://github.com/pollen-robotics/microduck) (61-D observation, 14 actions, 50 Hz). Runs 3.0 s while the daemon sweeps the phase over 4.0 s (to 0.75); the policy is the validated phase-commanded one, unchanged.

## Run it on a robot

```bash
# expects command.encoding=phase (period_s=4.0, end_phase=0.75) — the daemon writes [cos(2*pi*phi), sin(2*pi*phi), 0]
sudo robotctl policy add ground_pick dryrun/microduck-ground_pick
robotctl robot do ground_pick
```

The observation normalizer is baked into `policy.onnx`; feed raw observations.
`manifest.json` follows schema 2 of the microduck policy manifest (`docs/policy-manifest.md` in the daemon repo).

## Training

- **task_id**: `Mjlab-GroundPick-Flat-MicroDuck`
- **repo**: `pollen-robotics/microduck_rl`
- **branch**: `develop`
- **commit**: `cdbb6ecab`
- **checkpoint**: `999`
- exported from a checkout with uncommitted changes
