---
tags:
- microduck
- robotics
- reinforcement-learning
- onnx
library_name: onnx
---

# roller_crouch

Mjlab-RollerCrouch-Flat-MicroDuck

A **episodic** policy for the [microduck](https://github.com/pollen-robotics/microduck) (61-D observation, 14 actions, 50 Hz). Runs 3.0 s while the daemon sweeps the phase over 5.0 s (to 0.6); the policy is the validated phase-commanded one, unchanged.

## Run it on a robot

```bash
# expects command.encoding=phase (period_s=5.0, end_phase=0.6) — the daemon writes [cos(2*pi*phi), sin(2*pi*phi), 0]
sudo robotctl policy add roller_crouch dryrun/microduck-roller_crouch
robotctl robot do roller_crouch
```

The observation normalizer is baked into `policy.onnx`; feed raw observations.
`manifest.json` follows schema 2 of the microduck policy manifest (`docs/policy-manifest.md` in the daemon repo).

## Training

- **task_id**: `Mjlab-RollerCrouch-Flat-MicroDuck`
- **repo**: `pollen-robotics/microduck_rl`
- **branch**: `develop`
- **commit**: `cdbb6ecab`
- **checkpoint**: `1999`
- exported from a checkout with uncommitted changes
