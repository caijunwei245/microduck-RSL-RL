---
tags:
- microduck
- robotics
- reinforcement-learning
- onnx
library_name: onnx
---

# sitstand

Sit down or stand up on command; holds the commanded posture.

A **scripted** policy for the [microduck](https://github.com/pollen-robotics/microduck) (61-D observation, 14 actions, 50 Hz). Holds the commanded posture until the flag is toggled back; the daemon drives `command.sit` / `command.stand`. A flip is answered by a constant-rate glide over `ramp_s` (2.0 s), not a jump.

## Run it on a robot

```bash
# expects command.encoding=posture_flag (sit=1.0, stand=0.0, ramp_s=2.0) — the owner toggles the flag
sudo robotctl policy add sitstand <user>/microduck-sitstand --hold 1.0
robotctl robot do sitstand
```

The observation normalizer is baked into `policy.onnx`; feed raw observations.
`manifest.json` follows schema 2 of the microduck policy manifest (`docs/policy-manifest.md` in the daemon repo).

## Training

- **task_id**: `Mjlab-SitStand-Flat-MicroDuck`
- **commit**: `1905375`
- **run**: `2026-09-24_08-58-42_sitstand_hold2`
- **checkpoint**: `model_2498.pt`

