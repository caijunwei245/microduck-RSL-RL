---
tags:
- microduck
- robotics
- reinforcement-learning
- onnx
library_name: onnx
---

# sitstand

Mjlab-SitStand-Flat-MicroDuck

A **scripted** policy for the [microduck](https://github.com/pollen-robotics/microduck) (61-D observation, 14 actions, 50 Hz). Holds the commanded posture until the flag is toggled back; the daemon drives `command.sit` / `command.stand`.

## Run it on a robot

```bash
# expects command.encoding=posture_flag (sit=1.0, stand=0.0) — the owner toggles the flag
sudo robotctl policy add sitstand dryrun/microduck-sitstand --hold 1.0
robotctl robot do sitstand
```

The observation normalizer is baked into `policy.onnx`; feed raw observations.
`manifest.json` follows schema 2 of the microduck policy manifest (`docs/policy-manifest.md` in the daemon repo).

## Training

- **task_id**: `Mjlab-SitStand-Flat-MicroDuck`
- **repo**: `pollen-robotics/microduck_rl`
- **branch**: `develop`
- **commit**: `cdbb6ecab`
- **checkpoint**: `2498`
- exported from a checkout with uncommitted changes
