# Publish dry run — the four skills schema 2 was thought to exclude (2026-09-28)

`logs/publishability.md` recorded these as unpublishable and Route A (internal clock) was
measured to fail for spin (0/15). Re-reading the contract showed the capability was already
there: `validate_manifest` accepts `kind="scripted"` and `command.encoding` in
`{constant, phase, posture_flag}`, and the OFFICIAL SET as uploaded 2026-09-02 (carried in
`tests/test_publish_manifest.py`) uses exactly those shapes for sitstand and ground_pick.
Only our builder refused to emit them; it now does. **Nothing about the policies changes —
the validated, phase-commanded checkpoint is what ships.**

| skill | encoding | kind | manifest | timing (measured) |
|---|---|---|---|---|
| ground_pick | `phase` | episodic | **valid** | duration/end_phase measured: 3.0 s of the 4.0 s phase cycle completes the manoeuvre (official set used 2.8 s / 0.7 for the same task) |
| roller_crouch | `phase` | episodic | **valid** | passes at every duration >= 2.5 s; 3.0 s is the shortest that also ends standing (z_last 111 mm) |
| spin | `phase` | episodic | **valid** | 2.6 s = the brake end of the task's own phase envelope (SPIN_BRAKE_END 0.650): one full turn (6.6 rad measured at 15 rounds), handed back at the commanded zero rate. A longer window delivers the SAME rotation and then sits in the envelope's commanded rest segment - logs/publish_duration_spin15.txt |
| sitstand | `posture_flag` | scripted | **valid** | ramp_s == the cfg's POSTURE_RAMP_S; unwind_s 1.0 as in the official set |

## What each upload needs

### ground_pick

```bash
# expects command.encoding=phase (period_s=4.0, end_phase=0.75) — the daemon writes [cos(2*pi*phi), sin(2*pi*phi), 0]
sudo robotctl policy add ground_pick <user>/microduck-ground_pick
robotctl robot do ground_pick
```

### roller_crouch

```bash
# expects command.encoding=phase (period_s=5.0, end_phase=0.6) — the daemon writes [cos(2*pi*phi), sin(2*pi*phi), 0]
sudo robotctl policy add roller_crouch <user>/microduck-roller_crouch
robotctl robot do roller_crouch
```

### spin

```bash
# expects command.encoding=phase (period_s=4.0, end_phase=0.65) — the daemon writes [cos(2*pi*phi), sin(2*pi*phi), 0]
sudo robotctl policy add spin <user>/microduck-spin
robotctl robot do spin
```

### sitstand

```bash
# expects command.encoding=posture_flag (sit=1.0, stand=0.0, ramp_s=2.0) — the owner toggles the flag
sudo robotctl policy add sitstand <user>/microduck-sitstand --hold 1.0
robotctl robot do sitstand
```

## The one thing this repo cannot test

The FORMAT is accepted by our validator and used by the official set. Whether the daemon
accepts a **community-published** policy declaring `phase` / `posture_flag` (rather than
restricting those encodings to the official uploads) is a decision in the `microduck` repo —
so each generated manifest's README and the `robotctl` line name the driver the policy
expects, and the first hardware run should confirm it. That question is now the ONLY open
item between these four checkpoints and a `robotctl policy add`.

## What is still missing for a real upload

1. a Hugging Face token — none is configured in this environment, so nothing is uploaded;
2. confirmation that the daemon accepts these encodings from a COMMUNITY repo (above).

The ONNX half is no longer missing: `bash logs/publish_stage.sh` runs the real `uv run publish`
path per skill — export from the checkpoint, shape gate (61 -> 14), smoke run, manifest build +
validate, README — and stops at `--dry-run` with `publish-<name>/` staged
(`logs/publish_stage.txt`). This file is the manifest-level dry run and needs no GPU.
