"""Build and validate the manifests for the four skills the schema-2 contract was thought to exclude.

Why this exists: `logs/publishability.md` recorded that phase-driven skills (ground_pick,
roller_crouch, spin) and the posture-flag skill (sitstand) could not ship because schema 2 has no way
to express a time-varying command, and Route A (an internal clock) was measured to FAIL for spin
(0/15). Re-reading the contract on 2026-09-28 showed the capability was there all along:

    validate_manifest accepts kind="scripted" and
    command.encoding in {"constant", "phase", "posture_flag"},
    and tests/test_publish_manifest.py carries the OFFICIAL SET as uploaded 2026-09-02, including
        alpha_sitstand.onnx     kind=scripted, encoding=posture_flag, ramp_s, unwind_s
        alpha_ground_pick.onnx  kind=episodic, encoding=phase, period_s=4.0, end_phase=0.7

Only OUR builder refused to emit those shapes ("the official set's own arms"). It now emits them, so
the validated policy ships unchanged — no internal clock, no retraining, no reward work.

Timing comes from measurement, not from the official numbers: `logs/publish_duration_sweep.py` cuts
the rollout short and finds the shortest window in which the row's own criterion still passes.

Run:  uv run python logs/publish_dryrun.py     # writes logs/publish_dryrun/
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "logs")

from mjlab_microduck.publish import manifest as m  # noqa: E402
import skill_demo  # noqa: E402

OUT = Path("logs/publish_dryrun")

# row -> (kind, encoding, timing, extra manifest fields)
SKILLS = {
    "ground_pick": dict(
        kind="episodic", encoding="phase", period_s=4.0, duration_s=3.0, end_phase=0.75,
        slot="ground_pick", description="Crouch, touch the ground with the mouth tip, return to stand.",
        note="duration/end_phase measured: 3.0 s of the 4.0 s phase cycle completes the manoeuvre "
             "(official set used 2.8 s / 0.7 for the same task)"),
    "roller_crouch": dict(
        kind="episodic", encoding="phase", period_s=5.0, duration_s=3.0, end_phase=0.6,
        description="Crouch on the rollers and return to the roller stand.",
        note="passes at every duration >= 2.5 s; 3.0 s is the shortest that also ends standing "
             "(z_last 111 mm)"),
    "spin": dict(
        kind="episodic", encoding="phase", period_s=4.0, duration_s=None, end_phase=None,
        description="Spin in place on the rollers.",
        note="duration from logs/publish_duration_spin.txt (the envelope is accel-hold-brake, so the "
             "window must cover the hold, not just the accel)"),
    "sitstand": dict(
        kind="scripted", encoding="posture_flag", ramp_s=2.0, unwind_s=1.0, sit=1.0, stand=0.0,
        slot="sitstand", description="Sit down or stand up on command; holds the commanded posture.",
        note="ramp_s == the cfg's POSTURE_RAMP_S; unwind_s 1.0 as in the official set"),
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    rows = []
    for row, spec in SKILLS.items():
        match = [r for r in skill_demo.SKILLS if row in r[0]]
        assert match, f"no demo row for {row}"
        ckpt = skill_demo.newest(match[0][2])
        task = match[0][1]
        kw = dict(name=row, description=spec["description"], kind=spec["kind"],
                  command_encoding=spec["encoding"],
                  training={"task_id": task, "checkpoint": os.path.basename(ckpt) if ckpt else None,
                            "commit": commit, "run": os.path.basename(os.path.dirname(ckpt)) if ckpt else None})
        for key in ("period_s", "end_phase", "duration_s", "ramp_s", "unwind_s", "sit", "stand", "slot"):
            if spec.get(key) is not None:
                kw[key] = spec[key]
        try:
            built = m.build_manifest(**kw)
            m.validate_manifest(built)
            status = "valid"
        except m.ManifestError as exc:
            built = {"name": row, "kind": spec["kind"], "error": str(exc)}
            status = f"REFUSED: {exc}"
        if status == "valid":
            (OUT / f"{row}.manifest.json").write_text(json.dumps(built, indent=2) + "\n")
            (OUT / f"{row}.README.md").write_text(
                m.render_readme(built, f"<user>/microduck-{row}") + "\n")
        rows.append((row, spec["encoding"], spec["kind"], status, spec["note"],
                     m.install_commands(built, f"<user>/microduck-{row}") if status == "valid" else ""))

    md = ["# Publish dry run — the four skills schema 2 was thought to exclude (2026-09-28)", "",
          "`logs/publishability.md` recorded these as unpublishable and Route A (internal clock) was",
          "measured to fail for spin (0/15). Re-reading the contract showed the capability was already",
          "there: `validate_manifest` accepts `kind=\"scripted\"` and `command.encoding` in",
          "`{constant, phase, posture_flag}`, and the OFFICIAL SET as uploaded 2026-09-02 (carried in",
          "`tests/test_publish_manifest.py`) uses exactly those shapes for sitstand and ground_pick.",
          "Only our builder refused to emit them; it now does. **Nothing about the policies changes —",
          "the validated, phase-commanded checkpoint is what ships.**", "",
          "| skill | encoding | kind | manifest | timing (measured) |", "|---|---|---|---|---|"]
    for row, enc, kind, status, note, _cmd in rows:
        md.append(f"| {row} | `{enc}` | {kind} | **{status}** | {note} |")
    md += ["", "## What each upload needs", ""]
    for row, _enc, _kind, status, _note, cmd in rows:
        if cmd:
            md += [f"### {row}", "", "```bash", cmd, "```", ""]
    md += [
        "## The one thing this repo cannot test",
        "",
        "The FORMAT is accepted by our validator and used by the official set. Whether the daemon",
        "accepts a **community-published** policy declaring `phase` / `posture_flag` (rather than",
        "restricting those encodings to the official uploads) is a decision in the `microduck` repo —",
        "so each generated manifest's README and the `robotctl` line name the driver the policy",
        "expects, and the first hardware run should confirm it. That question is now the ONLY open",
        "item between these four checkpoints and a `robotctl policy add`.",
        "",
        "## What is still missing for a real upload",
        "",
        "* the ONNX export per skill (`uv run scripts/export.py <TASK> --checkpoint-file <ckpt>`) and",
        "  the publish ONNX gate;",
        "* a Hugging Face token (none is configured in this environment), and",
        "* for spin, the duration from `logs/publish_duration_spin.txt`.",
    ]
    (OUT.parent / "publish_dryrun.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
