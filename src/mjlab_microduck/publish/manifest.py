"""The policy manifest (schema 2) and the checks a published policy has to pass.

One vocabulary for two shapes — a single-policy repo (fields at the top level) and the official
set (the same fields per entry under ``policies``). This module writes the first; the daemon
(`pollen-robotics/microduck`, ``updater/src/policy.rs`` and ``robotd-params``) reads both. The
contract is `docs/policy-manifest.md` over there; the numbers below are what the daemon publishes
in ``duck_ipc_proto`` and refuses a policy for disagreeing with.

Deliberately free of mjlab / torch imports so the tests run on a laptop in milliseconds and the
CLI can validate an ONNX file without a GPU.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

SCHEMA_VERSION = 2
# `duck_ipc_proto`: the daemon refuses a policy whose manifest disagrees with these, and refuses
# at load a network whose graph does. 61 = 48 proprioception + 13 command; 14 = the servos.
MODEL_API = 1
OBS_LEN = 61
ACTION_LEN = 14
ROBOT: dict[str, Any] = {"model": "microduck", "hw_rev": 1, "servos": "xl330", "control_hz": 50}

# The one `.onnx` a repo carries. The daemon takes the sole `.onnx` in a repo and refuses several.
POLICY_FILE = "policy.onnx"

Kind = Literal["episodic", "perpetual"]
KINDS: tuple[str, ...] = ("episodic", "perpetual")

ZERO_TWIST: tuple[float, float, float] = (0.0, 0.0, 0.0)

# The daemon's policy slots, for a gait's `slot` hint (display-only: `robotctl policy load <slot>`).
SLOTS: tuple[str, ...] = ("walk", "stand", "sitstand", "ground_pick", "kick_left", "kick_right", "roulade")


class ManifestError(ValueError):
    """A manifest that the daemon would refuse, or that would load and run wrongly."""


@dataclass(frozen=True)
class Provenance:
    """Where the weights came from. Display-only for the daemon; the part people skip by hand."""

    task_id: str | None = None
    repo: str = "pollen-robotics/microduck_rl"
    commit: str | None = None
    branch: str | None = None
    dirty: bool | None = None
    run: str | None = None
    checkpoint: int | None = None
    source_file: str | None = None
    exported: str = field(default_factory=lambda: _now_utc())

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def git_provenance(repo_root: Path | None = None) -> dict[str, Any]:
    """`commit`, `branch`, `dirty` of the checkout the export ran from, or `{}` outside git."""
    root = str(repo_root or Path(__file__).resolve().parents[3])

    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", "-C", root, *args], capture_output=True, text=True, check=True, timeout=10
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    commit = git("rev-parse", "--short=9", "HEAD")
    if commit is None:
        return {}
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "branch": branch, "dirty": bool(status)}


def build_manifest(
    *,
    name: str,
    kind: str,
    description: str,
    duration_s: float | None = None,
    chain: bool = False,
    unwind_s: float | None = None,
    idle: tuple[float, float, float] = ZERO_TWIST,
    action_scale: float | None = None,
    entry_pose: str = "standing",
    slot: str | None = None,
    command_help: dict[str, Any] | None = None,
    training: dict[str, Any] | None = None,
    eval: dict[str, Any] | None = None,
    command_encoding: str = "constant",
    period_s: float | None = None,
    end_phase: float | None = None,
    sit: float = 1.0,
    stand: float = 0.0,
    ramp_s: float | None = None,
) -> dict[str, Any]:
    """A single-policy manifest the daemon loads without surprises.

    Three command encodings are supported, matching what the daemon drives (see `validate_manifest`
    and the official set in `tests/test_publish_manifest.py`):

    * ``constant`` (default) — a fixed twist.
    * ``phase`` — the daemon writes ``[cos(2*pi*phi), sin(2*pi*phi), 0]`` and sweeps ``phi`` over
      ``period_s`` seconds up to ``end_phase``. This is the encoding every phase-driven task in this
      repo already uses (ground_pick, roller_crouch, spin), which is why the actor's command slot has
      that shape; declaring it means the VALIDATED policy ships unchanged, with no internal clock and
      no retraining.
    * ``posture_flag`` — a binary flag in the twist slot that the owner toggles (sitstand).

    **Provenance of the claim, stated because it is the one thing this repo cannot test**: the format
    is accepted by `validate_manifest` and is used by the official set as uploaded 2026-09-02
    (`alpha_sitstand.onnx` = ``scripted`` + ``posture_flag`` + ``ramp_s``/``unwind_s``;
    `alpha_ground_pick.onnx` = ``episodic`` + ``phase`` + ``period_s``/``end_phase``). Whether the
    daemon ACCEPTS a community-published policy declaring them (as opposed to the official set) is a
    decision in the `microduck` repo, so `install_commands` prints a comment saying which driver the
    policy expects. Build and validate these manifests, then confirm on hardware.
    """
    if command_encoding not in ("constant", "phase", "posture_flag"):
        raise ManifestError(
            f"command_encoding must be constant, phase or posture_flag, not {command_encoding!r}"
        )
    if command_encoding == "constant":
        if kind not in KINDS:
            raise ManifestError(f"kind must be one of {KINDS}, not {kind!r}")
    elif command_encoding == "phase":
        if kind not in ("episodic", "scripted"):
            raise ManifestError(f"a phase-driven skill is episodic or scripted, not {kind!r}")
        if period_s is None or period_s <= 0:
            raise ManifestError("a phase command needs period_s > 0")
        if end_phase is None or not 0.0 <= end_phase <= 1.0:
            raise ManifestError("a phase command needs end_phase in [0, 1]")
        if duration_s is None or duration_s <= 0:
            raise ManifestError("a phase-driven skill still ends itself: give duration_s > 0")
    else:  # posture_flag
        if kind != "scripted":
            raise ManifestError(f"a posture-flag policy is scripted, not {kind!r}")
        if ramp_s is not None and ramp_s <= 0:
            raise ManifestError("ramp_s must be > 0 when given")
    if ramp_s is not None and command_encoding != "posture_flag":
        # Refuse rather than drop it: this argument was accepted and silently discarded once, so the
        # sitstand manifest declared no glide while the official set's `alpha_sitstand` declares
        # `ramp_s: 2.0`. An argument that does not reach the manifest is worse than a rejected one.
        raise ManifestError("ramp_s describes the glide a posture flag starts; it needs posture_flag")
    if not name or "/" in name or name != name.strip():
        raise ManifestError(f"name must be a bare word a client can ask for, not {name!r}")
    if kind == "episodic":
        if duration_s is None or duration_s <= 0:
            raise ManifestError(
                "an episodic policy ends itself: say how long it runs with duration_s > 0"
            )
        if unwind_s:
            raise ManifestError(
                "an episodic policy is already back when duration_s is up; unwind_s is for perpetual"
            )
    else:
        # Two things are perpetual: a gait, which lives in a slot (`policy load walk <repo>`) and
        # needs nothing here, and a held pose like the flamingo, which the owner runs as a
        # one-shot with `policy add --hold` and which then needs `unwind_s` so the robot is not
        # let go of on one foot. `unwind_s` is what says which.
        if duration_s is not None:
            raise ManifestError(
                "a perpetual policy has no length of its own; leave duration_s unset "
                "(a gait runs until told otherwise; a held pose gets --hold when added as a skill)"
            )
        if unwind_s is not None and unwind_s <= 0:
            raise ManifestError("unwind_s must be > 0 when given")
        if chain:
            raise ManifestError("chain is for episodic one-shots a held button repeats")
    if slot is not None and slot not in SLOTS:
        raise ManifestError(f"slot must be one of {SLOTS}, not {slot!r}")
    if action_scale is not None and not 0 < action_scale <= 2.0:
        raise ManifestError(f"action_scale {action_scale} is outside (0, 2]")
    if len(idle) != 3:
        raise ManifestError("idle is a 3-vector twist")

    if command_encoding == "constant":
        command: dict[str, Any] = {
            "encoding": "constant",
            "idle": [float(v) for v in idle],
            "twist": "unused (zeros)",
            "head": "unused (zeros)",
            "body": "unused (zeros)",
        }
    elif command_encoding == "phase":
        command = {
            "encoding": "phase",
            "period_s": float(period_s),          # type: ignore[arg-type]
            "end_phase": float(end_phase),        # type: ignore[arg-type]
            "idle": [float(v) for v in idle],
            "twist": "[cos(2*pi*phi), sin(2*pi*phi), 0]",
            "head": "unused (zeros)",
            "body": "unused (zeros)",
        }
    else:
        command = {
            "encoding": "posture_flag",
            "sit": float(sit),
            "stand": float(stand),
            "idle": [float(v) for v in idle],
            "twist": "[flag, side, 0]",
            "head": "unused (zeros)",
            "body": "unused (zeros)",
        }
    if command_help:
        command.update(command_help)

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model_api": MODEL_API,
        "obs_len": OBS_LEN,
        "action_len": ACTION_LEN,
        "robot": dict(ROBOT),
        "name": name,
        "kind": kind,
        "entry_pose": entry_pose,
        "description": description,
        "command": command,
    }
    if kind == "episodic":
        manifest["duration_s"] = float(duration_s)  # type: ignore[arg-type]
        manifest["chain"] = bool(chain)
    else:
        manifest["duration_s"] = None
        if ramp_s is not None:
            # The official `alpha_sitstand` entry carries `ramp_s` (2.0 s) next to `unwind_s`; it is
            # the glide the owner should expect after a flag flip, so it belongs in the manifest and
            # not only in the flag that asked for it.
            manifest["ramp_s"] = float(ramp_s)
        if unwind_s is not None:
            manifest["unwind_s"] = float(unwind_s)
    if slot is not None:
        manifest["slot"] = slot
    if action_scale is not None:
        manifest["action_scale"] = float(action_scale)
    if training:
        manifest["training"] = training
    if eval:
        manifest["eval"] = eval
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Refuse what the daemon would refuse, plus the mistakes it would load and run wrongly.

    Accepts both shapes and any schema version, because absence is not evidence — a repo is under
    no obligation to carry any field. Only a claim that is present and wrong fails.
    """
    if "policies" in manifest:
        for entry in manifest["policies"]:
            if "file" not in entry:
                raise ManifestError("every set entry needs a `file`")
            validate_manifest({k: v for k, v in entry.items() if k != "file"})
        return
    if (obs := manifest.get("obs_len")) is not None and obs != OBS_LEN:
        raise ManifestError(f"obs_len {obs}: this robot builds {OBS_LEN}")
    if (act := manifest.get("action_len")) is not None and act != ACTION_LEN:
        raise ManifestError(f"action_len {act}: this robot has {ACTION_LEN}")
    if (api := manifest.get("model_api")) is not None and api > MODEL_API:
        raise ManifestError(f"model_api {api}: this repo targets {MODEL_API}")
    model = (manifest.get("robot") or {}).get("model")
    if model is not None and model.lower() != ROBOT["model"]:
        raise ManifestError(f"robot.model {model!r}: this is a {ROBOT['model']} policy repo")
    kind = manifest.get("kind")
    if kind is not None and kind not in (*KINDS, "scripted"):
        raise ManifestError(f"kind {kind!r} is not one of episodic, perpetual, scripted")
    encoding = (manifest.get("command") or {}).get("encoding")
    if encoding is not None and encoding not in ("constant", "phase", "posture_flag"):
        raise ManifestError(f"command.encoding {encoding!r} is not one the daemon drives")
    if kind == "episodic" and encoding in (None, "constant"):
        duration = manifest.get("duration_s")
        if duration is None or duration <= 0:
            raise ManifestError("an episodic constant-command policy needs duration_s > 0")
    idle = (manifest.get("command") or {}).get("idle")
    if idle is not None and len(idle) != 3:
        raise ManifestError("command.idle is a 3-vector twist")
    ramp = manifest.get("ramp_s")
    if ramp is not None and ramp <= 0:
        raise ManifestError("ramp_s must be > 0 when present: it is the glide after a flag flip")


# ---------------------------------------------------------------------------------------------
# The ONNX file: the shape gate the daemon applies at load, applied before the upload.


@dataclass(frozen=True)
class OnnxShape:
    input_name: str
    output_name: str
    obs_len: int
    action_len: int


def inspect_onnx(path: Path) -> OnnxShape:
    """The graph's single input and output widths, as the daemon checks them at load."""
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    graph = model.graph
    initializers = {i.name for i in graph.initializer}
    inputs = [i for i in graph.input if i.name not in initializers]
    if len(inputs) != 1 or len(graph.output) != 1:
        raise ManifestError(
            f"{path.name}: expected one input and one output, found "
            f"{[i.name for i in inputs]} -> {[o.name for o in graph.output]}"
        )

    def last_dim(value) -> int:
        dims = value.type.tensor_type.shape.dim
        if not dims:
            raise ManifestError(f"{path.name}: {value.name} has no shape")
        last = dims[-1]
        if not last.HasField("dim_value"):
            raise ManifestError(f"{path.name}: {value.name}'s last dimension is symbolic")
        return int(last.dim_value)

    return OnnxShape(
        input_name=inputs[0].name,
        output_name=graph.output[0].name,
        obs_len=last_dim(inputs[0]),
        action_len=last_dim(graph.output[0]),
    )


def check_onnx(path: Path) -> OnnxShape:
    """Refuse a file the daemon would refuse at load: wrong widths, or one that is not 61 -> 14."""
    if not path.exists():
        raise ManifestError(f"{path}: no such file")
    shape = inspect_onnx(path)
    if shape.obs_len != OBS_LEN:
        raise ManifestError(
            f"{path.name}: observation width is {shape.obs_len}, the robot builds {OBS_LEN} "
            "(a 51-D policy is the legacy 3-value-command family, which the daemon refuses)"
        )
    if shape.action_len != ACTION_LEN:
        raise ManifestError(f"{path.name}: {shape.action_len} actions, the robot has {ACTION_LEN}")
    return shape


def smoke_run_onnx(path: Path, steps: int = 50, seed: int = 0) -> None:
    """Run the network on plausible inputs and refuse a NaN/inf or a saturated output.

    Not a physics rehearsal — `scripts/infer_policy.py` is that — but it catches a broken export
    (an un-baked normalizer producing NaNs on raw observations, a graph that will not execute)
    before anything is uploaded.
    """
    import numpy as np
    import onnxruntime as ort

    shape = inspect_onnx(path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(seed)
    obs = np.zeros((1, shape.obs_len), dtype=np.float32)
    outputs = []
    for _ in range(steps):
        (out,) = session.run([shape.output_name], {shape.input_name: obs})
        if not np.all(np.isfinite(out)):
            raise ManifestError(f"{path.name}: the network produced a non-finite action")
        outputs.append(out)
        # Feed the action back into the last-action slots and jitter the rest, the way an
        # observation evolves on the robot; enough to leave the zero point.
        obs = rng.normal(0.0, 0.05, size=obs.shape).astype(np.float32)
        obs[0, -ACTION_LEN - 13 : -13] = np.clip(out[0], -1, 1)
    spread = float(np.std(np.stack(outputs)))
    if spread == 0.0:
        raise ManifestError(f"{path.name}: the network's output never changes; is it a real policy?")


# ---------------------------------------------------------------------------------------------
# What else goes in the repo.


def install_commands(manifest: dict[str, Any], repo_id: str) -> str:
    """The `robotctl` lines that put this policy on a robot — one story per shape.

    Episodic: a skill, length from the manifest. Perpetual with `unwind_s`: a held pose the owner
    runs as a skill with `--hold`. Perpetual without: a gait, loaded into a slot.
    """
    name = manifest["name"]
    encoding = (manifest.get("command") or {}).get("encoding", "constant")
    if encoding == "phase":
        # The daemon drives the phase itself; the policy ships unchanged. The comment is the part a
        # human needs: it says which driver the policy expects, which is also the thing to confirm on
        # hardware before trusting the upload (phase/posture_flag may be official-set only).
        return (f"# expects command.encoding=phase (period_s={manifest['command'].get('period_s')}, "
                f"end_phase={manifest['command'].get('end_phase')}) — the daemon writes "
                f"[cos(2*pi*phi), sin(2*pi*phi), 0]\n"
                f"sudo robotctl policy add {name} {repo_id}\nrobotctl robot do {name}")
    if encoding == "posture_flag":
        return (f"# expects command.encoding=posture_flag (sit={manifest['command'].get('sit')}, "
                f"stand={manifest['command'].get('stand')}, ramp_s={manifest.get('ramp_s')}) "
                f"— the owner toggles the flag\n"
                f"sudo robotctl policy add {name} {repo_id}"
                + (f" --hold {manifest['unwind_s']}" if manifest.get("unwind_s") else "")
                + f"\nrobotctl robot do {name}")
    if manifest["kind"] == "episodic":
        return f"sudo robotctl policy add {name} {repo_id}\nrobotctl robot do {name}"
    if manifest.get("unwind_s") is not None:
        return f"sudo robotctl policy add {name} {repo_id} --hold <seconds>\nrobotctl robot do {name}"
    slot = manifest.get("slot", "<slot>")
    return f"sudo robotctl policy load {slot} {repo_id}"


def render_readme(manifest: dict[str, Any], repo_id: str) -> str:
    """A model card that says how to run the policy on a robot, generated so it cannot go stale."""
    kind = manifest["kind"]
    name = manifest["name"]
    description = manifest.get("description", "")
    training = manifest.get("training", {})
    run = install_commands(manifest, repo_id)
    encoding = (manifest.get("command") or {}).get("encoding", "constant")
    if encoding == "phase":
        timing = (f"Runs {manifest['duration_s']} s while the daemon sweeps the phase over "
                  f"{manifest['command'].get('period_s')} s (to {manifest['command'].get('end_phase')}); "
                  f"the policy is the validated phase-commanded one, unchanged.")
    elif encoding == "posture_flag":
        timing = ("Holds the commanded posture until the flag is toggled back; the daemon drives "
                  "`command.sit` / `command.stand`.")
        if manifest.get("ramp_s") is not None:
            timing += (f" A flip is answered by a constant-rate glide over `ramp_s` "
                       f"({manifest['ramp_s']} s), not a jump.")
    elif kind == "episodic":
        timing = f"Runs {manifest['duration_s']} s and returns itself to a standing pose."
        if manifest.get("chain"):
            timing += " Holding the button chains another run."
    elif manifest.get("unwind_s") is not None:
        timing = (
            f"Holds until told otherwise; the daemon drives `command.idle` for "
            f"{manifest['unwind_s']} s before handing back to the gait."
        )
    else:
        slot = manifest.get("slot")
        timing = "Runs until told otherwise" + (
            f" — a gait for the `{slot}` slot." if slot else " — a gait, loaded into a policy slot."
        )
    lines = [
        "---",
        "tags:",
        "- microduck",
        "- robotics",
        "- reinforcement-learning",
        "- onnx",
        "library_name: onnx",
        "---",
        "",
        f"# {name}",
        "",
        description,
        "",
        f"A **{kind}** policy for the [microduck](https://github.com/pollen-robotics/microduck) "
        f"({OBS_LEN}-D observation, {ACTION_LEN} actions, {ROBOT['control_hz']} Hz). {timing}",
        "",
        "## Run it on a robot",
        "",
        "```bash",
        run,
        "```",
        "",
        "The observation normalizer is baked into `policy.onnx`; feed raw observations.",
        "`manifest.json` follows schema 2 of the microduck policy manifest "
        "(`docs/policy-manifest.md` in the daemon repo).",
    ]
    if training:
        lines += ["", "## Training", ""]
        for key in ("task_id", "repo", "branch", "commit", "run", "checkpoint", "exported"):
            if key in training:
                lines.append(f"- **{key}**: `{training[key]}`")
        if training.get("dirty"):
            lines.append("- exported from a checkout with uncommitted changes")
    return "\n".join(lines) + "\n"


def dump_manifest(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, indent=2) + "\n"
