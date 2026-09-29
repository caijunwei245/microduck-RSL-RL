# Published skills — export set 2026-09-29

The deployable artifacts for the five skills that are solved, exported and staged by

```bash
bash logs/publish_solved.sh            # dry run: export + shape gate + smoke + manifest -> publish-<name>/
HF_USER=<name> bash logs/publish_solved.sh --upload    # same, then push to the Hub
```

Each `<skill>/` here is byte-for-byte what `uv run publish` writes into a Hub repo: `policy.onnx`
(obs normalizer baked in), `manifest.json` (schema 2) and `README.md` (generated from the manifest,
so it cannot go stale). The run log is `logs/publish_solved.txt`; the measured-timing story is
`logs/publish_solved_skills.md`; the per-skill evidence files are named in the table.

## The artifacts

| directory | task | checkpoint | kind / encoding | slot | duration |
|---|---|---|---|---|---|
| `spin/` | `Mjlab-Spin-Flat-MicroDuck` | `spin_warmstand_model_3999.pt` (2026-09-23 set) | episodic / `phase` | - | period 4.0 s, **2.6 s**, end_phase **0.65** |
| `roller_standup/` | `Mjlab-RollerStandUp-Flat-MicroDuck` | `roller_standup_lag12_model_1999.pt` | episodic / `constant` | - | **1.0 s** |
| `ball_kick_right/` | `Mjlab-BallKick-Flat-MicroDuck` | `ball_kick_right_kickr4_model_999.pt` | episodic / `constant` | `kick_right` | **1.5 s** |
| `ball_kick_left/` | `Mjlab-BallKick-Flat-MicroDuck` | `ball_kick_left_model_1499.pt` (2026-09-23 set) | episodic / `constant` | `kick_left` | **1.5 s** |
| `walk_turn/` | `Mjlab-Velocity-Flat-MicroDuck` | `walk_turn03_woboff_model_4998.pt` | perpetual / `constant` | `walk` | none (a gait) |

Three of the five source `.pt` files were not in `checkpoints/2026-09-23/` and are copied here
(`ball_kick_right_kickr4_model_999.pt`, `roller_standup_lag12_model_1999.pt`,
`walk_turn03_woboff_model_4998.pt`); the other two are already there under their 2026-09-23 names
(`spin_warmstand_model_3999.pt`, `ball_kick_left_model_1499.pt`). Export again with

```bash
uv run scripts/export.py <TASK_ID> --checkpoint-file <file> --onnx-file out.onnx
# or exactly what was run here:
uv run publish --task <TASK_ID> --checkpoint-file <file> --repo <user>/microduck-<skill> \
    --name <skill> --kind <kind> --duration-s <seconds> --dry-run
```

## Measured result behind each timing

Every `duration_s` is measured in this repo, never copied from the official set. Two of the five
numbers were **not** what the first read said, and both corrections are the same class of mistake —
reading a criterion against something the episode had not been asked to do.

* **spin 2.6 s / end_phase 0.65** (`logs/publish_duration_spin15.txt`). The duration sweep turns the
  spin on and off: 15/15 at 1.5-3.0 s, **0/15 at 3.5 s and 4.0 s**, while the rotation integrated over
  *every* window >= 2.5 s is 6.5-6.6 rad (one full turn). The task's own envelope commands rate 0 from
  `SPIN_BRAKE_END` (0.650 of the 4 s cycle = 2.6 s) on, so the failing windows are the ones containing
  the commanded REST segment and the row averages only the last third. 2.6 s is the brake end: one
  full turn, handed back at the commanded zero rate. The first draft carried 4.0 s, which would have
  shipped 1.4 s of commanded standstill.
* **roller_standup 1.0 s** (`logs/roller_standup_rise.txt`, `logs/roller_standup_rise.py`). The sweep
  reads "0.6 s passes 15/15, trunk ends at 139 mm" — impossible as a rise time, because the task
  spawns **50 % belly / 50 % already standing** (that bucket teaches HOLDING), so half the rounds pass
  a short window for free. Pinned one bucket at a time: belly rises in **0.20 s**, back (the hard
  direction) in **0.32 s**, standing in 0; 1.0 s covers the worst bucket three times over and every
  bucket ends standing at 139 mm. Note the pin has to go through the **curriculum**
  (`ground_state_mix` an `event_param_curriculum` rewrites the event's probabilities before each
  reset, so writing them into the event's params is a silent no-op).
* **ball_kick 1.5 s** (`logs/publish_duration_ball_kick.txt`, `..._left.txt`). The strike happens at
  step 5-8 (0.10-0.16 s), so every window from 0.5 s up already contains the kick, and the robot is
  standing (115-116 mm, 5-6 mm of trunk motion) in all of them; the ball simply keeps rolling, so
  there is no plateau to find. 1.5 s is a handoff choice: the ball is 0.30-0.44 m clear. Right and
  left measure the same (0.36 m / 0.38 m at 1.5 s).

## What backs each policy

| skill | gate (2026-09-28, measured latency envelope) | what to watch |
|---|---|---|
| spin | **15/15**, upright, 1.43-1.47 rad/s | one full turn per button press |
| roller_standup | **25/25**, `z_max` median 143 mm, upright 0.97-1.00 | rises from belly or back, then holds the roller stand |
| ball_kick right / left | **25/25 each**, median launch 1.301 m / 1.362 m | the actor never sees the ball; the old "52 %" was a world-frame scoring bug |
| walk_turn | turn row **21/25**, median 0.5541 rad/s at a commanded 0.5 | the wobble tax is off; in-place turn at gain 1.17-1.23 |

`walk_turn` loads into the `walk` slot, so it was paired against the policy that slot currently ships
before staging it (`logs/paired_walk_vs_turn.txt`, same seeds for both arms, 5 seeds x 5 rounds):
**rounds/seed delta +0.00 (5/5 seeds tied)**, `v_mean` **+0.00884 +/-0.003298** in its favour, pooled
22/25 vs 22/25. The turn row goes from standing still in a third of episodes to 21/25; the walk row
does not move.

## Install on a robot

```bash
# spin: phase-driven, the daemon sweeps [cos(2*pi*phi), sin(2*pi*phi), 0] over 4.0 s to phase 0.65
sudo robotctl policy add spin <user>/microduck-spin && robotctl robot do spin

# one-shots (constant zero command); the get-up starts from belly, back or standing
sudo robotctl policy add roller_standup  <user>/microduck-roller_standup
sudo robotctl policy add ball_kick_right <user>/microduck-ball_kick_right
sudo robotctl policy add ball_kick_left  <user>/microduck-ball_kick_left

# the gait goes into the slot rather than being run as a skill
sudo robotctl policy load walk <user>/microduck-walk_turn
```

## What is NOT verified

* **Real-robot validation is zero** for all five. Every number above is simulation.
* The **Hub upload has not happened**: no Hugging Face credential exists in the machine that produced
  these (`hf auth whoami` -> `Not logged in`), and that is the only thing missing —
  `HF_USER=<name> bash logs/publish_solved.sh --upload` finishes it. These directories are committed
  so the artifacts and their provenance are in git regardless.
* Whether the daemon accepts a **community-published** policy declaring `command.encoding: phase`
  (spin). The format validates here and the official set uses it (`alpha_ground_pick`), so the first
  hardware run is the confirmation.
* The CPU MuJoCo deployment rehearsal (`uv run scripts/infer_policy.py`, which drives the ONNX through
  the BAM actuator stack) was **not** run here: it opens a `mujoco.viewer` window and there is no
  display in this environment. It already knows these skills, e.g.
  `--walking walk_turn/policy.onnx --ang-vel-z 0.5 --new-cmd-obs` and
  `--kick-right ball_kick_right/policy.onnx --new-cmd-obs --kick-duration 1.5`.
