# Final checkpoints — 2026-09-23/24 session

Every checkpoint that backs a measured result in `logs/STATE_OF_PROJECT.md` and
`logs/GOAL_SUMMARY.md` (both kept locally; `logs/` is gitignored). Copied here from the run
directories under `logs/rsl_rl/`, which are NOT in git, so this README is the provenance record.

**Recipe defaults matter**: every switch below defaults to the previously-running recipe, so a
checkpoint + the env var(s) listed is enough to reproduce the behaviour. Load with
`uv run play <TASK_ID> --checkpoint ...` or export with
`uv run scripts/export.py <TASK_ID> --checkpoint-file <file> --onnx-file out.onnx`.

## Headline results

| file | task | recipe / env vars | measured |
|---|---|---|---|
| `standup_tiltstall_model_54994.pt` | `Mjlab-StandUp-Flat-MicroDuck` | `MICRODUCK_STALL_TILT_G=-0.9` | **floor flip 0/184 -> 184/184**; terminal 116.1 mm / g -0.993; rehearsal 116.0 mm at 0.3 deg from prone, 0.2 deg from supine |
| `standup_tiltstall_deployment.onnx` | same | exported from the above | the artifact validated in the deployment rehearsal (`--start-pose prone/supine`) |
| `velstand_fromstandup_model_5999.pt` | `Mjlab-VelStand-Flat-MicroDuck` | actor transplanted from the standup policy (critic/normalizer kept from velstand) | **floor flip 1/128 -> 127/128**, 0.996 forced-floor; rehearsal 113.9 mm at 0.9 deg; walks 0.00 m/s (recovery-only specialist) |
| `spin_ratecap_model_4998.pt` | `Mjlab-Spin-Flat-MicroDuck` | `MICRODUCK_SPIN_RATE_MAX=1.0` | **spin 0/5 -> 14/15 (93 %)**, upright at 112.0 mm / g -1.0 |
| `spin_warmstand_model_3999.pt` | same | actor transplanted from `roller_standup2` | **spin 0/5 -> 15/15 (100 %)**, 114.1 mm / g -1.0 |
| `rollers_tall2000_model_2999.pt` | `Mjlab-Velocity-Flat-MicroDuck-Rollers` | `MICRODUCK_WHEEL_ROLLING=1 MICRODUCK_ROLLER_TALL_STAGE2=2000` | two-stage recipe: **0.1911 m/s at 138.2 mm** (vs crouched 0.3500 @ 116, tall swap 0.2099 @ 138.9, naive combination 0.0459), demo 15/15 |
| `rollers_noskate_model_1999.pt` | same | `MICRODUCK_WHEEL_ROLLING=1 MICRODUCK_ROLLERS_NOSKATE=1` | the fastest wheeled recipe: **0.3500 m/s (117 % of the commanded push)** at 116 mm, demo 5/5 |
| `roller_standup_stall_model_2498.pt` | `Mjlab-RollerStandUp-Flat-MicroDuck` | `MICRODUCK_STALL_TILT_G=-0.9` | **76 % -> 100 % (15/15)**; the near-wall spawn spread added nothing measurable |
| `ball_kick_left_model_1499.pt` | `Mjlab-BallKick-Flat-MicroDuck` | `MICRODUCK_KICK_FOOT=left` | the **first left-footed** kick: 9/15 (60 %), parity with the right foot (52 %) |
| `walk_sustained_model_69996.pt` | `Mjlab-Velocity-Flat-MicroDuck` | `MICRODUCK_SUSTAINED_WALK=0.2` | walk-from-step-0: 48 % -> 56 % over 25 rounds |
| `walk_turn03_model_70996.pt` | same | `MICRODUCK_YAW_RANGE=0.3 MICRODUCK_SUSTAINED_TURN=0.3` | **negative result**: pinning 0.3 rad/s for 3k iterations did NOT open the in-place dead zone (0/5, achieved 0.035-0.066, slightly worse) |

## Arms that produced the negative evidence (worth keeping)

| file | recipe | what it showed |
|---|---|---|
| `spin_ema_model_4998.pt` | `MICRODUCK_SPIN_YAW_EMA=0.5` | spin 0/15 unchanged while the training metric rose **0.60 -> 5.37 (9x)** |
| `spin_feetflat_model_4998.pt` | `MICRODUCK_SPIN_FEETFLAT_MULT=0.25` | spin 0/10 unchanged, metric 1.36 (2.3x), |yaw| steady 1.05-1.17 rad/s at ~70 deg of tilt |
| `crouch_delta2_model_2498.pt` | `MICRODUCK_CROUCH_DELTA=4.0` | 24/25 (96 %) = identical to the baseline; no gain |
| `sitstand_hold2_model_2498.pt` | `MICRODUCK_SITSTAND_HOLD=0.3` | 17/25 -> 18/25 (one round, inside noise) |

## How these were validated

`logs/acceptance_gate.sh "<skill>" <checkpoint> [task]` — a multi-seed (default 3x5) demonstration
plus the tilt-keyed family evaluation, with a PASS/PARTIAL/FAIL verdict at 80 %/50 %; and
`logs/skill_demo.py --rounds 5 --video` for the recorded demonstrations (see `logs/demo_videos/`).

## Note on size and future checkpoints

These are ~4.7 MB each (14 files, 66 MB total). The repository tracked no checkpoints before this
commit, so this directory establishes the convention: **date-stamped directory, descriptive
filenames, and a README row per file** — a checkpoint without its recipe and its measurement is not
evidence. Consider Git LFS if this grows past a few hundred MB.
