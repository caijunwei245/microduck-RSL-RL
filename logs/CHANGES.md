# Code changes from the optimization-plan objective (started 2026-09-23, kept current)

Everything below is committed and pushed on `develop`: `a180e58` (the 24-file rebase), `c3c4ab3`
(actuator latency + internal-phase switch), `65a425a` (evaluation tooling), `ce00299` (the 15-row
rotation + the tooling behind it). Each switch defaults to the recipe that was already running, so
every A/B keeps the baseline reproducible.

## Environment-variable switches (the house A/B pattern)

| switch | file | default | effect |
|---|---|---|---|
| `MICRODUCK_WHEEL_ROLLING` | `microduck_velocity_rollers_env_cfg.py` | `0` | pay for ROLLING (distance with a slip penalty) instead of wheel rotation |
| `MICRODUCK_ROLLING_SLIP_STD` | same | `0.18` | slip tolerance of the rolling term, calibrated from measured slip (0.019 rollers / 0.092 swizzle) |
| `MICRODUCK_ROLLER_HEIGHT_BAND` | same | off | override the trunk-height band (`"lo,hi"`); the arm used `0.130,0.145` = the measured roller stand |
| `MICRODUCK_ROLLERS_NOSKATE` | same | `0` | delete `single_support` / `skating_air_time` / `glide` (worth +20 % on rollers, neutral on swizzle) |
| `MICRODUCK_ROLLER_TALL_STAGE2` | same | `0` | **two-stage recipe**: crouched band + muted skating terms, then the band lerps to the stand while they fade in |
| `MICRODUCK_SUSTAINED_WALK` / `_SPEED` | `microduck_velocity_env_cfg.py` | `0.0` / `0.3` | pin a fraction of envs on a forward command held 25 s (the deployment operating point) |
| `MICRODUCK_YAW_RANGE` | `microduck_velocity_env_cfg.py` | `0.5` | top of the commanded yaw-rate range, i.e. the rate the runtime holds at deploy time |
| `MICRODUCK_SUSTAINED_TURN` | same | `0.0` | pin a fraction of envs IN PLACE at the top of the yaw range, no resample for the episode (the turn analogue of the walk pin) |
| `MICRODUCK_SPIN_YAW_EMA` | `microduck_spin_env_cfg.py` | `0` (=off) | tau (s) of the EMA filter on the spin-rate reward; prices the escapable sustained rotation |
| `MICRODUCK_SPIN_RATE_MAX` | same | `3.0` | the commanded spin rate; 3.0 is ~3x what the policy reaches, which made "spin while down" optimal |
| `MICRODUCK_SPIN_FEETFLAT_MULT` | same | `1.0` | scale on the `feet_flat` penalty (a pivoting biped must scuff) |
| `MICRODUCK_SITSTAND_HOLD` | `microduck_sitstand_env_cfg.py` | `0.0` | pin a fraction of envs on ONE posture command for 25 s; half are commanded the OPPOSITE of their spawn |
| `MICRODUCK_STALL_TILT_G` | standup / velstand / roller_standup | unset | tilt clause on `recovery_stall`; **solved the floor flip: 0/184 -> 184/184** |
| `MICRODUCK_STANDUP_SHARP_HOLD` | `microduck_standup_env_cfg.py` | `0` | stop annealing the last-mile price (measured: no effect - the park is an economy problem) |
| `MICRODUCK_ROLLER_STANDUP_FACEUP_ROLL` | `microduck_roller_standup_env_cfg.py` | `0` | spread face-up starts partway along the roll (reverse-curriculum port from standup) |
| `MICRODUCK_CROUCH_DELTA` | `microduck_roller_crouch_env_cfg.py` | `0.0` | weight of the potential-based phase-progress delta (return-leg pricing) |
| `MICRODUCK_KICK_FOOT` | `microduck_ball_kick_env_cfg.py` | `right` | train the **left**-footed kick (never trained before) |
| `MICRODUCK_BALL_NOISE_SCALE` | same | `1.0` | diagnostic: how much of the 52 % hit rate is ball-spawn noise |
| `MICRODUCK_INTERNAL_PHASE` | ground_pick / roller_crouch / spin / sitstand cfgs | `0` | hide the phase from the ACTOR (zeroed twist slot, as the daemon sends) while the reward and critic keep it — the publishable shape, see `publishability.md` |
| `MICRODUCK_ACTUATOR_LAG` / `MICRODUCK_ACTUATOR_LAG_HOLD` | `robot/microduck_constants.py` | **`"1,2"` / `1000`** (changed 2026-09-27) | control-step actuator delay range, and how many steps to hold it. The default is now the **measured** envelope (1–2 steps = 20–40 ms) held coherently for a whole episode, instead of the declared 3–6 dithered every step — see `actuator_latency.md`; `"3,6"` + `0` reproduces the old recipe for A/B |
| `MICRODUCK_WARM_START` | `tasks/mdp.py` (Patch 6) | unset | restart `common_step_counter` + the iteration counter after a checkpoint load, so step-based curricula do not jump to their final stage in iteration 1 |

## New MDP terms / classes

* `mdp.height_band_curriculum` — an **interpolating** band ramp (the repo's other curricula step).
* `mdp.crouch_phase_progress_delta` — unfarmable progress along the commanded crouch phase.
* `mdp.spin_rate_track(..., yaw_ema_tau=)` — optional EMA filtering of the priced yaw rate.
* `VelocityCommandCommandOnly._sustained_walk_bucket` + cfg fields `rel_sustained_walk_envs`,
  `sustained_walk_speed`, `sustained_walk_hold_s`.
* `SitStandCommand._hold_posture_bucket` + cfg fields `rel_hold_envs`, `hold_s`.
* `recovery_stall_termination(..., tilt_clause_g=)` — the lever that solved the floor flip.
* `mdp.internal_phase_enabled()` / `mdp.hide_phase_from_actor(cfg)` — deep-copies the obs term so the
  critic keeps its phase scale while the actor's is zeroed (mjlab hands the SAME `ObservationTermCfg`
  object to both groups, so zeroing one silently zeroed the other until this was written).
* `robot/microduck_constants._actuator_lag()` — env-var-driven delay range, wired into
  `_BAM_ACTUATOR_KWARGS` (`delay_min_lag` / `delay_max_lag` / `delay_update_period`).

## Tools

* `scripts/warm_start_actor.py` — cross-family warm start when only the CRITIC differs (74D standup
  vs 76D velstand): keeps the base critic/normalizer, swaps in the donor actor, zeroes the Adam
  moments, resets the iteration counter.
* `scripts/infer_policy.py --crouch` — drives the crouch PHASE in the twist slot (period, wrapping);
  without it the crouch policy was rehearsed as if it were a roller policy, which is how its skill
  was mis-judged for a week. Also prints `tilt=<deg>` on its `[vel 1s avg]` line.
* `logs/acceptance_gate.sh` — one command -> 5-round demonstration (multi-seed, aggregate %) + the
  tilt-keyed family evaluation + a PASS/PARTIAL/FAIL verdict.
* `logs/skill_demo.py` — `--seed`, `--ckpt`, `--turn`, `--video`, `--log-cmd`, fair per-skill
  criteria (steady-state window, periodic tasks, terrain-relative height, no spawn-induced fall gate,
  bare reward-sum keys); **drops `fell_over` for rows that pin the robot on the floor**, without which
  a pinned row measures the env's own spawn mix.
* `logs/family_eval.py` — sustained recovery criterion, tilt-keyed buckets, spawn-pose diagnostics,
  body-speed/wheel-slip readout, episode min/max/span of trunk z.
* `logs/demo_finish.py` — per-skill clips → the reel (`all_skills_2rounds.mp4`) + the contact sheet,
  via the ffmpeg concat demuxer rather than holding 7 minutes of frames in RAM.
* `logs/turn_report.py` + `logs/turn_queue.sh` — the pre-registered turn dead-zone experiment
  (`logs/turn_deadzone_plan.md`): four arms, per-round median achieved yaw, decision rules fixed
  before the runs.

## Tests

`274 → 355 passed, 1 skipped` (8 s, CPU-only), including `test_roller_rolling_reward.py` (16),
`test_standup_floor_flip.py` (12), `test_roller_tall_curriculum.py` (10),
`test_sustained_walk_bucket.py` (12), the per-family cfg tests, and `test_aarch64_cuda_torch.py`.
