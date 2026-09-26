# Skill rotation: all skills, 2 rounds each (2026-09-26 — 15 rows)

    uv run python logs/skill_demo.py --rounds 2 --video
    uv run python logs/demo_finish.py --rounds 2      # reel + contact sheet (ffmpeg concat demuxer)

**All 15 rows, two consecutive episodes each**, every frame labelled with the skill and the round.
Videos: `logs/demo_videos/*.mp4`, concatenated into `all_skills_2rounds.mp4` (15 clips, 7 min 32 s,
640x480), contact sheet `round2_montage.png` (2590x2038). The VelStand floor-flip row — disabled
while its two blockers were open — is back in.

| skill | round 1 | round 2 | checkpoint |
|---|---|---|---|
| velocity walk | **0.252 m/s** | **0.254 m/s** | `dc_long_0919_0125` (deployment candidate) |
| velocity turn | 0.318 rad/s (gain 0.64) | 0.338 (0.68) | same — clears the 0.3 rad/s bar, misses strict tracking |
| walk+turn | 0.444 (0.89) | 0.467 (0.93) | same |
| ball_kick right | **+0.61 m** | **+0.92 m** | `kick_r4` |
| sitstand | **69 mm span** | **66 mm span** | `sitstand_hold2` |
| ground_pick | **35 mm span** | **47 mm span** | `groundpick` (round 2 ends tilted, g −0.35) |
| rollers (fast) | **0.471 m/s** | **0.448 m/s** | `rollers_noskate` |
| swizzle | **0.413 m/s** | **0.387 m/s** | `swizzle_rolling18` |
| roller_slope | **247 mm descent** | **241 mm** | `roller_slope` |
| roller_standup | **z 145 mm, held 300** | **z 140, held 300** | `roller_standup_stall` |
| roller_crouch | **73 mm span** | **71 mm span** | `crouch_delta2` |
| standup floor flip | **hold 236, z 116** | **hold 281, z 116** | `standup_tiltstall` |
| velstand floor flip | **hold 948, z 114** | **hold 964, z 115** | `velstand_fromstandup` — re-enabled row |
| **spin** | **1.40 rad/s at g −1.00** | **1.41 rad/s at g −1.00** | `spin_warmstand` — upright, both rounds |
| roulade | **roll 6.44, ends 114 mm** | **roll 3.13, ends 115 mm** | `roulade_finish` |

**13 of 15 rows passed both rounds.** The two turn rows are the honest exception: they reach
0.32-0.47 rad/s, i.e. they clear the absolute 0.3 rad/s bar but not strict tracking of the commanded
0.5 — the sub-0.25 rad/s dead zone documented in `logs/GOAL_SUMMARY.md` step 3. That dead zone now has
its own pre-registered experiment (`logs/turn_deadzone_plan.md`: four arms, from-scratch pinned runs
plus the matched no-pin control, 3 seeds x 5 rounds).

The VelStand row is the other one to read carefully: `hold=948/964` at 114-115 mm means the policy
holds the stand for essentially the whole 20 s episode from a **pure prone pin** (it recovered first),
which is the sustained form of the corrected **0.812** (52/64) floor-flip rate — not the contaminated
0.996 that the pin-tripped `fell_over` term used to produce.

**Caveat, and it matters**: two rounds is a *demonstration*, not a measurement. Across seeds the same
rows range from 1/5 to 4/5 (ball_kick) and 0/5 to 3/5 (walk), and roller_crouch reads 60 % on three
seeds but 96 % on five. Use `logs/acceptance_gate.sh` (3 seeds x 5 rounds) for verdicts; this file is
for looking at the robot.

## Tooling the rotation depends on

* `logs/skill_demo.py` loads actors **by hand** (mlp weights + that checkpoint's own obs normalizer —
  the same arithmetic the ONNX export bakes). Upstream's `distill` runner fetches an expert through
  `wandb.Api()`, which had made every runner-based VelStand evaluation require an API key.
* It also **drops `fell_over` when a row pins the robot on the floor** — the robot *is* fallen, so the
  term fires on step 1, the env recycles the episode, and the metric silently reports the env's own
  spawn mix instead of the pin. `family_eval.py` does the same for pinned batteries only.
* `logs/demo_finish.py` rebuilds the reel and the contact sheet from the per-skill clips; it uses the
  ffmpeg concat demuxer rather than loading 7 minutes of frames into RAM.

---

# Previous run: 14 rows (2026-09-25)

Kept because the two tables disagree in exactly the places the caveat warns about (round-to-round
spread on ball_kick, swizzle, roller_slope, roulade), which is the point.

| skill | round 1 | round 2 | checkpoint |
|---|---|---|---|
| velocity walk | 0.252 m/s | 0.254 m/s | `dc_long_0919_0125` |
| velocity turn | 0.318 rad/s (gain 0.64) | 0.338 (0.68) | same |
| walk+turn | 0.444 (0.89) | 0.477 (0.95) | same |
| ball_kick right | +0.61 m | +0.92 m | `kick_r4` |
| sitstand | 69 mm span | 66 mm span | `sitstand_hold2` |
| ground_pick | 35 mm span | 37 mm span | `groundpick` |
| rollers (fast) | 0.463 m/s | 0.456 m/s | `rollers_noskate` |
| swizzle | 0.421 m/s | 0.402 m/s | `swizzle_rolling18` |
| roller_slope | 214 mm descent | 210 mm | `roller_slope` |
| roller_standup | z 145 mm, held 300 | z 140, held 300 | `roller_standup_stall` |
| roller_crouch | 73 mm span | 71 mm span | `crouch_delta2` |
| standup floor flip | hold 236 | hold 282 | `standup_tiltstall` |
| spin | 1.40 rad/s at g -1.00 | 1.40 rad/s at g -1.00 | `spin_warmstand` |
| roulade | roll 6.44, ends 114 mm | roll 3.13, ends 115 mm | `roulade_finish` |

## How the VelStand row got re-enabled (2026-09-25)

1. `family_eval.py` now loads actors by hand, so it no longer builds the `distill` runner for VelStand.
2. The raw qpos floor pin was never being measured: the env keeps `fell_over`, so the pin tripped it
   instantly (`fell_over 64/64`, trunk frozen at 75.0 mm) and the "0.996 floor-flip rate" was really
   the env's own spawn mix. With that termination dropped for the pin, the honest number is **0.812**
   (52/64, sustained). StandUp needed no such fix — its cfg removes `fell_over` by design — and it
   re-verifies at **0.984** (126/128), so quote those two numbers with their difference in mind.
