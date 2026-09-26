# Skill rotation: all skills, 2 rounds each (2026-09-26 — 15 rows, all passing)

```bash
# the current artifacts: turn rows on the wobble-free turn policy
MICRODUCK_TURN_CKPT=logs/rsl_rl/velocity/2026-09-26_19-46-52_turn03_woboff/model_4998.pt \
  uv run python logs/skill_demo.py --rounds 2 --video
uv run python logs/demo_finish.py --rounds 2      # reel + contact sheet (ffmpeg concat demuxer)
```

**All 15 rows, two consecutive episodes each**, every frame labelled with the skill and the round.
Videos: `logs/demo_videos/*.mp4`, concatenated into `all_skills_2rounds.mp4` (15 clips, 7 min 32 s,
640x480), contact sheet `round2_montage.png` (2590x2038).

| skill | round 1 | round 2 | checkpoint |
|---|---|---|---|
| velocity walk | **0.252 m/s** | **0.254 m/s** | `dc_long_0919_0125` |
| **velocity turn** | **0.626 rad/s (gain 1.25)** | **0.601 (1.20)** | **`turn03_woboff` @4998 — wobble-free + pin** |
| **walk+turn** | **0.581 (1.16)** | **0.597 (1.19)** | same |
| ball_kick right | **+0.61 m** | **+0.92 m** | `kick_r4` |
| sitstand | **69 mm span** | **66 mm span** | `sitstand_hold2` |
| ground_pick | **35 mm span** | **42 mm span** | `groundpick` (round 2 ends tilted, g −0.40) |
| rollers (fast) | **0.465 m/s** | **0.454 m/s** | `rollers_noskate` |
| swizzle | **0.419 m/s** | **0.402 m/s** | `swizzle_rolling18` |
| roller_slope | **164 mm descent** | **160 mm** | `roller_slope` (reel-to-reel range 160-247 mm) |
| roller_standup | **z 145 mm, held 300** | **z 140, held 300** | `roller_standup_stall` |
| roller_crouch | **73 mm span** | **71 mm span** | `crouch_delta2` |
| standup floor flip | **hold 234, z 116** | **hold 281, z 116** | `standup_tiltstall` |
| velstand floor flip | **hold 952, z 114** | **hold 964, z 115** | `velstand_fromstandup` |
| spin | **1.40 rad/s at g −1.00** | **1.40 rad/s at g −1.00** | `spin_warmstand` |
| roulade | **roll 6.44, ends 114 mm** | **roll 3.13, ends 115 mm** | `roulade_finish` |

**15 of 15 rows passed both rounds.** The turn rows are the ones that changed: they run on the policy
produced by the dead-zone experiment (`logs/turn_deadzone_verdict.md`, `logs/turn_ship_verdict.md`) —
the deployed walking candidate warm-started for 5,000 iterations with the `angular_wobble` tax removed
and a 0.3 rad/s in-place command pinned into 30 % of the envs. The same rotation on the deployed
candidate alone reads **13 of 15**, with the turn rows at 0.318/0.338 rad/s (gain 0.64/0.68) — table
kept below, because that is the honest "as-deployed" number.

### The caveat that survives the fix: a DR-tail, not a systematic failure

The turn policy does not track on *every* episode. At seed 0, episodes 5-6 of six **stand still**
while commanded to turn (|yaw| 0.069 / 0.042 rad/s), ending upright at 115 mm with g −1.00 — it never
falls, it just does not rotate. Two controls localize it:

* with domain randomization off (`--play-cfg`), the same two episodes read 0.263 (gain 0.88) and 0.329
  (1.10), so most of the collapse lives in the DR draw;
* an independently trained wobble-free policy fails the **same episode indices** with nearly the same
  numbers (0.072, 0.053).

Two different policies failing identically at the same indices is a property of those episodes. In
roughly one episode in three-to-five the policy prefers standing still under an unfavourable draw.
That is a *tail* problem (gain ~1.15 whenever it turns), separate from the dead zone, which was
*systematic* (median gain 0.27 across all episodes). Next step is instrumentation of the per-episode
DR draw, not another reward term.

**Also true and unchanged**: two rounds is a *demonstration*, not a measurement — across seeds the same
rows range from 1/5 to 4/5 (ball_kick) and 0/5 to 3/5 (walk), and roller_crouch reads 60 % on three
seeds but 96 % on five. Use `logs/acceptance_gate.sh` (3 seeds x 5 rounds) for verdicts; this file is
for looking at the robot.

## Tooling the rotation depends on

* `logs/skill_demo.py` loads actors **by hand** (mlp weights + that checkpoint's own obs normalizer —
  the same arithmetic the ONNX export bakes). Upstream's `distill` runner fetches an expert through
  `wandb.Api()`, which had made every runner-based VelStand evaluation require an API key.
* It **drops `fell_over` when a row pins the robot on the floor** — the robot *is* fallen, so the term
  fires on step 1, the env recycles the episode, and the metric silently reports the env's own spawn
  mix instead of the pin. `family_eval.py` does the same for pinned batteries only.
* `--turn-ckpt` (or `MICRODUCK_TURN_CKPT`) points the two turn rows at a different checkpoint without
  disturbing the other thirteen.
* `--play-cfg` loads the no-DR play cfg, as a **diagnostic** for exactly the tail above: if failing
  rounds recover without DR, the residual is robustness, not policy shape. Never used for quoted
  numbers.
* Turn rows print `z=`, `g=` and `FELL`, because "stood still" and "fell over" are opposite failures.
* `logs/demo_finish.py` rebuilds the reel and the contact sheet from the per-skill clips; it uses the
  ffmpeg concat demuxer rather than loading 7 minutes of frames into RAM.

---

# Previous run (as deployed): 13 of 15, the turn rows failing

The same 15-row rotation with **all** rows on the deployed walking candidate `dc_long_0919_0125`.
Kept because it is the number that describes what is currently deployed, and because the two runs
disagree in exactly the places the caveat warns about (round-to-round spread on swizzle, roller_slope,
roulade — and, of course, the turn rows).

| skill | round 1 | round 2 |
|---|---|---|
| velocity walk | 0.252 m/s | 0.254 m/s |
| velocity turn | 0.318 rad/s (gain 0.64) | 0.338 (0.68) |
| walk+turn | 0.444 (0.89) | 0.467 (0.93) |
| ball_kick right | +0.61 m | +0.92 m |
| sitstand | 69 mm span | 66 mm span |
| ground_pick | 35 mm span | 47 mm span |
| rollers (fast) | 0.471 m/s | 0.448 m/s |
| swizzle | 0.413 m/s | 0.387 m/s |
| roller_slope | 247 mm descent | 241 mm |
| roller_standup | z 145 mm, held 300 | z 140, held 300 |
| roller_crouch | 73 mm span | 71 mm span |
| standup floor flip | hold 236, z 116 | hold 281, z 116 |
| velstand floor flip | hold 948, z 114 | hold 964, z 115 |
| spin | 1.40 rad/s at g −1.00 | 1.41 rad/s at g −1.00 |
| roulade | roll 6.44, ends 114 mm | roll 3.13, ends 115 mm |

## How the VelStand row got re-enabled (2026-09-25)

1. `family_eval.py` now loads actors by hand, so it no longer builds the `distill` runner for VelStand.
2. The raw qpos floor pin was never being measured: the env keeps `fell_over`, so the pin tripped it
   instantly (`fell_over 64/64`, trunk frozen at 75.0 mm) and the "0.996 floor-flip rate" was really
   the env's own spawn mix. With that termination dropped for the pin, the honest number is **0.812**
   (52/64, sustained). StandUp needed no such fix — its cfg removes `fell_over` by design — and it
   re-verifies at **0.984** (126/128), so quote those two numbers with their difference in mind.
