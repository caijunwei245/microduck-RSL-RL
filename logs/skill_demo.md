# Skill rotation: all skills, 5 rounds each (2026-09-28)

## 5-round run — 69 of 75 rounds passed (92 %)

```bash
# the current artifacts: turn rows on the wobble-free turn policy, measured under the default
# latency envelope (1-2 steps held coherently, see logs/actuator_latency.md)
MICRODUCK_TURN_CKPT=logs/rsl_rl/velocity/2026-09-26_19-46-52_turn03_woboff/model_4998.pt \
  uv run python logs/skill_demo.py --rounds 5 --video
uv run python logs/demo_finish.py --rounds 5   # all_skills_5rounds.mp4 + round5_montage.png
```

| skill | round 1 | round 2 | round 3 | round 4 | round 5 | verdict |
|---|---|---|---|---|---|---|
| velocity walk | **0.267** | **0.261** | **0.265** | **0.257** | **0.262** m/s | **5/5** |
| velocity turn | **0.601** | **0.613** | **0.601** | **0.584** | **0.598** rad/s | **5/5** (gain 1.17-1.23) |
| walk+turn | **0.591** | **0.596** | **0.556** | **0.600** | **0.594** | **5/5** |
| ball_kick right | **+1.152** | **+1.263** | **+1.400** | **+1.317** | **+1.181** m | **5/5** (re-measured in the robot's yaw frame) |
| sitstand | **67** | 4 | 18 | **62** | 15 mm span | **2/5** |
| ground_pick | **38** | **39** | **39** | **44** | 31 mm span | 4/5 |
| rollers (fast) | **0.535** | **0.517** | **0.529** | **0.538** | **0.518** m/s | **5/5** |
| swizzle | **0.506** | **0.484** | **0.494** | **0.544** | **0.511** m/s | **5/5** |
| roller_slope | **405** | **171** | **414** | **409** | **192** mm | **5/5** (upright 0.97-1.00) |
| roller_standup | **held 175** | **276** | **300** | **208** | **275** | **5/5** (upright 0.61-1.00) |
| roller_crouch | **70** | **71** | **69** | 33 | **69** mm span | 4/5 |
| standup floor flip | **286** | **286** | **189** | **285** | **261** | **5/5** |
| velstand floor flip | **972** | **961** | **968** | **973** | **542** hold | **5/5** |
| spin | **1.46** | **1.43** | **1.47** | **1.44** | **1.46** rad/s | **5/5** |
| roulade | **6.94** | **7.04** | **3.35** | **4.90** | **1.66** roll | **5/5** (all end 115-116 mm) |

Videos: `logs/demo_videos/*.mp4` (one clip per skill, 5 rounds back to back), concatenated into
`all_skills_5rounds.mp4` (18 min 41 s), contact sheet `round5_montage.png` (round 5 of each skill).

What the five rounds add over two:

* **The turn rows are the story**: 5/5 and 5/5, at gain 1.17-1.23 on every single round, under the
  measured latency envelope. In the 2-round runs of 2026-09-26 the same rows failed strict tracking
  (0.318-0.338 rad/s, gain 0.64-0.68) before the wobble-tax fix, and the 2026-09-27 runs still lost
  the occasional round to the dithered-latency artifact. Ten consecutive successful turn rounds is the
  strongest evidence yet that both fixes did what they claimed.
* **sitstand is the weak row** at 2/5, with three rounds that never descended (span 4-18 mm). Its
  measured per-episode rate has been 68-72 % for weeks and this sample is below that, which is what a
  25-round gate exists for.
* **ball_kick is 5/5, not 4/5** — and the "clean miss" was an evaluation bug, not a policy defect:
  the row scored the ball's **world-x** displacement while the task places the ball and pays the
  reward in the **robot's yaw frame**, and the spawn yaw is randomised. The round that read
  `ball +0.000 m` actually sent the ball **+1.181 m** straight ahead. Re-measured, every round
  launches the ball **1.15-1.40 m** (gate: 25/25, median 1.30 m). `logs/kick_frame_fix.txt`.
* **ground_pick's 4/5** (one 31 mm span against a 40 mm criterion) remains the one mid-set tail.
* **roulade passes 5/5 while its roll magnitude spreads 1.66-7.04 rad** — the criterion is "ends
  upright", and it does every time; the roll count is not a quality score and should not be quoted as
  one.
* **roller_standup passes 5/5 but `upright_frac` ranges 0.61-1.00**, i.e. some rounds spend a third of
  the episode tilted. The pass/fail column hides that; the fraction is in the table for that reason.

| skill | round 1 | round 2 | checkpoint |
|---|---|---|---|
| velocity walk | **0.267 m/s** | **0.261 m/s** | `dc_long_0919_0125` |
| velocity turn | **0.601 rad/s (gain 1.20)** | **0.613 (1.23)** | `turn03_woboff` @4998 |
| walk+turn | **0.591 (1.18)** | **0.599 (1.20)** | same |
| ball_kick right | **+0.61 m** | **+0.92 m** | `kick_r4` |
| sitstand | **67 mm span** | **span 4 — never descended (FAIL)** | `sitstand_hold2` |
| ground_pick | **38 mm span** | **39 mm span** | `groundpick` |
| rollers (fast) | **0.533 m/s** | **0.518 m/s** | `rollers_noskate` |
| swizzle | **0.504 m/s** | **0.479 m/s** | `swizzle_rolling18` |
| roller_slope | **352 mm descent** | **347 mm** | `roller_slope` |
| roller_standup | **z 147 mm, held 159, upright 0.55** | **z 151, held 276, upright 0.92** | `roller_standup_stall` |
| roller_crouch | **70 mm span** | **71 mm span** | `crouch_delta2` |
| standup floor flip | **hold 286, z 116** | **hold 286, z 117** | `standup_tiltstall` |
| velstand floor flip | **hold 972, z 115** | **hold 965, z 111** | `velstand_fromstandup` |
| spin | **1.46 rad/s at g −1.00** | **1.43 rad/s** | `spin_warmstand` |
| roulade | **roll 6.94, ends 118 mm** | **roll 7.03, ends 116 mm** | `roulade_finish` |

**14 of 15 rows passed both rounds**; the exception is `sitstand` round 2, which never descended
(span 4 mm) — that row's per-episode rate has measured 68-72 % for weeks, so one episode in two is
inside its normal band, not a regression. The rows the latency change should help most moved the right
way in this sample: walk 0.252-0.254 → **0.261-0.267 m/s**, rollers 0.454-0.465 → **0.518-0.533**,
swizzle 0.402-0.419 → **0.479-0.504**, spin 1.40 → **1.43-1.46 rad/s**, roulade's roll 6.44/3.13 →
**6.94/7.03** (more consistent). `roller_standup` is the one row worth watching: it passes on both
rounds but its `upright_frac` reads 0.55 / 0.92, i.e. it spends part of round 1 tilted — quote the
fraction, not just the pass.

## Previous runs: 2 rounds each (2026-09-26 and 2026-09-28)

Kept because the two runs differ in exactly the places the latency change predicts, and because that
envelope is still reachable (`MICRODUCK_ACTUATOR_LAG="3,6" MICRODUCK_ACTUATOR_LAG_HOLD=0`).

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
roughly one episode in three-to-five the policy stands still.

**Resolved 2026-09-27 — it is the simulator, not the policy**: the environment is chaotic and not
bit-reproducible across resets (identical state + identical actions diverge from step 1, `ctrl`
differing by 1e-06 and reaching O(1) within 40-200 steps), and this task is marginal — the stand-still
basin is only 0.32 % behind on return — so a fraction of episodes flips. Pinning the spawn pose, or
every per-episode DR draw, or paying 2x/4x extra for the turn (three training arms) all leave it
unchanged. **Quote episode counts next to rates; single episodes of a marginal task are coin flips.**
Full chain: `logs/turn_tail_findings.md`.
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
