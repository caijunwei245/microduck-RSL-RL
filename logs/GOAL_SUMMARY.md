# Optimization plan — FINAL verdicts (2026-09-23)

Everything below is measured, not projected. Every row names the evidence file under `logs/`. Code
inventory and the full switch table: `logs/CHANGES.md`. Chronological detail, including the mistakes
made along the way: `logs/optimization_plan.md`.

---

## Step 1 — acceptance gate (D1) + crouch rehearsal (D2) — **DONE**

| item | evidence | verdict |
|---|---|---|
| D2 `--crouch` phase-driven rehearsal | `logs/reh_crouch_mode.log`: trunk **69.4 <-> 109.3 mm**, tilt 3-5.5 deg, 5 s period | **works** — the deployment path that had been missing |
| D1 gate (multi-seed, PASS/PARTIAL/FAIL) | `logs/gate_ground_pick.txt` 15/15 -> PASS; `logs/gate_ball_kick.txt` 9/15 -> PARTIAL | **works**, after fixing a bug that labelled every run FAIL |

## Step 2 — A1 spin — **SOLVED (0/5 -> 15/15)**

| arm | demonstration | terminal pose | training-side `spin_rate_track` | verdict |
|---|---|---|---|---|
| baseline `spin/model_999` | 0/15 (0 %) | 52.7 mm, g -0.345 | 0.60 | FAIL |
| `spin_ema` (EMA-filtered rate) | 0/15 (0 %) | 58.7 mm, g **-0.414** | **5.37 (9x)** | FAIL |
| `spin_feetflat` (`feet_flat` -2.0 -> -0.5) | 0/10 (0 %) | 44.8 mm, g -0.349 | 1.36 (2.3x) | FAIL |
| **`spin_ratecap`** (target 3.0 -> **1.0 rad/s**) | **14/15 (93 %)** | **112.0 mm, g -1.0** | — | **PASS** |
| **`spin_warmstand`** (actor from a roller-stand specialist) | **15/15 (100 %)** | **114.1 mm, g -1.0** | — | **PASS** |

**The mechanism was the target, not the pricing.** `SPIN_RATE_MAX = 3.0 rad/s` is ~3x what the robot
can hold, and with a std-1.5 Gaussian that makes "rotate fast" and "stay upright" non-complementary:
lying down is the more stable way to spin fast, so the policy did exactly that (0.6-1.2 rad/s at
~70 deg of tilt). Asking for 1.0 rad/s makes the upright constraint affordable, and starting from a
policy that already holds the roller stand solves it outright. The two reward-side arms are the
session's cleanest example of metric/behaviour divergence: 2-9x better tracking, zero behavioural
change. Evidence: `logs/spin_gate_finish.out`, `logs/a1b_results.txt`.

## Step 3 — A2 turn target pinned into the distribution — **NEGATIVE**

`MICRODUCK_YAW_RANGE=0.3 MICRODUCK_SUSTAINED_TURN=0.3`, +3,000 iterations from the deployment
candidate, then the same 5-round sweep:

| scenario | before | after |
|---|---|---|
| velocity turn (in place, cmd 0.3) | 0/5, achieved 0.051-0.100 | 0/5, achieved **0.035-0.066 (worse)** |
| walk+turn (cmd 0.3 + yaw 0.3) | 2/5, achieved 0.13-0.34 | 1/5, achieved 0.08-0.33 |

Pinning the operating point for 3k iterations on a 68k-iteration checkpoint did **not** open the
sub-0.25 rad/s dead zone; the in-place rate drifted slightly *down*. Either the region needs a run
trained with the pinned command from the start, or in-place turning at 0.3 rad/s is outside what this
stepping gait can hold. Evidence: `logs/step3_results.txt`, `logs/turn_sweep_*.out`.

### RESOLVED 2026-09-26 — it was neither: the `angular_wobble` tax

Both branches of that "either" turned out to be wrong. A pre-registered four-arm experiment plus a
matched-pair diagnostic (`logs/turn_deadzone_plan.md`, `logs/turn_deadzone_verdict.md`,
`logs/turn_scratch_results.txt`, `logs/wobble_deadzone.txt`) shows the dead zone is the
`angular_wobble` cost, which prices the instantaneous roll/pitch rate a scuffing pivot-step must
produce. In-place achieved |yaw|, median of 3 seeds x 5 rounds:

| arm | cmd 0.3 | cmd 0.5 |
|---|---|---|
| deployed policy | 0.081 (gain 0.27) | 0.341 (0.68) |
| from scratch, no pin (control) | 0.038 (0.13) | 0.285 (0.57) |
| from scratch, **pin 0.3** | 0.173 (0.58) | 0.410 (0.82) |
| deployed + warm start, pin 0.3, **wobble OFF** | **0.256 (0.85)** | **0.471 (0.94)** |
| wobble-free reference (older recipe, no pin) | 0.309 (**1.03**) | 0.554 (1.11) |

So: removing the tax is the dominant lever (a wobble-free recipe *tracks* 0.3 rad/s at gain 1.03); the
pin is a real but second-order lever that never reaches tracking on its own — which is exactly why the
step-3 fine-tune failed; and a fresh optimization with the tax in place does not find the exit either.
The residual floor at cmd 0.2 is **not** explained by the tax (both arms fail there) and stays open.
Consolidation into a shippable recipe: `logs/turn_ship_plan.md`.

## Step 4 — B two-stage roller curriculum — **works, but does not beat the simple swap**

| recipe | forward speed | trunk | demo |
|---|---|---|---|
| crouched baseline (noskate, no ramp) | 0.3500 m/s | 116.0 mm | 5/5 |
| band swap, skating terms kept | 0.2099 m/s | 138.9 mm | — |
| **both switched at step 0** | **0.0459 m/s** | 138.9 mm | — |
| two-stage, stage 2 at iter 800 | 0.1523 m/s | 139.5 mm | 12/15 |
| **two-stage, stage 2 at iter 2000** | **0.1911 m/s** | **138.2 mm** | **15/15** |

Staging the change avoids the catastrophic interaction (0.046 -> 0.191, a 4x recovery) and holds the
named posture with a perfect demo, but it lands *below* the plain band swap (0.2099) on speed: the
ramp buys robustness, not performance. Practical reading: if the skating terms are to be re-introduced,
do it gradually; if only the posture matters, the plain band swap is simpler and slightly faster.
Evidence: `logs/step4_results.txt`.

## Step 5 — per-row fixes

| item | measurement | outcome | evidence |
|---|---|---|---|
| A3 walk "doesn't start" | 12/25 (48 %) | `MICRODUCK_SUSTAINED_WALK=0.2` -> **14/25 (56 %)** — a modest gain; the failing rounds are upright at 0.05-0.12 m/s, and the bucket makes that operating point more common without eliminating it | `logs/a3_results.txt` |
| A4 ball_kick misses | 13/25 (52 %) | **noise ruled out**: 1.0x / 0.5x / 0.25x spawn noise -> 52 % / 48 % / 48 %. The misses are in the kick itself, so the cheap fix is off the table | `logs/a4_diagnostics.txt` |
| A4 sitstand | 17/25 (68 %); audit: 4 of 5 failures = a CONSTANT posture flag ignored | `MICRODUCK_SITSTAND_HOLD=0.3` -> **18/25 (72 %)**: +1 round, inside noise. The bucket makes the case common but does not fix it, so the remaining failures are not a distribution gap alone | `logs/sitstand_cmd_audit.txt`, `logs/sitstand_hold_results.txt` |
| A4 roller_standup | 19/25 (76 %); demo 4/5 with one round stalled at 80 mm for its whole episode | **tilt-clause stall backstop alone -> 15/15 (100 %)**; the extra near-wall spawn spread adds nothing measurable | `logs/step5b_results.txt`, `logs/step5b2_results.txt` |
| A4 roller_crouch | **24/25 (96 %)** over 5 seeds (the earlier 9/15 = 60 % was 3-seed noise) | `MICRODUCK_CROUCH_DELTA=4.0` -> **24/25 (96 %), identical to the baseline**: no gain, as expected once the small-sample artefact was removed. The one residual failure persists | `logs/seed_sweep2_results.txt`, `logs/crouch_delta_results.txt` |
| A5 left-foot kick | never trained | **trained**: `MICRODUCK_KICK_FOOT=left`, 1,500 iters -> **9/15 (60 %)**, at parity with the right foot (52 %), with 4 rounds sending the ball 0.3-1.3 m | `logs/step5a_results.txt` |

Measured over 25 rounds and needing nothing: `ground_pick` 25/25, `roulade` 25/25, `roller_slope`
25/25 - two of which had previously been mis-filed as failures on a probe artifact.

---

## What this objective changed in the skill ledger

| skill | before | after |
|---|---|---|
| spin | 0/5, "spins while down" | **15/15 (100 %), upright at 114 mm / g -1.0** |
| rollers | crouched 0.3500 m/s OR tall 0.2099 m/s (forced choice) | + a staged recipe holding **138.2 mm at 0.1911 m/s**, 15/15 |
| roller_standup | 19/25 (76 %) | **15/15 (100 %)** |
| left-foot kick | never trained | **60 %**, parity with the right foot |
| walk from step 0 | 48 % | 56 % |
| sitstand constant-command holds | 68 %, cause identified | 72 % (bucket added; +1 round, inside noise) |
| turn at 0.3 rad/s in place | dead zone | **still a dead zone** (negative result, documented) |

## Three lessons worth carrying forward

1. **Metric up, behaviour unchanged** is as real as the reverse documented in AGENTS.md. The two spin
   arms moved the training metric 2-9x with zero behavioural effect; only what changed the robot's
   *task* (an affordable rate) or its *initialisation* (a standing specialist) worked.
2. **A 5-round demonstration is not a verdict**: across seeds ball_kick ranges 1/5 to 4/5 and walk
   0/5 to 3/5, and roller_crouch reads 60 % on 3 seeds but 96 % on 5. The gate aggregates seeds now.
3. **The plan's own items split cleanly by kind.** Everything that changed the robot's *task* or
   *initialisation* worked (spin rate cap 0 -> 93 %, warm start 0 -> 100 %, roller_standup stall
   backstop 76 -> 100 %, walk bucket 48 -> 56 %); everything that merely added or re-priced a reward
   term did nothing measurable (EMA and `feet_flat` on spin: 0 -> 0 with a 2-9x better metric;
   crouch delta: identical 96 %; sitstand bucket: 68 -> 72 %).
4. **Fix the measurement before the policy**: this arc produced five measurement artifacts (a
   `--delay 4` rehearsal, z-keyed recovery buckets, a mis-named height term, a rehearsal with no tilt
   readout, a height-only stall rule), each of which had produced a confident wrong verdict.

## Still open

* The turn dead zone (step 3) has no working fix yet - the next attempt should train the pinned
  command from scratch rather than fine-tuning a 68k-iteration checkpoint.
* The in-place walk failures (A3, still 44 %) point at the start transient rather than the gait.
* Real-robot validation remains zero for every claim in this document.


## Correction (2026-09-25): the VelStand floor-flip number was inflated by a termination

Fixing `family_eval.py` (below) made its VelStand path runnable again, and the first thing it showed
was that the pin was never being measured: the VelStand env keeps `fell_over`, so pinning the robot on
the floor **trips it immediately** (`fell_over: 64/64`, `final z: 75.0 mm`, nothing moved), the env
recycles the episode, and the metric quietly captured the env's OWN spawn mix instead - which contains
partway-up prone starts, exactly the contamination this project has now hit four times.

| measurement | before | after the fix |
|---|---|---|
| VelStand floor flip (prone pin, 64 envs) | "**0.996**" | **0.812** (52/64 sustained), terminal 108.3 mm / g -0.999 |
| StandUp floor flip (prone pin, 128 envs) | 1.000 (n=184) | **0.984** (126/128 sustained), terminal 116.0 mm / g -1.0, `time_out` 126 (no recycling) |

So: **StandUp's 184/184 stands** (its cfg removes `fell_over` by design, and the fixed tool reproduces
~98-100 % from a genuine pin), while **VelStand's floor flip is 81 %, not 99 %**. The tooling fixes are
in `logs/family_eval.py`: a hand-loaded actor (no rsl_rl runner, which upstream's new `distill` runner
made unusable for this task), and `fell_over` dropped **when and only when** a floor pin is requested.


---

## Re-measurement (2026-09-28): the floor-flip numbers, at 256 envs

Every row was re-gated under the current recipe (`logs/regate_2026-09-28.md`), including the two
recovery families with their documented protocol (`SPAWN_FLOOR=prone`, 256 envs, current latency):

| task | recovery (floor, tilt >= 60 deg) | sustained |
|---|---|---|
| StandUp | 0.953 (n=256) | 0.953 |
| VelStand | 0.984 (n=256) | **0.965** |

Controls: both are unchanged under the pre-fix latency envelope (0.973 / 0.984 sustained) and under the
event-replayed pin flavour (`prone_event`: 0.957 / 0.961), so neither the envelope nor the pin
implementation explains the difference from the number recorded here on 2026-09-25 (**0.812**, 52/64
sustained, terminal 108.3 mm vs 112.4 mm today). Same checkpoint, same protocol, 4x the envs. At
p ~ 0.97 a 52/64 sample is a ~1e-4 event, so that measurement — not today's — is the outlier, and the
honest position is that **0.812 is unexplained history** while ~0.97 sustained at n=256 is the current
number. The direction of the 2026-09-25 correction is unaffected: the 0.996 it replaced really was the
env's own spawn mix.
