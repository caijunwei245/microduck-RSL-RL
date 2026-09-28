# Actuator-latency debt: measured, and the fix (2026-09-24)

## The two defects

1. **Range.** `_BAM_ACTUATOR_KWARGS` declares `delay_min_lag=3, delay_max_lag=6` = **60-120 ms** at
   50 Hz. The latency measured on the real robot is **20-40 ms = 1-2 control steps**
   (owner-ratified, `logs/kick_r1_report.md` §45/§801). Training was hardening against a lag ~3x
   larger than the hardware has.
2. **Coherence.** mjlab's `DelayBuffer` with the shipped defaults (`delay_update_period=0`,
   `delay_hold_prob=0.0`) re-draws each env's lag **every control step**, so "3-6 steps" is a dither
   around ~4.5, never a coherent delay - while the robot's latency is coherent (and so is the
   rehearsal's `--delay`). Only the kick task holds its lag (`KICK_LAG_HOLD_STEPS=250`), and it
   measured what the difference costs: same checkpoint, same env, **dithered 0.2433/0.2448 m/s** ball
   speed vs **held 0.1215/0.1404**, the held version reproducing the deployment rehearsal's
   0.109-0.19 (`logs/kick_r1_report.md` §34).

## Decisive measurement: the declared envelope is the collapse regime

Deployment candidate `logs/scr_comb_64250.onnx` (the walking policy), rehearsal at the deployment
command (0.3 m/s), 14 s per run, trailing 1 s averages:

| rehearsal `--delay` | forward speed | trunk z | tilt |
|---|---|---|---|
| 0 (no lag) | 0.23-0.25 m/s | 117.7 mm | 0.8-1.9 deg |
| **1 (20 ms - MEASURED real)** | **0.26-0.27 m/s** | 117-120 mm | 0.0-1.0 deg |
| **2 (40 ms - MEASURED real)** | **0.24-0.25 m/s** | 113-119 mm | 1.7-4.0 deg |
| 4 (80 ms - inside the DECLARED band) | **0.00 m/s - collapsed** | 36.4 mm | 27.9 deg |
| 6 (120 ms - the DECLARED max) | **0.00 m/s - collapsed** | 35.6 mm | 27.6 deg |

So the band the policy was trained to expect contains only regimes in which it cannot even stand,
while the latency the robot actually has is where it performs **best** (delay 1 beats delay 0). The
same pattern was measured earlier across the wheeled families (rollers/swizzle/roller_crouch/roulade
all "collapse" at delay 4 and are fine at 0-2), which is exactly why a `--delay 4` probe produced
three wrong verdicts in this project.

## The fix, behind two switches (defaults = the recipe that has been running)

```
MICRODUCK_ACTUATOR_LAG="min,max"   control-step lag range; measured truth is "1,2"
MICRODUCK_ACTUATOR_LAG_HOLD=<N>    hold each env's lag for N steps (0 = mjlab's dither; 250 = one episode)
```

`src/mjlab_microduck/robot/microduck_constants.py`; validated at build (`assert 0 <= min <= max`),
locked by three CPU tests. The arm to train is

```
MICRODUCK_ACTUATOR_LAG=1,2 MICRODUCK_ACTUATOR_LAG_HOLD=250
```

## Why this is not cosmetic

* Every sim2real claim in this repo was made under a latency envelope that is both wrong by 3x and
  outside the policy's tolerance, and **dithered** where the hardware is coherent. The kick task
  shows the coherence half alone is worth a factor of ~2 on a fast task.
* The corrected envelope is *easier* to satisfy, so the same budget should buy a better policy: the
  effort currently spent surviving 60-120 ms (a regime the robot never visits) can go into tracking.
* Until an arm is trained with the corrected, coherent envelope, any statement of the form "the
  policy is robust to deployment latency" is untested in the direction that matters.

## Arm results (2026-09-25)

| arm | recipe | score | vs baseline |
|---|---|---|---|
| `walk_lag12` | `LAG=1,2 HOLD=250`, +2,000 iters from the deployment candidate | **13/25 (52 %)** walk rounds | baseline 12/25 (48 %) - +1 round, inside noise |
| `spin_lag12` | same envelope, +1,500 iters from the spin rate-cap policy | **15/15 (100 %)** | phase-commanded baseline 14/15 - no regression, perfect score |

**Reading**: the corrected envelope is *safe* (nothing regressed) and *correct by measurement* (training
now matches the hardware: 20-40 ms, coherent), but for these two families it is not a performance jump
- walking is quasi-static, exactly as the kick docstring predicted ("walking policies are quasi-static
and were fine either way, which is why nothing caught it"). The performance half of this debt lives in
the FAST tasks, where the kick task already measured the coherence difference as a factor of ~2
(0.243 dithered vs 0.121-0.140 held). The remaining work is therefore not "retrain everything": it is
to use this envelope for new fast-task runs and for the kick family, and to keep `--delay 1..2` (never
4) as the rehearsal default.

---

## The defaults were changed to this envelope (2026-09-27)

Both halves of the debt are now the *default*, not an arm:

| | before | after |
|---|---|---|
| range | `delay_min_lag=3, delay_max_lag=6` (60–120 ms) | **`1,2`** (20–40 ms, the measured truth) |
| coherence | `delay_update_period=0` — re-drawn every control step | **`1000`** = 20 s = at least one whole episode of the longest task in the family |

Reasons, in the order they were measured:

1. **Fidelity.** The robot's 20-40 ms is coherent; the dither was a regime it never visits, and the
   kick task had already priced the coherence half at ~2x on a fast task (0.2433/0.2448 m/s dithered vs
   0.1215/0.1404 held, `logs/kick_r1_report.md` §34).
2. **Reproducibility.** The dither is a *discrete* per-episode difference on top of the
   floating-point floor: with everything else pinned, the first-step `ctrl` differs by **5.4e-03 rad**
   under the old default, 2.6e-03 with one fixed lag, 1.1e-06 with `"0,0"` — and **1.5e-06 with the
   new default** (`logs/turn_tail_findings.md`). In a task whose two behaviours are 0.32 % apart in
   return, that was enough to flip whole episodes.
3. **Safety.** Already measured before the change: `walk_lag12` 13/25 (baseline 12/25) and
   `spin_lag12` 15/15 (baseline 14/15) — nothing regressed, so this is not a retrain-everything event.
   New fast-task runs want it regardless, because that is where the coherence difference lives.

Reproduce the pre-2026-09-27 recipe in an A/B with `MICRODUCK_ACTUATOR_LAG="3,6"
MICRODUCK_ACTUATOR_LAG_HOLD=0`. Locked by `tests/test_sustained_walk_bucket.py`
(`test_actuator_lag_defaults_to_the_measured_latency`,
`test_historical_dither_recipe_is_still_reachable`) and `tests/test_ball_kick_cfg.py`
(`test_kick_actuator_lag_is_coherent_not_dithered`, whose old assertion forbade exactly this change).
