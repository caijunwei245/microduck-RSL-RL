# The turn "tail" is a coin flip in a chaotic, non-bit-reproducible simulator (2026-09-27)

Instrumentation: `logs/dr_tail_probe.py` (drives `skill_demo.run_round` through its `probe` hook),
`logs/sim_determinism_test.py` (no policy at all: reset → same fixed actions → compare). Evidence:
`logs/dr_tail_gains.out`, `logs/dr_state_spread.out`, and the traces quoted below.

**Two claims in earlier versions of this file were wrong, and they are corrected here**, because each
was made on a control too weak to support it: first "bistability with a 0.32 % payoff gap selects the
basin", then "state carried across in-place resets (a fresh env per round removes it)". What the
measurements actually support is below.

## The measurement that settles it

`logs/sim_determinism_test.py`, play cfg (no domain randomization), spawn pose pinned to exactly
(0, 0, 0.125) m — so every trial starts from the same root pose, the same joint positions
(`init|dq| vs t1 = 0.000e+00`) and gets the **same 200 actions**:

| trial | root_z | gravity_z |
|---|---|---|
| 1 | 0.11352 | −0.984 |
| 2 | 0.11126 | −0.936 |
| 3 | 0.12242 | −0.99998 |
| 4 | 0.11512 | −0.998 |

Divergence by |Δqvel| 0.79 rad/s. A per-step trace of two such trials, with the actuator delay fixed
to a single value, shows where it starts:

| field | first step that differs | max abs diff there | max abs diff over 40 steps |
|---|---|---|---|
| `joint_pos_target` (what the policy asked for) | — (identical) | 0 | 0 |
| `ctrl` (what the actuator wrote) | **1** | 1.1e-06 | 2.7e-02 |
| `qpos` | 1 | 4.0e-07 | 1.3e-01 |
| `qvel` | 1 | 2.1e-05 | 1.6e+00 |

The command path is exactly reproducible; the actuation/physics path is not. A **1e-6** difference on
the first step — floating-point, i.e. a non-associative reduction somewhere in the GPU kernels — is
amplified to O(1) within 40-200 steps. This is a 25 cm contact-rich biped: its Lyapunov time is a few
tenths of a second, so "identical inputs" does not mean "identical trajectory" in this simulator.

## Consequences, in order of how much they change practice

1. **A single episode of a marginal task is a coin flip, and 15 of them are not a verdict.** The
   in-place turn tracks at gain ~1.15 and stands still in a fraction of episodes — measured 2/6, 1/6,
   and (fresh env per round) 0/6 in different runs, which at n=6 is all the same number. The
   stand-still basin scores **240.401** against the turning policy's **241.184** (−0.32 %), so it does
   not take much noise to flip an episode. `acceptance_gate.sh` runs 3 seeds x 5 rounds; for a task
   this marginal that resolves roughly ±13 %, and the docs should quote episode counts next to rates.
2. **The delay dither adds a second, discrete difference on top.** With the stock
   `MICRODUCK_ACTUATOR_LAG="3,6"` the first-step `ctrl` difference is 5.4e-03 (a whole lag step of
   command, not rounding); forcing a single lag (`"2,2"`) cuts it to 2.6e-03; `"0,0"` (no delay) to
   1.1e-06. So fixing/holding the lag **reduces variance between runs** — useful for A/B work — but it
   does not make the environment reproducible, because the floating-point floor is still there.
3. **What is NOT the cause** (all measured, all with controls that could have failed): the policy
   (three training arms paying 2x/4x extra for the in-place turn left the tail identical, and the 4x
   arm was slightly worse); the reward shape; the spawn pose (pinning it changes nothing); every
   per-episode DR draw (pinning them all moves the failure to another round rather than removing it —
   and the big ones are `startup` events drawn once per env, measured constant); the firmware gains
   (`ENABLE_KP/KD_RANDOMIZATION = False`, `kp_scale` ≡ 1.0); and the delay buffer's *contents* (its
   `reset` does clear them).
4. **`skill_demo --fresh-env` is not an "artifact-free" mode.** It was added to test the carried-state
   hypothesis; the hypothesis did not survive (0/6 vs 2/6 at n=6 is noise). It is still a legitimate
   *different* question ("what does the policy do in a fresh episode"), but it must not be advertised
   as removing an artifact. The default (rounds back to back in one env) is the training-faithful one.

## What this retires

* The turn dead zone was a real, systematic, *fixable* defect (the `angular_wobble` tax; median 0.081 →
  0.340 rad/s at cmd 0.3, shipped and re-gated — `logs/turn_deadzone_verdict.md`,
  `logs/turn_ship_verdict.md`).
* The residual "tail" is not a defect at all: it is what a marginal task looks like in a chaotic
  simulator that is not bit-reproducible. No reward, cfg or training change can remove it; the honest
  response is more episodes per verdict, and distributional (never per-episode) comparisons.
* This is also the most likely explanation for a family of older observations in this repo
  ("walk 0/5 to 3/5 across seeds", "spawn sensitivity", single-checkpoint reversals): they are the same
  phenomenon, sampled too thinly.

## Reproduce

```bash
uv run python logs/sim_determinism_test.py --steps 200 --trials 4 --play-cfg --pin-spawn
uv run python logs/sim_determinism_test.py --steps 40 --trace --play-cfg --pin-spawn
MICRODUCK_ACTUATOR_LAG="2,2" uv run python logs/sim_determinism_test.py --steps 40 --trace \
    --play-cfg --pin-spawn
uv run python logs/dr_tail_probe.py --rounds 6 --pin all --dump-state --ckpts <turn ckpt>
```

---

## Validation of the latency fix (2026-09-28)

The dither half of the tail was testable: after changing the default envelope to the measured,
coherent one (`1,2` steps held 1000 steps — `logs/actuator_latency.md`), the first-step `ctrl`
difference dropped from **5.4e-03 to 1.5e-06 rad** (only the floating-point floor remains), and:

| measurement | historical default (3-6, dithered) | new default (1-2, held) |
|---|---|---|
| turn episodes below 0.15 rad/s, cmd 0.3, seed 0 | 2 of 6 (0.073 / 0.042) | **0 of 12** (all 0.278-0.347, median 0.336) |
| velocity walk (3 rounds) | 0.252 / 0.254 m/s | 0.267 / 0.261 / 0.265 m/s |
| velocity turn (3 rounds) | 0.626 / 0.601 | 0.611 / 0.617 / 0.605 |
| walk+turn (3 rounds) | 0.581 / 0.597 | 0.597 / 0.572 / 0.562 |

So the discrete difference the dither injected was what flipped those episodes, and removing it
neither regresses nor costs the deployed policies anything (walking is ~5 % faster under the envelope
the hardware actually has). Honest weight: the episode counts are one seed (0/12 against a historical
~20 % rate is p ≈ 0.07 on its own), but the *mechanism* is measured directly — the first-step `ctrl`
difference falls by three orders of magnitude — and the two together are what make this a fix rather
than a coincidence. The floating-point floor is untouched: no envelope makes the environment
bit-reproducible, so the "quote episode counts" rule above still stands.

### Five-seed gate verdict on the turn row (2026-09-28)

`logs/acceptance_gate.sh "velocity turn" <turn03_woboff@4998>` now selects **5 seeds** for this row
(marginal list) and prints why; under the new latency default the row scores **21/25 rounds (84 %)
PASS**, per seed 5/5, 3/5, 4/5, 4/5, 5/5. Two things to read in it:

* the failing rounds now read **0.229-0.468 rad/s (gain 0.46-0.94)** — the policy *turns*, just less
  than commanded. Under the old dithered envelope the same kinds of round read **0.042-0.073**, i.e.
  no rotation at all. The "stood still" mode is what the coherent envelope removed; what is left is
  ordinary tracking spread.
* the spread across seeds (5/5 to 3/5) is exactly why this row is on the 5-seed list: at 3 seeds the
  same policy could have been reported anywhere from 12/15 (80 %, PASS) to 15/15.
