# Turn fix consolidation: verdict (2026-09-26)

Pre-registration: `logs/turn_ship_plan.md`. Raw sweep: `logs/turn_ship_results.txt`; per-round files in
`logs/turn_eval/`. Medians over 3 seeds x 5 rounds, in-place achieved |yaw|.

| arm | cmd 0.3 | cmd 0.5 | walk row |
|---|---|---|---|
| deployed policy (baseline) | 0.081 (gain 0.27) | 0.341 (0.68) | 0.252-0.258 m/s |
| pin + wobble OFF, 2,000 it | 0.256 (0.85) | 0.471 (0.94) | 0.251-0.256 |
| **pin + wobble OFF, 5,000 it** | **0.340 (1.13)** | **0.562 (1.12)** | **0.250-0.257 (3/3)** |
| wobble OFF, NO pin, 5,000 it | 0.278 (0.93) | 0.521 (1.04) | 0.260-0.266 (3/3) |
| wobble-free reference (older recipe) | 0.309 (1.03) | 0.554 (1.11) | 0.262-0.268 |

**Rule 1 fired** ("the untaxed arm tracks with the stock command mix, so the pin is not needed") and so
did **rule 3** (gain >= 0.9 at both commands with the walk row intact). Rule 1's threshold was set at
0.24 before the runs; read against the head-to-head rather than the threshold, the honest statement is:

* Removing the tax is what fixes the dead zone — the no-pin arm alone lifts 0.081 -> 0.278 (3.4x) and
  tracks 0.5 rad/s at gain 1.04.
* **The pin still buys something at the low command**: 0.340 vs 0.278 (gain 1.13 vs 0.93), and 5,000
  iterations of it reach the reference's ceiling (0.340 vs 0.309). It costs one cfg field and no
  runtime contract change, so the promoted policy keeps **both** levers.
* More iterations help: the pinned arm went 0.256 -> 0.340 at cmd 0.3 between 2k and 5k (the
  reference plateau is ~0.31-0.34), so 2k was not converged.

## Promoted policy

`logs/rsl_rl/velocity/2026-09-26_19-46-52_turn03_woboff/model_4998.pt` — deployed walking candidate +
`MICRODUCK_WARM_START=1` + `MICRODUCK_YAW_RANGE=0.3` + `MICRODUCK_SUSTAINED_TURN=0.3` +
`MICRODUCK_ANGULAR_WOBBLE=0`, 5,000 iterations. Both turn rows of the rotation now run on it
(`logs/skill_demo.py --turn-ckpt ...`) and pass 2/2 with video:

| row | round 1 | round 2 |
|---|---|---|
| velocity turn (cmd 0.5, in place) | 0.626 rad/s (gain 1.25) | 0.601 (1.20) |
| walk+turn (cmd 0.3 + yaw 0.5) | 0.581 (1.16) | 0.597 (1.19) |

Walking is unaffected (0.250-0.257 m/s at cmd 0.3, 3/3, trunk 113-121 mm).

## What is still wrong, stated plainly

The policy does not track on every episode. At seed 0, episodes 5 and 6 (of six) **stand still**
while commanded to turn: |yaw| 0.069 / 0.042 rad/s, trunk z = 115 mm, g = -1.00 at the end — upright,
never fell, simply did not rotate. Two controls pin the cause down:

* **DR off (`--play-cfg`)**: the same two episodes read 0.263 (gain 0.88) and 0.329 (1.10) — so most of
  the collapse is in the domain-randomization draw, not the policy shape.
* **An independently trained wobble-free policy** (`ref_wobbleoff`, 2026-09-17 recipe) fails the *same
  episode indices* with *nearly the same numbers* (0.072, 0.053). Two different policies failing
  identically at the same indices is a property of those episodes, not of either policy.

So the residual is a **DR-tail robustness gap**: in roughly one episode in three-to-five the policy
declines to turn under an unfavourable draw (friction / CoM / encoder bias / push), out of a
reward-economics preference for standing still. That is the next thing to attack, and it is a different
problem from the dead zone: the dead zone was systematic (median gain 0.27 across all episodes);
this is a tail (gain ~1.15 when it turns at all). Its first step is instrumentation of the per-episode
DR draw, not another reward term.
