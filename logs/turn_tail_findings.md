# The turn tail is bistability, and it is a reward-economics problem (2026-09-26)

Follow-up to `logs/turn_ship_verdict.md`, which found a residual: the fixed turn policy tracks a
commanded 0.3 rad/s at gain ~1.15 in most episodes but in some **stands still** — upright, g −1.00,
trunk 115 mm, never falls — and an independently trained wobble-free policy fails the *same* episode
indices. Instrumentation: `logs/dr_tail_probe.py` (24/6-episode runs in `logs/dr_tail_6.out`,
`logs/dr_tail_pin.out`).

## Method note first: the probe must drive the demo's own loop

A first version re-implemented the rollout loop and reported episodes 5-6 at 0.232/0.233 rad/s — it
**hid the very tail it existed to explain**. Re-driving `skill_demo.run_round` through a new `probe`
hook reproduces the demo exactly (0.073 / 0.042, matching `skill_demo` to three decimals). Same code,
or no conclusion; the hook is now part of `skill_demo.py`.

## What the DR draw is NOT

| control | result |
|---|---|
| pin the spawn pose (all axes) | still stands still at episodes 5-6 (0.071 / 0.046) |
| pin the spawn yaw only | unchanged (0.080 / 0.055) |
| pin the spawn z only | unchanged (0.071 / 0.041) |
| **pin every per-episode DR group** (CoM, friction scale, armature, damping, gains, orientation, mass, and no pushes) | the failure **moves to episode 3** (0.047) instead of disappearing |

So it is not a bad draw value, and not the spawn state. Removing all episode-to-episode variation
just relocates the failure — the signature of a **bistable decision**, where which basin the episode
lands in is decided by differences far below the DR magnitude.

Also checked and constant, as the cfg's event table says they should be: foot friction (0.335), trunk
CoM (0.000), damping, `joint_pos_abs_mean`, `encoder_bias_abs_mean` (0.00726) — the big DR factors
(`foot_friction` 0.7-1.3, `encoder_bias` ±0.015 rad, `base_com` ±25-30 mm, mass/inertia) are all
`startup` events, drawn **once per env**, so they cannot explain episode-to-episode differences at all.

## What it IS: the two basins pay almost the same

Episode reward sums, stand-still episodes vs turning episodes (bare term names, same checkpoint, same
6 episodes):

| term | stand still | turning | delta |
|---|---|---|---|
| **total** | **240.401** | **241.184** | **−0.783 (−0.32 %)** |
| `track_linear_velocity` | 95.942 | 95.149 | **+0.793** |
| `track_angular_velocity` | 50.479 | 54.151 | −3.672 |
| `upright` | 37.645 | 36.904 | +0.741 |
| `angular_wobble` | −10.978 | −15.619 | **+4.641** |

Standing still gives up **less than one point out of 241** — and it *wins* on three of the four terms:
it collects the full linear-tracking reward (the commanded linear velocity is zero, so not moving is
correct), pays 4.6 less wobble, and is slightly more upright. Only the yaw term distinguishes the two,
and turning costs more wobble than the yaw term gains.

That is exactly why the wobble tax mattered so much (removing it moved the median 4x) and why a
residual tail survives it: **with the tax gone, turning is only barely better than standing**, so a
marginally harder episode tips back into the stand basin. It also reframes the "dead zone" finding:
the tax did not merely suppress the low-rate region, it made the stand basin the *argmax* there.

## The fix this implies (not yet run)

Make the yaw term dominate in the region where nothing else is at stake — the sustained-turn bucket
(`|cmd_vx|, |cmd_vy| ≈ 0`, `|cmd_yaw| ≥ 0.25`) that the cfg already identifies. Concretely: an
env-var-gated bonus (house pattern, default off) of the same Gaussian form as
`track_angular_velocity`, paid **only** on that bucket, with a weight large enough that the turn's
marginal payoff exceeds the wobble it costs — i.e. weight ≳ 2-4x the current yaw term. Not a penalty,
not a re-pricing: the diagnosis above says the marginal payoff of turning is *too small by
construction* in that region, which is a different defect from the re-pricings that failed on this
family before.

Two candidate arms, both from the promoted checkpoint with the tax already off:

* **focus arm** — the bonus above at weight 2.0 and 4.0, warm-started, 2,000 iterations.
* **mixture control** — `MICRODUCK_SUSTAINED_TURN=0.5` (half the envs on the turn command) with no
  reward change, to check whether the fix is merely "more turn data".

Readout: the same 3 seeds x 5 rounds at cmd 0.3 and 0.5 — the tail is what must move (count of
episodes below 0.15 rad/s), since the median is already at gain ~1.15. A fix that moves the median and
not the tail has not fixed this.
