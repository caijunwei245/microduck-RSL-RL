# The turn tail is carried state across in-place episode resets (2026-09-27)

Follow-up to `logs/turn_ship_verdict.md`. Instrumentation: `logs/dr_tail_probe.py` (drives
`skill_demo.run_round` through its `probe` hook — see the method note), evidence in
`logs/dr_tail_gains.out`, `logs/dr_tail_6.out`, `logs/dr_tail_pin.out`, and the runs quoted below.

**Result in one line**: the policy is fine. Episodes run back to back in one environment inherit
state that no reset-mode event clears, and in this marginal task that is enough to flip whole
episodes. With a **fresh env per round the tail disappears: 0 of 6 episodes stand still**, against
**2 of 6** with in-place resets (same checkpoint, same command, same seed).

## The controls, in the order they were run

| # | control | result |
|---|---|---|
| 1 | pin the spawn pose (all axes / yaw only / z only) | still stands still at rounds 5-6 (0.071 / 0.046) |
| 2 | **pin every per-episode DR group** (CoM, friction scale, armature, damping, gains, orientation, mass, no pushes) | failure **moves to round 3** (0.047) instead of disappearing |
| 3 | re-init the BAM actuator's action-delay buffer at every episode start | unchanged (0.047 at round 3) — not the delay queue |
| 4 | **build a new env for every round** (DR still active) | **0/6 stand still** (0.283-0.352, median 0.317) |
| 5 | reward arms `MICRODUCK_TURN_FOCUS` = 2.0 / 4.0 / 0.0 (control), +2,000 iterations each | **identical tail in all three** (rounds 5-6: 0.071/0.050, 0.054/0.062, 0.067/0.054) |

Control 1-3 are the search for a *draw*: pinning everything that is drawn per episode does not remove
the failure, it relocates it. Control 4 is the decisive one: when the episode starts in a brand-new
environment — nothing to carry — every round turns.

Also checked and constant, as the cfg's event table says they must be: foot friction (0.335), trunk
CoM (0.000), damping, `joint_pos_abs_mean`, `encoder_bias_abs_mean` (0.00726), and `kp_scale` /
`kd_scale` = 1.000 (`ENABLE_KP_RANDOMIZATION = False`, `ENABLE_KD_RANDOMIZATION = False` in the
current recipe, so the firmware gains are not drawn at all). The big DR factors (`foot_friction`
0.7-1.3, `encoder_bias` ±0.015 rad, `base_com` ±25-30 mm, mass/inertia) are `startup` events, drawn
once per env — structurally unable to explain episode-to-episode differences.

## Why a residue can flip a whole episode: the task is marginal

Episode reward sums, stand-still rounds vs turning rounds (same checkpoint, same six rounds):

| term | stand still | turning | delta |
|---|---|---|---|
| **total** | **240.401** | **241.184** | **−0.783 (−0.32 %)** |
| `track_linear_velocity` | 95.942 | 95.149 | +0.793 |
| `track_angular_velocity` | 50.479 | 54.151 | −3.672 |
| `upright` | 37.645 | 36.904 | +0.741 |
| `angular_wobble` | −10.978 | −15.619 | **+4.641** |

Standing still gives up under one point in 241 and *wins* on three of the four terms — not moving
already collects the full linear-tracking reward because the commanded linear velocity is zero. That
is why the system is *sensitive* to a residue. It is **not** what selects the basin: arms 5 paid 2x
and 4x extra for turning in exactly that region and changed nothing (the 4.0 arm even pulled the
low-rate median down from 0.340 to ~0.25 rad/s while turning at a visibly lower stance, z 99-108 mm
against 112-118). The reward side of this problem is now exhausted, which is what pre-registered
decision rule 3 in `logs/turn_focus_plan.md` said would end it.

## What it costs the numbers already quoted

Same checkpoint, cmd 0.3 rad/s in place, seed 0, six rounds:

| | r1 | r2 | r3 | r4 | r5 | r6 |
|---|---|---|---|---|---|---|
| rounds back to back (default) | 0.357 | 0.360 | 0.379 | 0.382 | **0.073** | **0.042** |
| fresh env per round (`--fresh-env`) | 0.357 | 0.331 | 0.281 | 0.344 | 0.333 | 0.395 |

So the artifact is worth roughly one to two rounds in six, and it is the *whole* difference between
"two rounds where the robot did not rotate at all" and "six rounds that all rotate, one of them at
0.94 gain". `skill_demo.py --fresh-env` now exposes the artifact-free measurement; the default stays
in-place because that is what training does, and **every number in these ledgers should say which of
the two it came from**.

This is not specific to turning: `skill_demo`, `family_eval` and `acceptance_gate.sh` all run episodes
sequentially in one env, so any marginal task in the set inherits the same fraction. It plausibly
matters in training too (mjlab resets environments in place).

## What is left to find

The carried state is *not*: the spawn pose, any per-episode DR draw, the actuator's action-delay
buffer, the action history (`reset_action_history` exists), or the firmware gains. The remaining
candidates live in per-world simulator state that mjlab's `sim` wrapper does not expose (contact
cache / solver warm-start), so this is a reset-completeness question at the mjlab/MuJoCo-Warp layer,
not something a cfg or reward change can reach. Concretely worth doing next: identify which kernel
state survives `env.reset()` and either clear it or report it upstream — and until then, treat
in-place-reset multi-episode rates as a *lower bound* for marginal tasks.

## Method note (kept because it nearly cost the finding)

The first version of the probe re-implemented the rollout loop and reported rounds 5-6 at
0.232/0.233 rad/s — it **hid the very tail it existed to explain**. Driving `skill_demo.run_round`
through a new `probe` hook reproduces the demo to three decimals (0.073 / 0.042). The same lesson bit
twice more: the first `--pin all` did not pin the firmware gains (`kp_range` / `kd_range` were not in
its key list, and they are the one DR term that lives on the actuator rather than the model), and the
first attribution blamed the delay buffer without a control that could fail. Same code, and a control
that can fail, or no conclusion.
