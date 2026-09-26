# Turn tail: the in-place-turn bonus (pre-registration, 2026-09-26)

Follow-up to `logs/turn_tail_findings.md`. The tail is bistability with a **0.32 % payoff gap**:
over matched episodes the stand-still basin scored 240.401 against the turning policy's 241.184, and
it *wins* on three of the four terms (full `track_linear_velocity` because the commanded linear
velocity is zero, 4.6 less `angular_wobble`, marginally more upright). Only the yaw term separates
them, and the turn's wobble outweighs it. Pinning the spawn pose - or every per-episode DR draw -
moves the failure instead of removing it, which is what a near-tie looks like.

New switch: `MICRODUCK_TURN_FOCUS` (default 0.0 = off) pays `mdp.turn_focus_bonus`, the same Gaussian
as `track_angular_velocity`, gated to envs that are **in place** (`|cmd_x|, |cmd_y| <= 0.05`) with a
real turn commanded (`|cmd_yaw| >= 0.25`). Locked by `tests/test_turn_focus_cfg.py` (5 tests), smoke
tested at weight 4.0 (term computes, no NaN, `Episode_Reward/turn_focus` 0.000 -> 0.147 as the policy
turns).

## Arms

All three **resume `2026-09-26_19-46-52_turn03_woboff/model_4998.pt`** (the promoted turn policy:
wobble tax off, `MICRODUCK_YAW_RANGE=0.3`, `MICRODUCK_SUSTAINED_TURN=0.3`) for **+2,000 iterations**,
so the only difference between them is the new weight:

| arm | `MICRODUCK_TURN_FOCUS` |
|---|---|
| `turnfocus2` | 2.0 |
| `turnfocus4` | 4.0 |
| `turnfocus0` (control) | 0.0 (i.e. just 2,000 more iterations of the same recipe) |

## Readout (fixed now)

1. **The tail itself**: `logs/dr_tail_probe.py --rounds 6 --ckpts <arm>` at cmd 0.3, seed 0 - the
   direct reproduction of the demo's stand-still episodes (0.073 / 0.042 for the baseline). Count of
   episodes below 0.15 rad/s is the primary number; the median is secondary (the median is already at
   gain ~1.15, so **a fix that moves the median and not the tail has not fixed this**).
2. The 3-seed x 5-round sweep at cmd 0.3 and 0.5 (`skill_demo --only turn`), reported by
   `logs/turn_report.py`.
3. The walk row (3 rounds) - a turning policy that cannot walk is not a candidate.

## Decision rules (written before the data)

1. A focus arm shows **0/6 stand-still episodes** at seed 0 while the control still shows >= 1
   -> the payoff gap was binding. Adopt the SMALLEST weight that does it and re-gate the turn rows
   (`acceptance_gate.sh` + the rotation).
2. Control also shows 0/6 -> the two extra thousand iterations removed the tail by themselves (they
   widen the yaw term's margin by making turning better), and the bonus is unnecessary.
3. Every arm still shows >= 1 stand-still episode -> **the payoff gap is not the binding constraint**.
   Then the honest conclusion is "more reward for turning does not make this policy turn", the switch
   stays default-off, and the next lever is gait-level (step timing / contact pattern in the low-rate
   turn), not another reward term. This is the outcome that would retire the reward side of this
   problem for good.
4. Any arm whose walk row drops below 0.20 m/s at cmd 0.3 is disqualified regardless of its turn
   numbers.

## Cost

3 arms x 2,000 iterations at ~1.1 s/iteration, two GPUs -> ~80 min wall clock, then ~30 min of
readouts.
