# Turn dead zone: from-scratch pinned runs (pre-registration, 2026-09-26)

Written **before** the runs start, so the readout cannot be chosen after seeing the result — this
project has produced seven measurement artifacts by retro-fitting a criterion to a number.

## The question

The deployment command is an in-place yaw rate of **|yaw| = 0.5 rad/s held for minutes**
(`MICRODUCK_YAW_RANGE` default 0.5 is the top of the range; AGENTS.md). The deployed walking policy
tracks it at **50 %** (0.249 rad/s median) and falls off a cliff below it:

| commanded | achieved (median) | gain |
|---|---|---|
| 0.5 | 0.249 | 0.50 |
| 0.4 | 0.135 | 0.34 |
| 0.3 | 0.062 | 0.21 |
| 0.2 | 0.023 | 0.12 |

The achieved rate scales smoothly with the command, so this is **not** a footstep quantum with a hard
floor — it is a policy that under-responds at low rates. Step 3 of `optimization_plan.md` pinned 0.3
for 3,000 iterations on a 68k-iteration checkpoint and made it *worse* (0.035-0.066), but that
fine-tune inherited a checkpoint whose step-based curricula were already at their final stage, so it
tested "pin on top of a frozen gait", not "pin from the start of a run". `GOAL_SUMMARY.md` names the
missing experiment: **train the pinned command from scratch**.

**Hypothesis H1**: the dead zone is an artifact of the optimization history (a 68k-iteration policy
that settled into a step pattern with a minimum useful yaw increment); a run trained with the low
command present from iteration 0 opens it.

**Hypothesis H2**: the dead zone is a property of the stepping gait at this speed — the robot cannot
sustain a small in-place yaw rate without stalling the gait, and the pin cannot fix it.

H1 and H2 predict different things and the arms below separate them.

## Arms (all `Mjlab-Velocity-Flat-MicroDuck`, 4096 envs, defaults except where listed)

| arm | init | `MICRODUCK_YAW_RANGE` | `MICRODUCK_SUSTAINED_TURN` | iters | why |
|---|---|---|---|---|---|
| `turn05_warm` | deployment ckpt `dc_long_0919_0125/model_67997.pt` + `MICRODUCK_WARM_START=1` | 0.5 | 0.3 | 2,000 | the confound-fixed version of the negative result: same checkpoint, curricula restarted, deployment command pinned |
| `turn03_scratch` | from scratch | 0.3 | 0.3 | 5,000 | H1's test: the whole distribution is ≤ 0.3, with 30 % of envs held exactly at 0.3 |
| `turn00_scratch` | from scratch | 0.5 (default) | 0.0 (default) | 5,000 | **control** — separates "the pin helped" from "a fresh run differs" |
| `turn05_scratch` | from scratch | 0.5 | 0.5 | 5,000 | H1 at the deployment point: does a fresh run track 0.5 better than 50 %? |

`rel_sustained_turn_envs` pins a fraction of envs **in place** with |yaw| at the **top of the range**
and no resample for the rest of the episode, so `RANGE=0.3` + `SUSTAINED_TURN=0.3` is the only way to
pin exactly 0.3. Every arm uses the stock recipe otherwise (no reward re-weighting, no new terms —
those have now failed twice on this exact problem).

## Readout (fixed now)

`logs/skill_demo.py --rounds 5 --only turn --turn T --seed S` for `T ∈ {0.3, 0.5}`, `S ∈ {0, 1, 2}`,
on each arm's final checkpoint **and** on the baseline `dc_long_0919_0125/model_67997.pt` measured in
the same session. Reported per arm and command: median achieved |yaw| over the 30 rounds, mean gain
(achieved/commanded), and rounds clearing the script's 0.30 rad/s absolute bar. This is a 3-seed
sweep, not one checkpoint, per AGENTS.md.

## Decision rules (written before the data)

1. `turn03_scratch` median ≥ **0.20** at cmd 0.3 (gain ≥ 0.67) while the control stays ≤ 0.10
   → **H1 wins**: the dead zone is an optimization artifact, and pinning the command from step 0 is
   the fix. Blueprint goes into `optimization_plan.md` for the deployed recipe.
2. `turn03_scratch` ≤ 0.10 at cmd 0.3, i.e. indistinguishable from the control
   → **H2 survives**: this gait cannot hold a small in-place rate. Then the dead zone is a *design*
   constraint: the runtime must either command ≥ 0.4 rad/s or accept bang-bang heading control, and
   that goes in `AGENTS.md` + the runtime contract note instead of another training attempt.
3. `turn00_scratch` (control) itself reaches ≥ 0.20 at cmd 0.3 → a fresh run suffices and the pin is
   irrelevant; the deployed 68k policy is the odd one out, and the answer is "retrain", not "pin".
4. Any arm whose walk collapses (forward speed < 0.15 m/s at cmd 0.3 on the "velocity walk" row) is
   reported as **uninformative on turning** — a weak gait's yaw number says nothing about the dead
   zone. Checked with one `--only walk` run per arm before quoting its turn number.

## Cost and cost control

≈ 1 s/iteration at 4096 envs (measured: `walk_turn03` ran ~3,000 iterations in ~52 min), so 5,000
iterations ≈ 85 min. Four arms on two GPUs ≈ 3 h wall clock, ~5.7 GPU-hours. `turn05_scratch` is the
lowest-value arm and is dropped if the first three already settle the question.

---

## Supplement, added AFTER the arms launched (2026-09-26): the wobble-tax hypothesis

Not part of the pre-registration above — recorded as a post-hoc probe, and it is cheap enough to run
alongside the arms (`logs/wobble_deadzone.sh` → `logs/wobble_deadzone.txt`).

AGENTS.md already records that the `angular_wobble` cost — which prices the **instantaneous** roll and
pitch rate, i.e. exactly what a scuffing pivot-step produces — is what holds turning back at the
deployment command: **0.26-0.30 rad/s with it, 0.68-0.75 without it**. If the same tax is what makes
low commanded rates unattractive, then the dead zone is a *reward-shape* artifact after all, and the
right fix is a wobble-weight curriculum keyed on |commanded yaw| — not a from-scratch pinned run, and
certainly not the "the gait cannot do it" conclusion of decision rule 2.

A matched pair makes this measurable today, with no training: `2026-09-17_00-02-38_yawwobble_long` and
`2026-09-17_00-03-38_yawonly_long` (both `model_63998.pt`) started from the **same** source checkpoint
(`yawonly_lrbound@57999`), ran the same code for the same 6,000 iterations, and differ only in
`MICRODUCK_ANGULAR_WOBBLE`.

First three commands measured (3 rounds each, seed 0):

| commanded | wobble ON | wobble OFF |
|---|---|---|
| 0.5 | 0.370 / 0.359 / 0.394 (gain 0.72-0.79) | **0.568 / 0.554 / 0.563 (gain 1.11-1.14, 3/3 pass)** |

So at the deployment command the tax costs a third of the turn rate on this pair too. The commands
below 0.5 are the ones that decide whether the tax also *creates* the dead zone.

**How this interacts with the decision rules**: if the wobble-free arm tracks 0.3 at gain ≳ 0.7 while
the wobble-on arm sits at 0.36-0.50 (already measured: 0.108-0.150 rad/s), then rule 2's design
constraint is wrong and the correct fix is the weight curriculum. The four arms still answer the
separate question of whether a *fresh* optimization finds the dead zone's exit; both answers are worth
having, and they are not exclusive.

### Amendment (2026-09-26, ~16:10 — after the diagnostic, while two arms were still running)

The full diagnostic landed (`logs/wobble_deadzone.txt`) and it is unambiguous, so the arm set was
changed mid-flight. Recorded here because silently swapping arms after seeing data is how this project
produced seven measurement artifacts:

| commanded | wobble ON (matched pair) | wobble OFF (matched pair) |
|---|---|---|
| 0.5 | 0.359 / 0.370 / 0.394 (gain 0.72-0.79) | 0.554 / 0.563 / 0.568 (gain 1.11-1.14), 3/3 pass |
| 0.3 | 0.108 / 0.128 / 0.150 (gain 0.36-0.50) | **0.334 / 0.345 / 0.346 (gain 1.11-1.15), 3/3 pass** |
| 0.2 | 0.022 / 0.022 / 0.032 (gain 0.11-0.16) | 0.008 / 0.010 / 0.012 (gain 0.04-0.06) |

So the dead zone has **two** components: a large wobble-tax component that fully explains 0.3 rad/s,
and a residual floor at 0.2 rad/s that the tax does *not* explain (both arms fail there, and the
wobble-free arm is if anything worse). The dose-response at cmd 0.3 on the `yaw05` recipe agrees:
−3.0 → 0.079-0.122, −1.5 → 0.105-0.153 — halving the weight buys ~30 %, removing it buys 3x.

Changes:

* **Dropped `turn05_scratch`** (pre-registered as the droppable arm, and its question — the deployment
  point from scratch — is now the less interesting one).
* **Added `turn03_woboff`**: deployment checkpoint + `MICRODUCK_WARM_START=1` + pin 0.3 +
  `MICRODUCK_ANGULAR_WOBBLE=0`, 2,000 iterations. That is the *fix the diagnostic implies*, tested on
  the deployed policy rather than on a from-scratch gait.
* **Added `ref_wobbleoff` to the sweep** (the wobble-free reference policy) so the queue's own table
  contains the contrast measured under identical conditions. Flagged as a reference: it is an older,
  yaw-only recipe, so its absolute numbers are not comparable to the others.
* **New decision rule 0**, checked first: if `turn03_woboff` reaches ≥ 0.25 rad/s at cmd 0.3, ship that
  recipe and report the residual 0.2 floor from the reference instead of waiting on the scratch arms.
* Also fixed a scheduler bug: the completion test counted "arms not running" and broke at zero, which
  never fires when arms finish (the eval would never have run). It now breaks when every arm has
  finished, and a restarted scheduler recovers each running arm's GPU from `/proc/<pid>/environ` so it
  cannot double-book a GPU.


