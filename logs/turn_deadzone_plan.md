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
