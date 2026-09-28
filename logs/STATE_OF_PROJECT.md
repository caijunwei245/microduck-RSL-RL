# State of the project + where further optimization pays (2026-09-24)

Consolidates this session's arc. Raw evidence: the per-topic files under `logs/` (index below);
code inventory: `logs/CHANGES.md`; per-step verdicts: `logs/GOAL_SUMMARY.md`; assembled raw evidence:
`logs/GOAL_EVIDENCE.md`; chronological log with every wrong turn: `logs/optimization_plan.md`.

---

## 1. The skill ledger (refreshed 2026-09-28, gate + 5-round rotation)

Every row below was re-gated on 2026-09-28 under the current recipe (measured latency 1-2 steps held
coherently; turn rows on the wobble-tax fix) — 5 seeds = 25 rounds for the marginal rows, 3 seeds = 15
for the rest. Detail: `logs/regate_2026-09-28.md`, `logs/regate_summary.txt`, per row `logs/gate_*.txt`.
The 5-round rotation that goes with it: `logs/skill_demo.md`.

| skill | gate (2026-09-28) | 5-round rotation | the number it rests on |
|---|---|---|---|
| velocity walk | **PASS 22/25 (88 %)** | 5/5 | 0.257-0.267 m/s at cmd 0.3 — the largest move in the set (was 48 % on 2026-09-23) |
| velocity turn | **PASS 21/25 (84 %)** | 5/5 | 0.584-0.613 rad/s for a commanded 0.5 (gain 1.17-1.23) |
| walk+turn | PARTIAL 15/25 (60 %) | 5/5 (seed 0) | turning while walking; the five-seed number is the honest one |
| ball_kick (both feet) | **PASS 25/25 (100 %)** each (re-measured 2026-09-28) | **5/5** | right: median 1.301 m; left: median 1.362 m, both along the robot's own forward axis. The earlier right 52 % / left 60 % scored the ball's WORLD-x displacement while the task places the ball and pays the reward in the robot's yaw frame — an evaluation artifact, not a policy defect (`logs/kick_frame_fix.txt`) |
| sitstand | PARTIAL 17/25 (68 %) | 2/5 | 67 mm span when it works; the failures never descend |
| ground_pick | PASS 14/15 (93 %) | 4/5 | 38-44 mm span (mouth to ground and back) |
| rollers (fast) | PASS 15/15 (100 %) | 5/5 | 0.517-0.538 m/s on the passive wheels |
| swizzle | PASS 15/15 (100 %) | 5/5 | 0.484-0.544 m/s |
| roller_slope | PASS 15/15 (100 %) | 5/5 | descends the ramp upright every round |
| roller_standup | **PASS 22/25 (88 %)** — 25/25 under the pre-fix envelope | 5/5 | rises onto the wheels; the row is envelopE-dependent, see below |
| roller_crouch | **PARTIAL 19/25 (76 %)** — 25/25 under the pre-fix envelope | 4/5 | crouch-and-return span; same envelope dependence |
| standup floor flip | PASS 14/15 | 5/5 | prone pin at 256 envs: **0.953-0.973 sustained**, 116 mm, g -1.0 |
| velstand floor flip | PASS 15/15 | 5/5 | prone pin at 256 envs: **~0.97 sustained** (the recorded 0.812 is not reproducible today; the envelope is ruled out — see `regate_2026-09-28.md`) |
| spin | **PASS 15/15 (100 %)** | 5/5 | 1.43-1.47 rad/s, upright every round (the old gate file said FAIL 0/15 — stale) |
| roulade | PASS 14/15 (93 %) | 5/5 | rolls, ends upright at 115-116 mm |

**The one trade-off found by the refresh**: controlled A/Bs under the pre-fix envelope give walk
21/25 vs 22/25 (no effect) and recovery 0.95-0.99 either way, so the measured-latency default's ONLY
pass-rate effect is a cost on the two skating rows (roller_crouch 100 -> 76 %, roller_standup
100 -> 88 %) when their OLD checkpoints are evaluated under it; its benefits are fidelity and
reproducibility (first-step `ctrl` spread 5.4e-03 -> 1.5e-06), and the walk row's 48 -> 88 % move
belongs to the era (the sustained-forward bucket and the turn work), not to the envelope. That is a train/test mismatch until those families are re-trained under the new envelope; quote
roller rows with their envelope, and use the pre-fix one when the question is "did this policy
regress?".

Also demonstrated: a **continuous chain** - walk 0.27-0.32 m/s -> fall -> **get up in 0.86-1.02 s** ->
run 0.29-0.35 m/s, twice in one episode (`logs/demo_videos/chain_walk_fall_up_run.mp4`).

### Update (2026-09-26): the 15-row rotation, and the one row that still fails

The full skill set was re-demonstrated in simulation, 2 rounds each, with video: **13 of 15 rows passed
both rounds** (`logs/skill_demo.md`, reel `logs/demo_videos/all_skills_2rounds.mp4`, contact sheet
`round2_montage.png`, per-clip table in `logs/demo_videos/README.md`). Notes that change the ledger
above:

* **The VelStand floor-flip row is back in the rotation** and behaves like the corrected number says:
  from a pure prone pin it holds 948/964 steps at 114-115 mm - the sustained form of **0.812**, which is
  the figure to quote, never the 127/128.
* **The only failing rows are the two turn rows**: 0.318-0.467 rad/s for a commanded 0.5 rad/s. They
  clear the absolute 0.3 rad/s bar but under-track — and that dead zone was **solved the same day** by
  the pre-registered experiment it was handed to (`logs/turn_deadzone_verdict.md`): it is the
  `angular_wobble` tax, not the gait and not the optimization history. Removing it while pinning the
  low command takes the in-place rate from 0.081 to 0.256 rad/s (gain 0.27 -> 0.85) on the deployed
  policy, and a fully wobble-free recipe tracks 0.3 rad/s at gain **1.03**.
* **The residual turn "tail" is a coin flip in a chaotic, non-bit-reproducible simulator** (resolved
  2026-09-27, `logs/turn_tail_findings.md`): with domain randomization off and the spawn pinned, four
  trials of the *same 200 actions* from *identical* states diverge from step 1 — `joint_pos_target` is
  identical every step while `ctrl` differs by 1.1e-06 and reaches O(1) within 40-200 steps. In a task
  whose two basins are 0.32 % apart (240.401 vs 241.184) that is enough to flip whole episodes, so the
  in-place turn stands still in 2/6, 1/6 or 0/6 episodes depending on the run — the same number at
  n=6. Nothing about the policy or the reward is wrong here (three training arms paying 2x/4x for the
  turn left it identical); **quote episode counts next to rates and never verdicts from single
  episodes**. The actuator delay dither adds a discrete extra difference on top (first-step `ctrl`
  5.4e-03 by default, 1.1e-06 with `MICRODUCK_ACTUATOR_LAG="0,0"`), so pinning the lag lowers variance
  for A/B work without making the environment reproducible.
* The evaluation machinery itself is now **in the public repo** (`logs/skill_demo.py`,
  `logs/family_eval.py`, `logs/acceptance_gate.sh`, `logs/demo_finish.py`), so every number in these
  ledgers has a reproducible path instead of living in a gitignored directory.

## 2. What actually moved the needle (and what never did)

Everything that changed the robot's **task or initialisation** worked:

| change | result |
|---|---|
| spin target 3.0 -> 1.0 rad/s (a rate it can hold) | 0/5 -> 14/15 |
| spin actor warm start from a roller-stand specialist | 0/5 -> **15/15** |
| roller_standup: tilt-clause stall backstop | 76 % -> **100 %** |
| standup floor flip: tilt clause on `recovery_stall` | 0/184 -> **184/184** |
| velstand floor flip: cross-family actor transplant | 1/128 -> **127/128** |
| walk: sustained-forward-command bucket | 48 % -> 56 % |

Everything that merely **added or re-priced a reward term** did nothing measurable:

| change | result |
|---|---|
| spin: EMA-filtered yaw-rate pricing | 0/15 -> 0/15, **training metric 0.60 -> 5.37 (9x)** |
| spin: `feet_flat` -2.0 -> -0.5 | 0/10 -> 0/10, metric 1.36 (2.3x) |
| standup: stop annealing the last-mile price | 0/184 -> 0/184 |
| roller_crouch: potential-based return-leg delta | 96 % -> 96 % |
| sitstand: sustained-posture bucket | 68 % -> 72 % (one round, in noise) |
| roller: band swap + skating terms at step 0 | 0.2099 -> **0.0459 m/s** (a catastrophic interaction the staged version avoids) |

## 3. The measurement saga (six artifacts, each of which produced a confident wrong verdict)

1. **`--delay 4` rehearsal** - made rollers/swizzle look immobile (0.000 m/s) and roller_crouch
   collapsed; the policies trained on mjlab's per-step lag *dither*. Three verdicts reversed.
2. **Recovery buckets keyed on trunk z** - silently mixed the sitting keyframe (upright, 50-90 mm)
   with true floor poses; the "14 %/23 % prone/supine recovery" was entirely sitting starts
   (recovered envs had a median spawn tilt of 7.1 deg, non-recovered 90.0 deg).
3. **`com_height_target` measures the trunk, not the CoM** - and its band topped out 14.5 mm *below*
   the measured 138 mm roller stand while charging for anything above it: the reward taxed the very
   posture the family is named for.
4. **A rehearsal that printed no tilt** - could not tell a 33.4 deg park from a 0.3 deg stand at the
   same height (113.7 vs 116.0 mm), so "the fix changed nothing" was wrong.
5. **A height-only stall rule** - could not see a robot parked at 113 mm, which is why a 3,000-iteration
   retrain changed the floor-flip rate by exactly nothing.
6. **A 5-round demonstration treated as a verdict** - across seeds ball_kick ranges 1/5 to 4/5, walk
   0/5 to 3/5, and roller_crouch reads 60 % on 3 seeds but 96 % on 5.

Plus two tooling gaps: no `--crouch` rehearsal mode (now added) and a gate that labelled every run
FAIL (now fixed). The gate now aggregates seeds and prints PASS/PARTIAL/FAIL at 80 %/50 %.

## 4. Two facts that reframe "recovery"

* **Zero action stands the robot up**: with the default joint targets (action = 0) the servos lift it
  from prone to standing in **~20 steps (0.4 s)**. The pre-fix policy's 0/184 was therefore a
  *policy-selection* failure, not a physical limit - and any recovery demo must hand over to the
  recovery policy immediately or the zero-action pose fakes the get-up.
* **The walk policy cannot be knocked over**: raw qvel writes did nothing, a single engine push
  (2.6 m/s lateral, 11 rad/s pitch, 1.3 m/s lift) was shrugged off, and a *sustained* shove threw it to
  223 mm of trunk height and ~70 deg of tilt and it still recovered, every time, for 200 steps. Its
  push DR is only +-0.2 m/s, so the trained robustness is ~10x outside its envelope.

---

## 5. Where further optimization pays - space x necessity

Necessity is judged by "what breaks if we do not do it"; space by "is there a mechanism we know works".

### Tier 1 - necessary, and the mechanism is known

| item | space | necessity | cost |
|---|---|---|---|
| **Commit the work** (24 files uncommitted, including every fix above) | trivial | **critical** - the entire session is unprotected; one stray checkout loses it | minutes |
| **Real-robot validation of the two solved skills** (standup floor flip, spin) | large: the ONNX/BAM rehearsal path, `publish`, and the runtime all exist | **critical** - every verdict in this document is simulation-only; the floor flip in particular is the kind of behavior whose sim2real gap (contact, friction, servo saturation) matters most | hours on hardware |
| **Ship the spin recipe** (`MICRODUCK_SPIN_RATE_MAX=1.0`, and/or the warm start) | done - the arms exist, 14/15 and 15/15 | high - it is a solved skill sitting in a checkpoint | one export + publish |
| **Actuator-latency debt**: only the kick task trains with a coherent per-episode lag; `_BAM_ACTUATOR_KWARGS` declares 60-120 ms against a measured 20-40 ms | large - the kick task already implements the fix (`KICK_LAG_HOLD_STEPS`) | **high** - it silently weakens every sim2real claim in this repo | cross-repo decision + retrain budget |
| **Publishability of phase-driven skills** (ground_pick, spin, roller_crouch) | medium - needs a constant-command wrapper or a runtime phase driver | high if they are to ship; the schema-2 contract currently excludes them | bounded code |

### Tier 2 - worth doing, mechanism partly known

| item | space | necessity | cost |
|---|---|---|---|
| ~~**Turn dead zone below ~0.25 rad/s**~~ **SOLVED 2026-09-26** | it was neither the gait nor the optimization history: the `angular_wobble` **tax** on the scuffing a slow pivot-step needs. Wobble-off + pin takes cmd 0.3 from 0.081 to **0.256** (gain 0.85) on the deployed policy, and a fully wobble-free recipe tracks it at **1.03**; the pin alone (both fine-tune and from-scratch) plateaus at 0.14-0.17. Consolidation in flight (`turn_ship_plan.md`); the 0.2 rad/s floor is a separate, still-open question | - | done (`turn_deadzone_verdict.md`) |
| **ball_kick reliability (52 %)** | noise is ruled out, so the fix is a contact-gated shaping term or a wider contact area - both untested | medium - a kick that misses half the time is a demo, not a capability | 1-2 arms + a 25-round hit-rate protocol |
| **walk start transient (56 %)** | the failures are upright-but-not-moving; the sustained-forward bucket helped by 8 points, so the rest is the stand-to-walk transition itself | medium - it is the difference between "walks" and "walks when it feels like it" | 1-2 arms |
| **Sitstage the roller recipe as the shipped one** | measured: staged holds 138.2 mm at 0.1911 m/s with a 15/15 demo, vs 0.2099 for the plain swap and 0.0459 for the naive combination | medium - it is a deployment choice between posture and speed, and the staged recipe is the robust option | export + publish |

### Tier 3 - low value, do not spend GPU time

* More reward-side work on spin: two arms moved the metric 2-9x with zero behavioural effect; the
  solved levers were the task and the initialisation.
* roller_crouch pricing (96 % -> 96 %) and the sitstand bucket (+1 round): measured dead ends.
* Chasing the roller tall+fast combination: the two gaits are mutually exclusive by construction
  (at 138 mm the legs are straight; only the skating cycle generates thrust, and that is what the
  fast crouched gait removes).

### The one structural idea still untried

**Recovery may not need a learned flip at all for many falls.** Since the nominal pose stands the
robot up in 0.4 s, a hand-designed "recover" controller (command the nominal pose + a balance policy)
could cover the easy half of the 184/184 distribution, leaving the learned policy for the cases where
that fails. That is worth measuring before spending more training: run the floor-spawn evaluation with
the zero-action baseline as the "policy" and see what fraction it recovers. If it is large, the
recovery deliverable changes shape - which is exactly the kind of finding that has repeatedly
overturned verdicts in this project.

## 6. Evidence index (all under `logs/`)

`GOAL_SUMMARY.md` (per-step verdicts) - `CHANGES.md` (switches + tools) - `GOAL_EVIDENCE.md`
(assembled raw evidence) - `optimization_plan.md` (chronological, incl. failures) -
`session_report_2026-09-22.md` (the earlier marathon) - `chain_demo.md` + `demo_videos/` (15 per-skill
clips + the 2-round reel; `demo_videos/README.md` is the table) - `skill_demo.md` (the rotation, and
which rows pass) - `turn_deadzone_plan.md` (the pre-registered turn experiment) -
`turn_sweep_*.out`, `spin_gate_finish.out`, `a1b_results.txt`, `step3/4/5a/5b/5b2_results.txt`,
`a3_results.txt`, `a4_diagnostics.txt`, `crouch_delta_results.txt`, `sitstand_hold_results.txt`,
`sitstand_cmd_audit.txt`, `seed_sweep_results.txt`, `seed_sweep2_results.txt`, `gate_*.txt`,
`family_*.md`, `task_verification.md`.
