# State of the project + where further optimization pays (2026-09-24)

Consolidates this session's arc. Raw evidence: the per-topic files under `logs/` (index below);
code inventory: `logs/CHANGES.md`; per-step verdicts: `logs/GOAL_SUMMARY.md`; assembled raw evidence:
`logs/GOAL_EVIDENCE.md`; chronological log with every wrong turn: `logs/optimization_plan.md`.

---

## 1. The 13-skill ledger (as measured, not as intended)

| skill | verdict | the number it rests on |
|---|---|---|
| velocity walk / turn / head | usable, one caveat | 0.22-0.23 m/s at cmd 0.3 (73 %), in-place turn 0.656 rad/s at cmd 0.5, head yaw/pitch 111 %/99 %, 140 s no falls |
| **spin** | **solved this session** | 0/5 -> **15/15 (100 %)**, upright at 114 mm / g -1.0, from either a capability-matched target (1.0 rad/s) or a roller-stand warm start |
| ball_kick (right) | usable but unreliable | 4 rounds send the ball 0.3-1.3 m, 5th misses entirely; **13/25 (52 %)** over 5 seeds |
| **ball_kick (left)** | **trained this session** | **9/15 (60 %)** - parity with the right foot |
| sitstand | usable | 17/25 (68 %); failures are a CONSTANT posture flag the robot ignores |
| ground_pick | solid | 25/25 (100 %) |
| **rollers** | usable, recipe now a choice | crouched 0.3500 m/s @ 116 mm; tall 0.2099 @ 138.9 mm; **staged 0.1911 @ 138.2 mm with a 15/15 demo** |
| swizzle | usable | 0.34-0.36 m/s (113-120 % of the commanded push) |
| roller_slope | solid | 25/25, descends at 137 mm |
| **roller_standup** | **fixed this session** | 19/25 -> **15/15 (100 %)** with the tilt-clause stall backstop |
| roller_crouch | solid | **24/25 (96 %)** - the earlier 60 % was 3-seed noise |
| **standup floor flip** | **solved this session** | 0/184 -> **184/184**; rehearsal 116.0 mm at 0.3 deg of tilt |
| **velstand floor flip** | **solved - corrected DOWN on 2026-09-25** | **0.812 (52/64, sustained)**, not 127/128: the pin trips `fell_over`, which recycled every episode on step 1, so the older number was measuring the env's own spawn mix (correction at the end of `GOAL_SUMMARY.md`; `skill_demo.py`/`family_eval.py` now drop that term for pinned rows only) |
| roulade | usable | **25/25**; rolls (progress 3.1-6.7) and ends upright at 115-116 mm |

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
* **The residual turn "tail" is an evaluation artifact, not a policy property** (2026-09-27): the
  policy stands still in ~1 episode in 5 when rounds run back to back in ONE env, and in **0 of 6**
  when each round gets a fresh env. Pinning the spawn pose, pinning every per-episode DR group, or
  paying 2x/4x extra for the turn (`MICRODUCK_TURN_FOCUS`, three arms) all leave it unchanged — it
  is state that survives an in-place reset. `skill_demo.py --fresh-env` measures without it; see
  `logs/turn_tail_findings.md`.
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
