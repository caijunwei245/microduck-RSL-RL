# Where optimization pays now (2026-09-28)

Supersedes §5 of `logs/STATE_OF_PROJECT.md` (2026-09-24), whose tier list predates four closures and
one new constraint. Every claim below carries the measurement it rests on; stale numbers are marked.

## 0. What changed since the last analysis

**Closed** (do not spend GPU hours here):

| was | now | evidence |
|---|---|---|
| turn dead zone, "no working fix" | the `angular_wobble` tax; wobble-off + pin takes cmd 0.3 from 0.081 to 0.340 rad/s (gain 1.13), gate 21/25 (84 %) PASS | `turn_deadzone_verdict.md`, `turn_ship_verdict.md`, `gate_velocity_turn.txt` |
| actuator-latency debt (declared 60-120 ms, dithered) | default is the measured 20-40 ms, held coherently; first-step `ctrl` spread 5.4e-03 -> 1.5e-06. **Strict win, no pass-rate cost**: the apparent roller cost (76 % / 88 %) was a train/test mismatch and two retrained arms are back at 100 % each; walk unchanged (88 vs 84 %), recovery unchanged (0.95-0.99) | `actuator_latency.md`, `regate_2026-09-28.md`, `roller_lag12_results.txt` |
| floor flips | standup 0.984, velstand **0.812** (the 0.996 was the spawn mix) | `GOAL_SUMMARY.md` correction |
| spin, roller_standup | 15/15 both | `spin_gate_finish.out`, `gate_roller_standup.txt` |
| "reward-side can fix the turn" | closed by measurement: 2x/4x extra pay changed nothing | `turn_focus_results.txt` |

**New constraint, and it re-ranks everything**: the simulator is chaotic and **not bit-reproducible
across resets** — identical state plus identical actions diverge from step 1 (`ctrl` differs by 1.1e-06
on step 1, reaching O(1) within 40-200 steps). A marginal task is therefore a **coin flip per episode**
(`turn_tail_findings.md`). The gate now takes 5 seeds on the marginal rows (`acceptance_gate.sh`).

## 1. Measurement resolution is the binding constraint

Round-pass rates at the gate's episode counts, 95 % CI half-width:

| true rate | n=15 (3 seeds) | n=25 (5 seeds) |
|---|---|---|
| 0.60 | ±24.8 pts | ±19.2 pts |
| 0.72 | ±22.4 | ±17.6 |
| 0.84 | ±18.6 | ±14.4 |

Episodes needed to *detect* an improvement (one-sample against a known baseline, 80 % power):

| improving | 5 pts | 10 pts | 20 pts |
|---|---|---|---|
| 0.60 -> | 753 rounds (151 seeds) | 188 (38 seeds) | 47 (9 seeds) |
| 0.72 -> | 632 | 158 (32 seeds) | 40 (8 seeds) |
| 0.84 -> | 421 | 105 (21 seeds) | 26 (5 seeds) |

Three consequences that should steer the next work:

1. **Any reward-level tweak worth <10 points on a marginal row is unmeasurable at this budget** — and
   the measured history of this repo is that such tweaks deliver 0-8 points (`crouch delta` 96->96,
   sitstand bucket 68->72, walk bucket 48->56, spin EMA/feet_flat 0->0). Stop booking them.
2. **Prefer binary or deterministic readouts.** "A skill that does not exist yet" (ball speed, trunk
   height, a rate in rad/s) is measurable in a handful of episodes; a change in *how often* a marginal
   skill succeeds needs hundreds.
3. **Paired evaluation is the cheap multiplier.** The between-seed spread dominates: the turn row's
   per-seed rounds are 5/3/4/4/5 (SD 0.84 rounds, ±0.73 on the mean at 5 seeds). Re-running the *same
   seeds* before/after cancels that term, so a ~1-round/seed shift becomes visible at 5 seeds instead
   of ~30. The gate and `skill_demo` already take `--seed`; what is missing is a paired runner that
   reports per-seed deltas with the seed as the unit.

## 2. Tier 1 — necessary, mechanism known, bounded cost

| item | why it is first | cost | expected effect vs noise |
|---|---|---|---|
| **Real-robot validation** (walk, floor flip, spin, sitstand) | **every** verdict in this project is simulation-only. The rehearsal, `export`, `publish` and the runtime path all exist; nothing has run on hardware | hours on hardware, no GPU | n/a — it is the only way to falsify the whole ledger |
| ~~**Publishability decision for phase-driven skills**~~ **RESOLVED 2026-09-28** | it was never a contract gap: `validate_manifest` already accepts `kind="scripted"` and `command.encoding in {constant, phase, posture_flag}`, and the official set (uploaded 2026-09-02) uses exactly those shapes. Only our builder refused to emit them | builder + CLI extended, four manifests built, validated and smoke-tested end-to-end with `--dry-run` (`logs/publishability.md`, `logs/publish_dryrun_e2e.txt`) | done — open only: daemon acceptance of community phase/posture_flag, and an HF token |
| **Ship the solved skills** | spin (ratecap/warm-stand 15/15), roller_standup (15/15), the staged roller recipe (138.2 mm @ 0.1911 m/s), the fixed turn policy | export + publish only | none — it is packaging, and it is what makes the work visible |
| ~~**Evidence hygiene**~~ **DONE 2026-09-28** | `logs/gate_*.txt` mixed eras: `gate_ground_pick.txt` read "FAIL 15/15 (100 %)" (pre-fix verdict parser) and `gate_spin.txt` read "FAIL 0/15" from a checkpoint the ledger calls 15/15 | **every row re-gated under the current recipe** (`logs/regate_summary.txt`, `logs/regate_2026-09-28.md`) | done |

## 3. Tier 2 — the real skill gaps, ranked by evidence quality

**As of 2026-09-28 this list is EMPTY.** Both entries that survived the first pass were criterion
artifacts: `ball_kick` scored the ball in the world frame instead of the robot's (52 % -> 100 %), and
`sitstand` demanded a cycle the command never asked for (68 % -> 96 %). What remains is one genuine
failure in 25 sitstand rounds (a commanded switch that never descended) and the rows already at
93-100 %. The next unit of work is therefore measurement policy and the non-GPU items below, not
another training arm.

| gap | the number | what is known | the cheapest decisive step |
|---|---|---|---|
| ~~**sitstand**~~ **ARTIFACT, 2026-09-28** | 2/5 in the rotation and 17/25 (68 %) on its gate — but the row demanded a full sit-and-return CYCLE while the task commands a BINARY posture (sit_flag, often held). Judged against each episode's own command it is **24/25 (96 %)**; 11 of the 25 rounds were held-command rounds and 10 of those obeyed | the policy was never the problem; one genuine failure in 25 | no arm needed — `logs/sitstand_criterion_fix.txt`, `logs/sitstand_fixed.txt` |
| ~~**ball_kick**~~ **ARTIFACT, 2026-09-28** | it was 13/25 (52 %) because the row scored the ball's **world-x** displacement while the task places the ball and pays the reward in the **robot's yaw frame**; corrected it is **25/25 (100 %)**, 1.15-1.40 m per round | the "spawn noise makes no difference" ablation (52/48/48 %) was a symptom of the same bug, not a finding | no arm needed — `logs/kick_frame_fix.txt`, `logs/kick_contact_probe.py` |
| **walk start transient** | 56 % of rounds start walking | failures are upright at 0.05-0.12 m/s; the sustained-forward bucket bought +8 points, so the rest is the stand-to-walk transition | one arm that spawns *partway into* the first step (reverse curriculum), which is the pattern that fixed the floor flips |
| **ground_pick** | 4/5 in the rotation (one 31 mm span vs a 40 mm criterion); historically 25/25 | most likely the same per-episode noise as everything else | re-gate at 5 seeds before spending anything |
| **in-place turn below 0.2 rad/s** | 0.008-0.032 rad/s achieved | the tax explains 0.3 but not 0.2; the runtime's held command is 0.5 | low necessity — leave it, or one diagnostic only if the runtime ever needs gentle heading correction |
| **roller posture/speed** | staged 0.1911 m/s @ 138.2 mm (15/15) vs 0.2099 @ 138.9 plain | it is a deployment choice, not a training gap | ship the staged recipe; stop training this |
| ~~**re-train the two roller families under the measured envelope**~~ **DONE 2026-09-28** | the loss was a train/test mismatch: 2 warm-start arms under the new envelope are back at **25/25 (100 %)** each (crouch z_max 138.9 mm, standup 143.0 mm, `upright_frac` 0.97-1.00 vs 0.61-1.00 for the old checkpoint) | confirmed mismatch, not a preference for the dithered stream | done (`logs/roller_lag12_results.txt`) |

## 4. Tier 3 — do not spend GPU hours

* **Reward-side work on any marginal task.** Five measured dead ends now (spin EMA, spin `feet_flat`,
  crouch `crouch_delta`, sitstand bucket, `turn_focus` 2x/4x). The pattern is consistent: task,
  initialisation and command-distribution changes move behaviour; pricing changes do not.
* **"Train the turn from scratch."** Answered: the from-scratch pinned arm reached 0.173 rad/s against
  0.340 for the warm+untaxed arm, and the mechanism is understood.
* **Chasing roller tall+fast.** The gaits are mutually exclusive by construction (at 138 mm the legs
  are straight; only the skating cycle generates thrust).
* **Anything whose expected effect is <10 points on a pass rate** — see §1.

## 5. The structural items still untried

1. **Measure the recovery baseline before training more recovery.** The nominal pose stands the robot
   up in 0.4 s; a hand-designed "recover" controller (nominal pose + balance policy) might cover the
   easy half of the floor-spawn distribution. The experiment is nearly free: run the floor-spawn
   battery with the **zero-action baseline** as the policy and see what fraction recovers. If that
   fraction is large, the deliverable changes shape.
2. ~~**A paired evaluation runner.**~~ **BUILT 2026-09-28** — `logs/paired_eval.py`, validated on a
   known difference (the turn policy at 2k vs 5k iterations, cmd 0.3, 3 seeds x 5 rounds):

   | readout | before 2k | after 5k | verdict |
   |---|---|---|---|
   | round pass rate | 0/15 | 0/15 | **sees nothing** (neither arm clears the 0.3 bar) |
   | paired `yaw_abs_mean` | 0.2205 | 0.2790 | **+0.051 +/- 0.007 (95 %)**, all three seeds same sign |

   Same difference, one readout blind and one resolving it with a ±0.007 interval — which is the
   practical form of §1: judge changes by the paired continuous delta, not the pass rate. It reuses
   `skill_demo.build_agent` / `run_round` and imports the row definitions from `skill_demo.SKILLS`, so
   a comparison cannot judge a different thing than the demos do. Usage:
   `uv run python logs/paired_eval.py --skill "velocity turn" --a old.pt --b new.pt --seeds 5`.
3. **Continuous readouts in the gate.** It already prints per-round physical values; report their
   medians with CIs alongside the pass counts, because the physical values are far less noisy than the
   pass/fail derived from them. Partly available now via `paired_eval.py`, which always prints the
   row's primary continuous metric; folding it into `acceptance_gate.sh` itself is still open.

## 6. Recommended order for the next GPU hours

| # | item | why first | cost | detectable? |
|---|---|---|---|---|
| 1 | Paired-evaluation runner + re-gate the 5 rows that have none (walk, standup flip, velstand flip, swizzle, roller_slope) | it makes every later verdict cheaper and stops stale numbers being quoted | hours of eval, no training | n/a (tooling) |
| 2 | ~~Ball-kick contact gate (1 arm)~~ **dropped** | the "gap" was a measurement frame bug — there is nothing to fix | 0 | — |
| 3 | Walk stand-to-walk reverse curriculum (1 arm) | 44 % of the walk failures live here and the pattern is proven elsewhere | 1 arm | 25-round gate + physical speed readout |
| 4 | Route-A test on sitstand and roller_crouch (no training) | decides whether 2 of the 4 unpublished skills can ship as-is | eval only | binary |
| 5 | ~~Sitstand descent-hold arm~~ **dropped** | its "gap" was a second criterion artifact (68 % -> 96 % once the row is judged against the command each episode carried) | 0 | — |

Not on the list, deliberately: any further reward-side tuning, any from-scratch turn run, and any
retraining of the families the latency change already improved (walk, rollers, swizzle, spin all moved
up under the measured envelope without a single new iteration).
