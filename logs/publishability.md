# Publishing the phase-driven skills (2026-09-24)

## The obstacle

`src/mjlab_microduck/publish/manifest.py` defines `Kind = Literal["episodic", "perpetual"]`:

* **episodic** - a button runs the policy for `duration_s`, then it returns on its own;
* **perpetual** - holds until told (and may declare `unwind_s` / a gait `slot`).

A **phase-driven** skill fits neither. Its cycle is carried in the 3-D twist slot as
`[cos(2*pi*phi), sin(2*pi*phi), 0]`, so it needs a *command that changes over time* - and the runtime
sends a constant one. That is why `ground_pick`, `spin`, `roller_crouch` and `sitstand` were recorded
as unpublishable, even though three of them are otherwise usable skills.

## The house precedent: roulade

`microduck_roulade_env_cfg.py` opens with "policy switch = roll starts immediately; no phase clock, no
reference motion" - a button-triggered trick whose schedule is **internal**. It is episodic and
publishable, and it needs no special command. That is the shape every phase-driven skill should take
to ship.

## Route A (implemented): internal clock, actor never sees the phase

`MICRODUCK_INTERNAL_PHASE=1` (`mdp.internal_phase_enabled` / `mdp.hide_phase_from_actor`, wired into
the four phase-driven cfgs):

* the **reward** keeps reading the real phase from the command manager - the task is unchanged, so
  every measured result stays comparable;
* the **critic** keeps it as privileged information;
* the **actor** sees a zeroed twist slot, i.e. exactly what the daemon sends when a button triggers an
  episodic policy.

Implementation detail worth knowing: mjlab hands the **same** `ObservationTermCfg` object to the actor
and the critic groups, so zeroing the actor's silently zeroed the critic's too (measured). The switch
therefore deep-copies the term for the critic and restores its scale.

**The trade-off is real and must be measured, not assumed**: with no phase input the cycle is
OPEN LOOP for the actor - it has to infer where it is from proprioception and elapsed time. A policy
retrained this way may be weaker than the phase-commanded one (or may be fine: the crouch cycle is
~5 s and the robot's own dynamics carry a lot of the timing). The arm `spin_internal` is queued:
`MICRODUCK_INTERNAL_PHASE=1` on the solved spin recipe, scored with the acceptance gate against the
phase-commanded policy's 14/15.

## Route B (not implemented): let the runtime drive the phase

A manifest field such as `"driver": {"kind": "phase", "period_s": 5.0, "slot": "twist"}` would let
`robotd` write `[cos, sin, 0]` itself, keeping the trained policy exactly as validated (no retrain, no
open loop). It is the *better* engineering answer and the *more expensive* one: the contract lives in
the other repo (`docs/policy-manifest.md` in `pollen-robotics/microduck`, schema 2), so it is a
cross-repo change plus daemon support. Note the runtime already has this capability in spirit - the
deployment rehearsal's `--ground-pick` / `--crouch` modes write the phase into the twist slot, and
`robotd` drives the `sitstand` posture flag the same way.

**Recommendation**: Route B if the daemon is going to grow a phase driver anyway (it already drives
one for sitstand), Route A if a skill must ship this week - and either way, measure the open-loop cost
first with the queued arm, because that number decides whether Route A is even acceptable.

## Arm result: Route A FAILS for spin (2026-09-25)

`spin_internal` = the solved spin recipe (`ratecap`, 14/15 phase-commanded) fine-tuned for +4,000
iterations with the actor's twist slot zeroed:

| policy | score | terminal pose |
|---|---|---|
| phase-commanded (`spin_ratecap`) | **14/15 (93 %)** | 112.0 mm, g -1.0 |
| **internal clock (`spin_internal`)** | **0/15 (0 %)** | 54.6 mm, g -0.406 |

So for spin the phase input is **load-bearing**: with no phase the actor cannot infer where it is in
the accel-hold-brake envelope, and a fine-tune does not relearn the cycle open-loop (it collapses to
the same lying-down attractor as the pre-fix policy).

Consequences, stated honestly:

1. **Route B (daemon-side phase driver) is the answer for spin** - and it is cheap in spirit, since
   `robotd` already drives the sitstand posture flag and the rehearsal already has `--ground-pick` /
   `--crouch` modes that write the phase into the twist slot. The contract change is the only real
   cost, and it must be made in the `microduck` repo.
2. **Route A is not ruled out for the slower phase tasks** (`roller_crouch`, 5 s cycle; `sitstand`,
   posture holds): timing there is far more forgiving than a spin envelope, and roulade proves an
   open-loop schedule can work. But it must be measured per family with this same gate - one
   measurement on spin is not a verdict for a 5 s crouch cycle.
3. **Neither route needs a new reward.** The failure is informational (the actor lost an input), not
   economic, which is why the same recipe scores 14/15 with the phase and 0/15 without.

---

## RESOLVED 2026-09-28: Route B was already in the contract — only our builder refused it

The conclusion above ("the contract lives in the other repo, so it is a cross-repo change") was wrong,
and re-reading the contract is what showed it. `validate_manifest` accepts `kind="scripted"` and
`command.encoding` in `{constant, phase, posture_flag}`, and `tests/test_publish_manifest.py` carries
the **official set as uploaded 2026-09-02**:

    alpha_sitstand.onnx     kind=scripted, encoding=posture_flag, ramp_s=2.0, unwind_s=1.0
    alpha_ground_pick.onnx  kind=episodic, encoding=phase,       period_s=4.0, end_phase=0.7

So the daemon-side half of Route B exists; the blocker was one sentence in OUR `build_manifest`
("phase and posture-flag encodings are the official set's own arms and are not something a community
policy can be") plus a CLI whose `kind` literal had two values. Both now emit and validate the
official shapes, so **the validated, phase-commanded checkpoints ship unchanged** — no internal clock,
no retraining, no reward work.

**Route A is now unnecessary, and the screening says it would have been a mistake anyway.** With the
actor's twist slot zeroed at eval time (an upper bound: the actor gets no chance to adapt), the
existing checkpoints score:

| task | phase visible (today's gate) | phase hidden (upper bound on Route A) |
|---|---|---|
| roller_crouch | 25/25 (100 %) | 15/15 (100 %) — but the cycle is shallower (span 53 vs 71-75 mm) |
| ground_pick | 14/15 (93 %) | **0/15 (0 %)** — span 10 mm, it barely moves |
| sitstand | 24/25 (96 %) | **6/15 (40 %)** — with the flag hidden it just holds the stand |

### End-to-end dry run: four skills, no upload

`logs/publish_stage.sh` (reproducible; it resolves every checkpoint through `skill_demo.newest`
instead of taking a path by hand) writing `logs/publish_stage.txt`. Each skill goes through the publish
path — ONNX export from the checkpoint, ONNX shape gate (61 -> 14), the smoke run (finite,
non-constant output), manifest build + `validate_manifest`, README render — and stops at `--dry-run`
with a staged `publish-<name>/` directory:

| skill | kind | encoding | timing (measured, not copied from the official set) |
|---|---|---|---|
| ground_pick | episodic | `phase` | period 4.0 s, duration 3.0 s, end_phase 0.75 — sweep in `logs/publish_duration_sweep.txt` (the official policy used 2.8 s / 0.7, same ballpark) |
| roller_crouch | episodic | `phase` | period 5.0 s, duration 3.0 s, end_phase 0.6 — passes at every duration >= 2.5 s |
| spin | episodic | `phase` | period 4.0 s, duration **2.6 s**, end_phase **0.65** — its own envelope's brake end: one full turn (6.6 rad measured), handed back at the commanded zero rate |
| sitstand | `scripted` | `posture_flag` | ramp_s 2.0 (= the cfg's POSTURE_RAMP_S), unwind_s 1.0, sit 1.0 / stand 0.0 |

The first hand-typed run of these commands is kept as history in `logs/publish_dryrun_e2e.txt`; two of
its four first attempts died on **wrong checkpoint paths** (a run directory that does not exist), which
is why the script resolves them programmatically now.

### Two follow-ups the dry run itself produced (2026-09-28)

**1. `ramp_s` was accepted and silently dropped.** The first version of the manifest fix *validated*
`ramp_s` and then never wrote it: the emitted sitstand manifest declared no glide while the official
`alpha_sitstand` declares `ramp_s: 2.0`. Found by diffing our output field-for-field against
`OFFICIAL_SET`, not by reading the code. Now emitted, refused when given outside a posture flag
(instead of ignored), validated when present, and the manifest tests compare our output against the
official entries key by key so the next dropped field fails a test instead of shipping.

**2. spin's duration was never measured — the sweep that "was uninformative" was measuring the rest
phase.** The 3-round sweep read 0/3 at 4.0 s against 3/3 at 2/3/6/8 s, which looked like per-episode
variance. Re-run at 15 rounds (`logs/publish_duration_spin15.txt`) it is perfectly deterministic, and
the readout that explains it is the rotation *integrated over the window*:

| window | row criterion | tail \|yaw rate\| | turned | net |
|---|---|---|---|---|
| 1.5 s | 15/15 | 3.05 rad/s | 3.8 rad | +3.8 |
| 2.0 s | 15/15 | 3.13 | 5.4 | +5.4 |
| 2.5 s | 15/15 | 2.62 | 6.5 | +6.5 |
| 3.0 s | 15/15 | 1.18 | 6.6 | +6.5 |
| 3.5 s | **0/15** | 0.26 | 6.5 | +6.5 |
| 4.0 s | **0/15** | 0.02 | 6.5 | +6.5 |

Every window >= 2.5 s delivers the *same* full turn; the 0/15 rows are windows that contain the
envelope's commanded REST segment (phase >= 0.650, i.e. after 2.6 s — `SPIN_ACCEL_END/HOLD_END/
BRAKE_END` in `mdp.py`), and the row criterion averages only the last third of the window. The policy
was obeying "stop", and the criterion called it a failure. Same class as the ball_kick frame and the
sitstand held-command bugs, third instance in one day: **before believing a rate, check the criterion
against the command the episode actually carried** — and for a button, quote the rotation the window
delivers, not the rate at the instant it is cut. 2.6 s (the brake end) is the published number; the
4.0 s full cycle that the first draft used would have shipped 1.4 s of commanded standstill.

**The one thing this repo cannot test**: whether the daemon accepts a COMMUNITY-published policy
declaring `phase` / `posture_flag`, or reserves those encodings for the official uploads. The format is
accepted here and used by the official set, and each generated `install_commands` line names the
driver the policy expects — so the first hardware run is the confirmation. That question, plus a
Hugging Face token (none is configured here), is all that stands between these four checkpoints and
`robotctl policy add`.
