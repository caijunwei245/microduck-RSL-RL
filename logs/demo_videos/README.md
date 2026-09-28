# Simulation demonstrations (2026-09-28) — all skills, 2 rounds each, under the measured latency

Recorded offscreen from the training environments (`logs/skill_demo.py --rounds 2 --video`), 640x480,
25 fps (every 2nd control step of the 50 Hz policy), one file per skill with both rounds back to back.
Each frame carries a burned-in label `<skill> (round N)`.

* **`all_skills_2rounds.mp4`** — all 15 clips concatenated in table order, 7 min 32 s, 14 MB.
* **`round2_montage.png`** — contact sheet, one frame per skill from round 2.

| video | duration | round 1 / round 2 | what to watch |
|---|---|---|---|
| `velocity_walk.mp4` | 0:40 | **OK / OK** | 0.267 / 0.261 m/s at cmd 0.3, upright at 116 mm |
| `velocity_turn.mp4` | 0:40 | **OK / OK** | **0.601 / 0.613 rad/s for a commanded 0.5 (gain 1.20 / 1.23)** — the fixed turn policy: wobble tax removed, low command pinned |
| `walk_turn.mp4` | 0:40 | **OK / OK** | 0.591 / 0.599 rad/s (gain 1.18 / 1.20) while walking |
| `ball_kick_right.mp4` | 0:10 | **OK / OK** | ball sent +0.61 / +0.92 m; the actor never sees the ball |
| `sitstand.mp4` | 0:24 | **OK / XX** | 67 mm span, then a round that never descended (span 4 mm) — this row measures 68-72 % per episode |
| `ground_pick.mp4` | 0:24 | **OK / OK** | 38 / 39 mm span, returns upright both rounds |
| `rollers_(fast).mp4` | 0:40 | **OK / OK** | 0.533 / 0.518 m/s on the passive wheels |
| `swizzle.mp4` | 0:40 | **OK / OK** | 0.504 / 0.479 m/s |
| `roller_slope.mp4` | 0:40 | **OK / OK** | 352 / 347 mm descent, upright 0.99-1.00 (this readout varies widely with spawn: 160-352 mm across reels) |
| `roller_standup.mp4` | 0:12 | **OK / OK** | rises onto the wheels, z_max 147 / 151 mm, but `upright_frac` 0.55 / 0.92 — round 1 is tilted for part of it |
| `roller_crouch.mp4` | 0:40 | **OK / OK** | 73 / 71 mm crouch-and-return span |
| `standup_floor_flip.mp4` | 0:12 | **OK / OK** | from a pure prone pin: holds the stand 286 / 286 of 300 steps at 116 / 117 mm |
| `velstand_floor_flip.mp4` | 0:40 | **OK / OK** | from a pure prone pin: holds 972 / 965 steps at 115 / 111 mm (sustained 64-env rate 0.812) |
| `spin.mp4` | 0:40 | **OK / OK** | 1.46 / 1.43 rad/s at g -1.00, **upright** — the 0/5 "spins while down" result is fixed |
| `roulade.mp4` | 0:10 | **OK / OK** | rolls 6.94 / 7.03 rad and finishes standing at 118 / 116 mm |

The two turn clips run on a **different checkpoint** from the rest: `turn03_woboff@4998`, the
wobble-free, command-pinned policy from the turn dead-zone experiment (`logs/turn_deadzone_verdict.md`,
`logs/turn_ship_verdict.md`). Reproduce with
`MICRODUCK_TURN_CKPT=logs/rsl_rl/velocity/2026-09-26_19-46-52_turn03_woboff/model_4998.pt`.

Read the caveat before quoting any row as a rate: the rotation is a *demonstration*. Across seeds the
rows range widely (ball_kick 1/5-4/5, walk 0/5-3/5), and the simulator is chaotic and **not
bit-reproducible across resets** — identical state plus identical actions diverge from step 1 — so a
marginal task like the in-place turn flips whole episodes on noise alone (`logs/turn_tail_findings.md`).
Quote episode counts next to rates. Per-skill interpretation: `logs/skill_demo.md`; verdicts:
`logs/acceptance_gate.sh` (3 seeds x 5 rounds).

Reproduce:

    uv run python logs/skill_demo.py --rounds 2 --video   # per-skill clips
    uv run python logs/demo_finish.py --rounds 2          # reel + contact sheet
