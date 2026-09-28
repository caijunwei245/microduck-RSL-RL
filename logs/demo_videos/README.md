# Simulation demonstrations (2026-09-28) — all skills, **5 rounds each**

Recorded offscreen from the training environments (`logs/skill_demo.py --rounds 5 --video`), 640x480,
25 fps (every 2nd control step of the 50 Hz policy), one file per skill with all five rounds back to
back. Each frame carries a burned-in label `<skill> (round N)`.

* **`all_skills_5rounds.mp4`** — all 15 clips concatenated in table order, **18 min 41 s, 39 MB**.
* **`round5_montage.png`** — contact sheet, one frame per skill from round 5.
* `all_skills_2rounds.mp4` / `round2_montage.png` were the earlier 2-round set; superseded and removed
  from the repository (the per-skill clips below are the 5-round ones).
* **69 of 75 rounds passed (92 %)**: 11 skills 5/5, three at 4/5 (`ball_kick`, `ground_pick`,
  `roller_crouch`) and `sitstand` at 2/5. Per-round numbers and the honest reading of each:
  `logs/skill_demo.md`.

| video | duration | round 1 / round 2 | what to watch |
|---|---|---|---|
| `velocity_walk.mp4` | 1:40 | **5/5** | 0.257-0.267 m/s at cmd 0.3, upright every round |
| `velocity_turn.mp4` | 1:40 | **5/5** | **0.584-0.613 rad/s for a commanded 0.5 (gain 1.17-1.23, every round)** — the fixed turn policy: wobble tax removed, low command pinned, measured latency |
| `walk_turn.mp4` | 1:40 | **5/5** | 0.556-0.600 rad/s (gain 1.11-1.20) while walking |
| `ball_kick_right.mp4` | 0:25 | **5/5** | ball +1.15 / +1.26 / +1.40 / +1.32 / +1.18 m along the robot's own forward axis — the actor never sees the ball (the old "clean miss" was a world-frame scoring bug) |
| `sitstand.mp4` | 1:00 | **2/5** | 67 / 62 mm spans; three rounds never descended (span 4-18 mm) — the weak row, 68-72 % per episode |
| `ground_pick.mp4` | 1:40 | **4/5** | 38 / 39 / 39 / 44 mm spans, then one 31 mm span against a 40 mm criterion |
| `rollers_(fast).mp4` | 1:40 | **5/5** | 0.517-0.538 m/s on the passive wheels |
| `swizzle.mp4` | 1:40 | **5/5** | 0.484-0.544 m/s |
| `roller_slope.mp4` | 1:10 | **5/5** | 171-414 mm descent (varies with spawn), upright 0.97-1.00 every round |
| `roller_standup.mp4` | 0:30 | **5/5** | rises onto the wheels every round (z_max 141-147 mm), but `upright_frac` spreads **0.61-1.00** — the pass column hides that |
| `roller_crouch.mp4` | 1:20 | **4/5** | 69-71 mm spans, then one 33 mm span ending tilted (g -0.38) |
| `standup_floor_flip.mp4` | 0:30 | **5/5** | from a pure prone pin: holds the stand 189-286 of 300 steps at 116-117 mm |
| `velstand_floor_flip.mp4` | 1:40 | **5/5** | from a pure prone pin: holds 542-973 of 1000 steps at 112-116 mm |
| `spin.mp4` | 1:40 | **5/5** | 1.43-1.47 rad/s at g -1.00, **upright** every round |
| `roulade.mp4` | 0:25 | **5/5** | roll magnitude spreads 1.66-7.04 rad but every round finishes standing at 115-116 mm |

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
