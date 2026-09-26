# Simulation demonstrations (2026-09-26) — all skills, 2 rounds each

Recorded offscreen from the training environments (`logs/skill_demo.py --rounds 2 --video`), 640x480,
25 fps (every 2nd control step of the 50 Hz policy), one file per skill with both rounds back to back.
Each frame carries a burned-in label `<skill> (round N)`.

* **`all_skills_2rounds.mp4`** — all 15 clips concatenated in table order, 7 min 32 s, 14 MB.
* **`round2_montage.png`** — contact sheet, one frame per skill from round 2.

| video | duration | round 1 / round 2 | what to watch |
|---|---|---|---|
| `velocity_walk.mp4` | 0:40 | **OK / OK** | 0.252 / 0.254 m/s at cmd 0.3, upright at 116 mm both rounds |
| `velocity_turn.mp4` | 0:40 | XX / XX | 0.318 / 0.338 rad/s for a commanded 0.5 — clears the 0.3 rad/s bar, under-tracks (the documented dead zone) |
| `walk_turn.mp4` | 0:40 | XX / XX | 0.444 / 0.467 rad/s (gain 0.89 / 0.93): turning while walking tracks far better than turning in place |
| `ball_kick_right.mp4` | 0:10 | **OK / OK** | ball sent +0.61 / +0.92 m; the actor never sees the ball |
| `sitstand.mp4` | 0:24 | **OK / OK** | 69 / 66 mm trunk span, commanded sit -> stand, stays upright |
| `ground_pick.mp4` | 0:24 | **OK / OK** | 35 / 47 mm span; round 2 ends tilted (g -0.35) after touching the ground |
| `rollers_(fast).mp4` | 0:40 | **OK / OK** | 0.471 / 0.448 m/s on the passive wheels |
| `swizzle.mp4` | 0:40 | **OK / OK** | 0.413 / 0.387 m/s |
| `roller_slope.mp4` | 0:40 | **OK / OK** | 247 / 241 mm descent, upright 1.00 throughout |
| `roller_standup.mp4` | 0:12 | **OK / OK** | rises onto the wheels and holds 145 / 140 mm for the whole hold window |
| `roller_crouch.mp4` | 0:40 | **OK / OK** | 73 / 71 mm crouch-and-return span |
| `standup_floor_flip.mp4` | 0:12 | **OK / OK** | from a pure prone pin: holds the stand 236 / 281 of 300 steps at 116 mm |
| `velstand_floor_flip.mp4` | 0:40 | **OK / OK** | from a pure prone pin: holds 948 / 964 steps at 114 / 115 mm (re-enabled row; sustained 64-env rate 0.812) |
| `spin.mp4` | 0:40 | **OK / OK** | 1.40 / 1.41 rad/s **upright** (g -1.00) — the 0/5 "spins while down" result is fixed |
| `roulade.mp4` | 0:10 | **OK / OK** | rolls 6.44 / 3.13 rad and finishes standing at 114 / 115 mm |

13 of 15 rows passed both rounds; the two turn rows are the honest exception and have a
pre-registered experiment (`logs/turn_deadzone_plan.md`). Per-skill interpretation, the pass
criteria and the reasoning behind them: `logs/skill_demo.md`; use `logs/acceptance_gate.sh`
(3 seeds x 5 rounds) before quoting any row as a rate — two rounds is a demonstration, not a
measurement.

Reproduce:

    uv run python logs/skill_demo.py --rounds 2 --video   # per-skill clips
    uv run python logs/demo_finish.py --rounds 2          # reel + contact sheet
