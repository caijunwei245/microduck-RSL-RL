# Simulation demonstrations (2026-09-26) — all skills, 2 rounds each, 15/15 passing

Recorded offscreen from the training environments (`logs/skill_demo.py --rounds 2 --video`), 640x480,
25 fps (every 2nd control step of the 50 Hz policy), one file per skill with both rounds back to back.
Each frame carries a burned-in label `<skill> (round N)`.

* **`all_skills_2rounds.mp4`** — all 15 clips concatenated in table order, 7 min 32 s, 14 MB.
* **`round2_montage.png`** — contact sheet, one frame per skill from round 2.

| video | duration | round 1 / round 2 | what to watch |
|---|---|---|---|
| `velocity_walk.mp4` | 0:40 | **OK / OK** | 0.252 / 0.254 m/s at cmd 0.3, upright at 116 mm |
| `velocity_turn.mp4` | 0:40 | **OK / OK** | **0.626 / 0.601 rad/s for a commanded 0.5 (gain 1.25 / 1.20)** — the fixed turn policy: wobble tax removed, low command pinned |
| `walk_turn.mp4` | 0:40 | **OK / OK** | 0.581 / 0.597 rad/s (gain 1.16 / 1.19) while walking |
| `ball_kick_right.mp4` | 0:10 | **OK / OK** | ball sent +0.61 / +0.92 m; the actor never sees the ball |
| `sitstand.mp4` | 0:24 | **OK / OK** | 69 / 66 mm trunk span, commanded sit -> stand |
| `ground_pick.mp4` | 0:24 | **OK / OK** | 35 / 42 mm span; round 2 ends tilted (g -0.40) after touching the ground |
| `rollers_(fast).mp4` | 0:40 | **OK / OK** | 0.465 / 0.454 m/s on the passive wheels |
| `swizzle.mp4` | 0:40 | **OK / OK** | 0.419 / 0.402 m/s |
| `roller_slope.mp4` | 0:40 | **OK / OK** | 164 / 160 mm descent, upright 1.00 throughout (reel-to-reel range 160-247 mm) |
| `roller_standup.mp4` | 0:12 | **OK / OK** | rises onto the wheels and holds 145 / 140 mm for the whole hold window |
| `roller_crouch.mp4` | 0:40 | **OK / OK** | 73 / 71 mm crouch-and-return span |
| `standup_floor_flip.mp4` | 0:12 | **OK / OK** | from a pure prone pin: holds the stand 234 / 281 of 300 steps at 116 mm |
| `velstand_floor_flip.mp4` | 0:40 | **OK / OK** | from a pure prone pin: holds 952 / 964 steps at 114 / 115 mm (sustained 64-env rate 0.812) |
| `spin.mp4` | 0:40 | **OK / OK** | 1.40 rad/s at g -1.00, **upright** — the 0/5 "spins while down" result is fixed |
| `roulade.mp4` | 0:10 | **OK / OK** | rolls 6.44 / 3.13 rad and finishes standing at 114 / 115 mm |

The two turn clips run on a **different checkpoint** from the rest: `turn03_woboff@4998`, the
wobble-free, command-pinned policy from the turn dead-zone experiment (`logs/turn_deadzone_verdict.md`,
`logs/turn_ship_verdict.md`). Reproduce with
`MICRODUCK_TURN_CKPT=logs/rsl_rl/velocity/2026-09-26_19-46-52_turn03_woboff/model_4998.pt`.

Read the caveat before quoting any row as a rate: the rotation is a *demonstration*, the turn policy
still stands still in roughly one episode in three-to-five under an unfavourable DR draw, and across
seeds the other rows range widely (ball_kick 1/5-4/5, walk 0/5-3/5). Per-skill interpretation:
`logs/skill_demo.md`; verdicts: `logs/acceptance_gate.sh` (3 seeds x 5 rounds).

Reproduce:

    uv run python logs/skill_demo.py --rounds 2 --video   # per-skill clips
    uv run python logs/demo_finish.py --rounds 2          # reel + contact sheet
