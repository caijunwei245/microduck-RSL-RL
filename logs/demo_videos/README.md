# Simulation demonstrations (re-recorded 2026-09-29) — all skills, **5 rounds each**

Recorded offscreen from the training environments (`logs/skill_demo.py --rounds 5 --video`), 640x480,
25 fps (every 2nd control step of the 50 Hz policy), one file per skill with all five rounds back to
back. Each frame carries a burned-in label `<skill> (round N)`.

* **`all_skills_5rounds.mp4`** — all 15 clips concatenated in table order, **19 min 26 s, 41 MB**.
* **`round5_montage.png`** — contact sheet, one frame per skill from round 5.
* `all_skills_2rounds.mp4` / `round2_montage.png` were the earlier 2-round set; superseded and removed
  from the repository (the per-skill clips below are the 5-round ones).
* **73 of 75 rounds passed (97 %)**: thirteen skills 5/5, `ground_pick` and `standup floor flip` at
  4/5 — after the ball_kick scoring fix (a world-frame bug), the sitstand criterion fix (it demanded a
  cycle the command never asked for) and the two roller arms retrained under the measured latency
  envelope. Per-round numbers and the honest reading of each: `logs/skill_demo.md`; raw log
  `logs/skill_demo_5rounds_run.txt`.

| video | duration | round 1 / round 2 | what to watch |
|---|---|---|---|
| `velocity_walk.mp4` | 1:40 | **5/5** | 0.257-0.267 m/s at cmd 0.3, upright every round |
| `velocity_turn.mp4` | 1:40 | **5/5** | **0.584-0.613 rad/s for a commanded 0.5 (gain 1.17-1.23, every round)** — the fixed turn policy: wobble tax removed, low command pinned, measured latency |
| `walk_turn.mp4` | 1:40 | **5/5** | 0.556-0.600 rad/s (gain 1.11-1.20) while walking |
| `ball_kick_right.mp4` | 0:25 | **5/5** | ball +1.15 / +1.26 / +1.40 / +1.32 / +1.18 m along the robot's own forward axis — the actor never sees the ball (the old "clean miss" was a world-frame scoring bug) |
| `sitstand.mp4` | 1:00 | **5/5** | commanded postures: switch, held stand, then held sit x3 — each round judged against its own command (gate 24/25) |
| `ground_pick.mp4` | 1:40 | **4/5** | 38 / 39 / 39 / 44 mm spans, then one 31 mm span against a 40 mm criterion (same row, different round, as the 2026-09-28 set) |
| `rollers_(fast).mp4` | 1:40 | **5/5** | 0.518-0.549 m/s on the passive wheels |
| `swizzle.mp4` | 1:40 | **5/5** | 0.488-0.536 m/s |
| `roller_slope.mp4` | 1:40 | **5/5** | 179-186 mm descent this run (the range across spawns is wider), upright 1.00 every round |
| `roller_standup.mp4` | 0:30 | **5/5** | rises onto the wheels every round (z_max 141-145 mm, `upright_frac` 0.97-1.00) with the arm retrained under the measured envelope |
| `roller_crouch.mp4` | 1:20 | **5/5** | 71-75 mm crouch-and-return spans, retrained arm under the measured envelope |
| `standup_floor_flip.mp4` | 0:26 | **4/5** | from a pure prone pin: holds 221-286 of 300 steps at 116-117 mm in four rounds; round 3 never gets off its back (`hold=0`, trunk 65 mm) — the genuine 1-in-5 of a 14/15 gate row |
| `velstand_floor_flip.mp4` | 1:40 | **5/5** | from a pure prone pin: holds 542-977 of 1000 steps at 111-116 mm |
| `spin.mp4` | 1:40 | **5/5** | 1.43-1.48 rad/s at g -1.00, **upright** every round; 32.0-32.9 rad turned per 20 s episode (~5 turns) |
| `roulade.mp4` | 0:25 | **5/5** | roll magnitude spreads 1.66-6.99 rad but every round finishes standing at 115-118 mm |

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

    MICRODUCK_TURN_CKPT=logs/rsl_rl/velocity/2026-09-26_19-46-52_turn03_woboff/model_4998.pt \
      uv run python logs/skill_demo.py --rounds 5 --video   # per-skill clips (all rows)
    uv run python logs/demo_finish.py --rounds 5            # reel + contact sheet
