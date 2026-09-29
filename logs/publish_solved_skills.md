# Exporting and publishing the solved skills (2026-09-28)

Five policies go out through the same `uv run publish` path the runtime loads from: **spin**,
**roller_standup**, **ball_kick right**, **ball_kick left**, and **walk_turn** (the turn-fixed walking
gait). Reproduce with

```bash
bash logs/publish_solved.sh                    # dry run: export + shape gate + smoke + manifest, staged locally
HF_USER=<name> bash logs/publish_solved.sh --upload   # creates/updates the Hub repos
```

Log of the dry run: `logs/publish_solved.txt`. Staged output: `publish-<name>/{policy.onnx,
manifest.json, README.md}`, ~794 KB each, 61 -> 14 shape gate passed and a finite, non-constant smoke
run on every one.

| skill (repo `microduck-<name>`) | checkpoint | kind / encoding | slot | timing | gate behind it |
|---|---|---|---|---|---|
| `spin` | `spin/2026-09-23_20-26-02_spin_warmstand/model_3999.pt` | episodic / `phase` | - | period 4.0 s, duration **2.6 s**, end_phase **0.65** | 15/15, upright, 1.43-1.47 rad/s |
| `roller_standup` | `roller_standup/2026-09-28_14-42-12_rollerstandup_lag12/model_1999.pt` | episodic / `constant` | - | duration **1.0 s** | 25/25, `z_max` median 143 mm, upright 0.97-1.00 |
| `ball_kick_right` | `ball_kick_right/2026-09-20_14-14-35_kick_r4/model_999.pt` | episodic / `constant` | `kick_right` | duration **1.5 s** | 25/25, median launch 1.301 m |
| `ball_kick_left` | `ball_kick_left/2026-09-23_16-47-24_ball_kick_left/model_1499.pt` | episodic / `constant` | `kick_left` | duration **1.5 s** | 25/25, median launch 1.362 m |
| `walk_turn` | `velocity/2026-09-26_19-46-52_turn03_woboff/model_4998.pt` | perpetual / `constant` | **`walk`** | none (a gait) | 21/25 turn, median 0.5541 rad/s at a commanded 0.5 |

Every `duration_s` below is measured in this repo, not copied from the official set. Three of the five
numbers were **not** what the first read said, and each time the fix was the same: the criterion was
being read against something the episode had not been asked to do.

## `roller_standup`: the mixed-spawn sweep cannot see the rise

`logs/publish_duration_sweep.py` says every window from **0.6 s** up passes 15/15 with the trunk ending
at 139 mm — which is impossible as a rise time, and it is not one. RollerStandUp spawns **50 % belly /
50 % already standing** (`set_ground_state`, and the standing bucket is deliberate: it is what teaches
the policy to HOLD, not just to rise), so half the rounds pass a short window for free.

`logs/roller_standup_rise.py` pins one bucket at a time and measures the rise directly
(`logs/roller_standup_rise.txt`, 15 rounds per bucket, 3 s window):

| spawn bucket | reaches 130 mm | rise (median) | z at the end | hold inside the window |
|---|---|---|---|---|
| belly (face down) | 15/15 | **0.20 s** (0.18-0.20) | 139 mm | 140/150 steps |
| back (face up, the hard direction) | 15/15 | **0.32 s** (0.30-0.32) | 139 mm | 134/150 steps |
| already standing | - | - | 139 mm | 150/150 steps, upright 1.00 |

So the published window is **1.0 s**: it covers the worst bucket (0.32 s) three times over, ends with
every bucket standing at 139 mm, and the standing bucket — the one that tests holding — never falls
(0/15; the `fell` flag on belly/back rounds is the pose-based test firing at step 0 on a spawn that is
already low, which is why the row's own criterion has no fall gate).

**Footgun for the next measurement like this**: pinning the *event* is a silent no-op —
`ground_state_mix` is an `event_param_curriculum` that runs before the reset events and rewrites those
four probabilities on every reset. Pinning the event gave a still-50/50 mix; the working pin rewrites
the curriculum's own stage. Same shape as AGENTS.md's "writes to `env.cfg` are ignored".

## `ball_kick`: the window does not change the kick, only how far the ball has got

`logs/publish_duration_ball_kick.txt` (right) and `logs/publish_duration_ball_kick_left.txt` (left,
`MICRODUCK_KICK_FOOT=left`), 15 rounds per window:

| window | right, ball travelled | left, ball travelled | robot at the end |
|---|---|---|---|
| 0.5 s | 0.10 m (min 0.08) | 0.11 m (min 0.09) | standing, 115-116 mm |
| 1.0 s | 0.23 m (min 0.20) | 0.23 m (min 0.20) | standing, 115-116 mm |
| **1.5 s** | **0.36 m (min 0.30)** | **0.38 m (min 0.29)** | standing, 115-116 mm |
| 2.0 s | 0.49 m (min 0.38) | 0.49 m (min 0.41) | standing, 116 mm |

The strike happens at step 5-8 (0.10-0.16 s, `logs/kick_frame_fix.txt`), so *every* window from 0.5 s
up contains the kick; the trunk moves 5-6 mm across the window in all of them. The ball simply keeps
rolling, which is why the distance has no plateau to find. 1.5 s is the choice: the ball is
demonstrably clear (0.29-0.44 m) when the daemon switches back, and the robot has been standing still
for over a second. `duration_s` here is a handoff decision, not a physics measurement.

## `walk_turn`: does shipping the turn fix cost the walk row? No

The turn policy is a *velocity* policy (wobble tax off + sustained-turn sampling), so publishing it
means the `walk` slot, which currently holds a different checkpoint. Paired evaluation on the walk row,
same seeds, both arms (`logs/paired_walk_vs_turn.txt`, 5 seeds x 5 rounds):

```
rounds/seed delta: mean +0.00 +/-0.00 (95%), sign test p=1.000   ->  5 seeds tied
'v_mean'   delta: mean +0.00884 +/-0.003298 (95%)               ->  in its favour, every seed
UNPAIRED contrast: 22/25 = 88% +/-13  vs  22/25 = 88% +/-13      ->  blind to an effect this size
```

So the trade is: the turn row goes from "stands still in a third of episodes" to 21/25 with a median
0.5541 rad/s at a commanded 0.5, and the walk row does not move (marginally better on `v_mean`). The
manifest declares the twist honestly (`--twist-help`): the constant encoding renders `command.twist` as
"unused (zeros)", which is right for the zero-command skills but wrong for a gait, where the daemon
drives a real `[vx, vy, wz]` and `wz` is the whole point of this build.

## What the manifests say

All five pass `validate_manifest`; the install lines are generated from the manifest, not written by
hand (`logs/publish_solved.txt`):

```bash
# spin (phase: the daemon writes [cos(2*pi*phi), sin(2*pi*phi), 0], period 4.0 s, end_phase 0.65)
sudo robotctl policy add spin <user>/microduck-spin
robotctl robot do spin

# roller_standup and the two kicks: episodic one-shots, constant zero command
sudo robotctl policy add roller_standup <user>/microduck-roller_standup
sudo robotctl policy add ball_kick_right <user>/microduck-ball_kick_right
sudo robotctl policy add ball_kick_left  <user>/microduck-ball_kick_left

# walk_turn: a gait, loaded into the slot rather than run as a skill
sudo robotctl policy load walk <user>/microduck-walk_turn
```

## Status of the upload

**Nothing is uploaded yet: this environment has no Hugging Face credential**
(`uv run hf auth whoami` -> `Not logged in`; no `HF_TOKEN`, no `~/.cache/huggingface/token`). The
network reaches `huggingface.co`, and `uv run publish --upload` needs only the token — it creates the
repo, pushes `policy.onnx`, `manifest.json` and `README.md`, and the ONNX has already passed the gate
the daemon applies at load. To finish, one of:

```bash
uv run hf auth login          # interactive, writes ~/.cache/huggingface/token
export HF_TOKEN=hf_...        # or in the environment
HF_USER=<your-hf-username> bash logs/publish_solved.sh --upload
```

Two open items remain after that, both outside this repo: whether the daemon accepts a
**community-published** policy declaring `phase` (`command.encoding`, as `alpha_ground_pick` does) —
the format validates here and the official set uses it — and real-robot validation, which is still
**zero** for every policy above.
