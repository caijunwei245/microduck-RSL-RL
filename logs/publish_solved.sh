#!/usr/bin/env bash
# Export and publish the SOLVED skills (spin / roller_standup / turn / ball_kick L+R) through the real
# `uv run publish` path: ONNX export from the checkpoint -> shape gate -> smoke run -> manifest build +
# validate -> README -> (upload).
#
#   bash logs/publish_solved.sh              # dry run: stage publish-<name>/ locally, upload nothing
#   HF_USER=<hf-username> bash logs/publish_solved.sh --upload   # create/update the Hub repos
#
# Every timing below is MEASURED here, not copied from the official set - the per-skill evidence file
# is named in the table. Two of them corrected a first (wrong) answer:
#   * roller_standup: the mixed-spawn sweep says "0.6 s passes 15/15" and that is unreadable, because
#     the task spawns 50 % belly / 50 % ALREADY STANDING (the standing bucket teaches HOLDING, and it
#     passes a short window for free). Pinned per bucket (logs/roller_standup_rise.txt): belly rise
#     0.20 s, back 0.32 s, standing 0 - so 1.0 s covers the worst bucket 3x and every bucket ends at
#     139 mm. Pinning the EVENT is a silent no-op: `ground_state_mix` rewrites those probabilities
#     before each reset, so the pin has to go through the curriculum.
#   * spin: 2.6 s = its own envelope's brake end (SPIN_BRAKE_END 0.650), one full turn; a longer window
#     delivers the SAME rotation and then sits in the commanded rest phase
#     (logs/publish_duration_spin15.txt).
# ball_kick's window does not change what the kick does - every window >= 0.5 s contains the strike
# (contact at step 5-8) and leaves the robot standing at 115-116 mm - 1.5 s is chosen so the ball is
# demonstrably clear (0.30-0.44 m) at handoff; the ball keeps rolling after the window either way.
set -uo pipefail
cd "$(dirname "$0")/.."

UPLOAD=0
[ "${1:-}" = "--upload" ] && UPLOAD=1
HF_USER="${HF_USER:-<user>}"
ONLY="${ONLY:-}"        # ONLY=<name> re-stages one skill (the export is the slow part)
FAILED=0

ckpt() {  # row glob -> newest checkpoint (skill_demo.newest, the same resolution the demo rows use)
  # The mdp import banner goes to stderr while importing: it used to land INSIDE the
  # --checkpoint-file argument and kill publish on a four-line "path".
  uv run python - "$1" <<'PY'
import sys
sys.path.insert(0, "logs")
_out, sys.stdout = sys.stdout, sys.stderr
import skill_demo                                 # noqa: E402
sys.stdout = _out
p = skill_demo.newest(sys.argv[1])
assert p, f"no checkpoint matches {sys.argv[1]!r}"
print(p)
PY
}

stage() {  # name task glob envvars... -- publish flags...
  local name="$1" task="$2" glob="$3" envs="$4"; shift 4
  [ -n "$ONLY" ] && [ "$name" != "$ONLY" ] && return 0
  local ck
  ck="$(ckpt "$glob")" || { echo "!! $name: no checkpoint for $glob"; FAILED=1; return 1; }
  echo "=================== $name ==================="
  echo "task:       $task"
  echo "checkpoint: $ck"
  [ -n "$envs" ] && echo "env:        $envs"
  local mode="--dry-run"
  [ "$UPLOAD" = "1" ] && mode=""
  # shellcheck disable=SC2086
  env $envs WANDB_MODE=offline uv run publish --task "$task" --checkpoint-file "$ck" \
      --repo "$HF_USER/microduck-$name" --name "$name" "$@" $mode \
    || { echo "!! $name: publish failed"; FAILED=1; }
}

# name             task                                 checkpoint glob                                     env for the export
stage spin          Mjlab-Spin-Flat-MicroDuck          'logs/rsl_rl/spin/*_spin_warmstand/model_*.pt'      '' \
      --description "Spin in place on the rollers: one full turn, ending at the commanded zero rate." \
      --entry-pose "standing on the rollers (passive wheels)" \
      --kind episodic --duration-s 2.6 \
      --command-encoding phase --period-s 4.0 --end-phase 0.65

stage roller_standup Mjlab-RollerStandUp-Flat-MicroDuck 'logs/rsl_rl/roller_standup/*rollerstandup_lag12/model_*.pt' '' \
      --description "Get up onto the passive wheels from the belly or the back, then hold the roller stand." \
      --entry-pose "belly, back, or already standing on the rollers - it gets up from any of them" \
      --kind episodic --duration-s 1.0
      # constant command (the task trains with |twist| <= 0.01/0.05 and the demo runs it at zero);
      # no --slot: SLOTS has no roller_standup.

stage ball_kick_right Mjlab-BallKick-Flat-MicroDuck     'logs/rsl_rl/ball_kick_right/*kick_r4/model_*.pt'   '' \
      --description "Kick the ball forward with the right foot, then recover to the stand." \
      --entry-pose "standing, ball in front of the right foot" \
      --kind episodic --duration-s 1.5 --slot kick_right

stage ball_kick_left  Mjlab-BallKick-Flat-MicroDuck     'logs/rsl_rl/ball_kick_left/*/model_*.pt'           'MICRODUCK_KICK_FOOT=left' \
      --description "Kick the ball forward with the left foot, then recover to the stand." \
      --entry-pose "standing, ball in front of the left foot" \
      --kind episodic --duration-s 1.5 --slot kick_left

# The turn policy is a WALKING gait (same task, wobble tax off, sustained-turn sampling), so it is
# perpetual with the walk slot - no duration to declare. Paired on the walk row against the policy the
# walk row currently ships (logs/paired_walk_vs_turn.txt): 5 seeds, rounds/seed delta +0.00 (5 ties),
# v_mean +0.0088 +/-0.0033 in its favour, i.e. shipping it does not cost the walk row.
#
# `--twist-help` is not decoration here: the constant encoding renders `command.twist` as
# "unused (zeros)", which is true for the zero-command skills (kick, roller_standup) but FALSE for a
# gait - the daemon drives a real [vx, vy, wz] into a walk-slot policy, and wz is exactly what this
# build was fixed for. The lock-step with the rest of the family is the point.
stage walk_turn     Mjlab-Velocity-Flat-MicroDuck       'logs/rsl_rl/velocity/*turn03_woboff/model_*.pt'    '' \
      --description "Walk, and turn in place at the commanded rate (wobble tax off: 0.5 rad/s at gain 1.17-1.23). Loads into the walk slot; the factory walk policy is a different repo." \
      --kind perpetual --slot walk \
      --twist-help "[vx, vy, wz] = the daemon's velocity command. This build is the turn-fixed one: it tracks |wz| = 0.5 rad/s at gain 1.17-1.23 and matches the current walk policy on the walk row."

echo
echo "=== staged ==="
for d in publish-spin publish-roller_standup publish-ball_kick_right publish-ball_kick_left publish-walk_turn; do
  ls -1 "$d" 2>/dev/null | tr '\n' ' ' | sed "s|^|$d/: |"
  echo
done
[ "$UPLOAD" = "0" ] && echo "(dry run - nothing uploaded; re-run with --upload and HF_USER=<name>)"
exit "$FAILED"
