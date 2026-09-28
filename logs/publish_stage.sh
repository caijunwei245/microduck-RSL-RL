#!/usr/bin/env bash
# Stage the four skills schema 2 already knows how to drive, through the REAL `uv run publish` path
# (ONNX export from the checkpoint -> shape gate -> smoke run -> manifest build + validate -> README),
# stopping at --dry-run. Nothing is uploaded: there is no HF token in this environment, and whether
# the daemon accepts a COMMUNITY-published `phase`/`posture_flag` policy is a decision in the
# `microduck` repo. See logs/publish_dryrun.md for the contract story and logs/publish_dryrun_e2e.txt
# for the first (hand-typed) run, whose two failures were wrong checkpoint paths - which is why this
# script resolves every checkpoint through skill_demo.newest instead of taking one on the command line.
#
# Timing per skill comes from measurement, not from the official set's numbers:
#   ground_pick   3.0 s of the 4.0 s cycle  (logs/publish_duration_sweep.txt)
#   roller_crouch 3.0 s of the 5.0 s cycle  (logs/publish_duration_sweep.txt)
#   spin          2.6 s = the brake end of its own envelope (phase 0.650) - ONE full turn, handed back
#                 at the commanded zero rate. logs/publish_duration_spin15.txt: 2.5 s and 3.0 s both
#                 turned 6.5 rad but 3.5 s and 4.0 s read 0/15 on the demo criterion, because that
#                 criterion measures the LAST THIRD of the window and the envelope commands REST from
#                 phase 0.650 (2.6 s) on. The rotation is identical (6.5-6.6 rad) in every window
#                 >= 2.5 s, so the button is the brake end, not the full cycle.
#   sitstand      ramp_s = the cfg's POSTURE_RAMP_S (2.0), unwind_s 1.0, both as in alpha_sitstand
#
# Usage:  bash logs/publish_stage.sh 2>&1 | tee logs/publish_stage.txt
set -uo pipefail
cd "$(dirname "$0")/.."

SPIN_DURATION_S="${SPIN_DURATION_S:-2.6}"   # the envelope's own brake end; see the note above

ckpt() {  # row glob -> newest checkpoint, resolved the same way the demo rows do it
  # `import skill_demo` prints the mdp patch banner on stdout (weeks of debugging artefact), so the
  # banner goes to stderr while the import happens and only the path comes back on stdout. Without
  # this the banner lands INSIDE the --checkpoint-file argument and publish dies on a path that is
  # four log lines long.
  uv run python - "$1" <<'PY'
import sys
sys.path.insert(0, "logs")
_out, sys.stdout = sys.stdout, sys.stderr        # keep the import banner out of the payload
import skill_demo                                 # noqa: E402
sys.stdout = _out
p = skill_demo.newest(sys.argv[1])
assert p, f"no checkpoint matches {sys.argv[1]!r}"
print(p)
PY
}

FAILED=0
stage() {  # name task glob  <extra publish flags...>
  local name="$1" task="$2" glob="$3"; shift 3
  local ck
  ck="$(ckpt "$glob")" || { echo "!! $name: no checkpoint for $glob"; FAILED=1; return 1; }
  echo "=================== $name ==================="
  echo "task:       $task"
  echo "checkpoint: $ck"
  uv run publish --task "$task" --checkpoint-file "$ck" \
                 --repo "<user>/microduck-$name" --name "$name" \
                 --description "MicroDuck $name" "$@" --dry-run \
    || { echo "!! $name: publish failed"; FAILED=1; }
}

stage ground_pick Mjlab-GroundPick-Flat-MicroDuck 'logs/rsl_rl/ground_pick/*/model_*.pt' \
      --kind episodic --duration-s 3.0 \
      --command-encoding phase --period-s 4.0 --end-phase 0.75 --slot ground_pick

stage roller_crouch Mjlab-RollerCrouch-Flat-MicroDuck 'logs/rsl_rl/roller_crouch/*rollercrouch_lag12/model_*.pt' \
      --kind episodic --duration-s 3.0 \
      --command-encoding phase --period-s 5.0 --end-phase 0.6
      # no --slot: the daemon's slot list (walk, stand, sitstand, ground_pick, kick_left, kick_right,
      # roulade) has no roller_crouch, and `uv run publish --slot roller_crouch` is refused.

stage spin Mjlab-Spin-Flat-MicroDuck 'logs/rsl_rl/spin/*_spin_warmstand/model_*.pt' \
      --kind episodic --duration-s "$SPIN_DURATION_S" \
      --command-encoding phase --period-s 4.0 --end-phase "$(uv run python -c "print($SPIN_DURATION_S/4.0)")"

stage sitstand Mjlab-SitStand-Flat-MicroDuck 'logs/rsl_rl/microduck_sitstand/*/model_*.pt' \
      --kind scripted --command-encoding posture_flag \
      --ramp-s 2.0 --unwind-s 1.0 --sit 1.0 --stand 0.0 --slot sitstand

echo
echo "=== staged ==="
for d in publish-ground_pick publish-roller_crouch publish-spin publish-sitstand; do
  ls -1 "$d" 2>/dev/null | tr '\n' ' ' | sed "s|^|$d/: |"
  echo
done
exit "$FAILED"
