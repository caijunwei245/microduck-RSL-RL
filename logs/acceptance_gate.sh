#!/bin/bash
# ACCEPTANCE GATE (optimization_plan.md D1): one command that turns a training run into a verdict.
#
#   logs/acceptance_gate.sh "<demo skill name>" <checkpoint> [task_id]
#
# Runs (a) the 5-round demonstration for that skill and (b) the tilt-keyed family evaluation when a
# task id is given, writes both to logs/gate_<slug>.txt and prints a one-line verdict. This is the
# check that caught five wrong verdicts in one session; run it after every training run.
set -u
cd /home/iisr/.unsloth/studio/sandbox/__LOCALID_W8vum7H/microduck_rl
SKILL="$1"; CKPT="${2:-}"; TASK="${3:-}"
SLUG=$(echo "$SKILL" | tr ' +()' '____')
OUT="logs/gate_${SLUG}.txt"
GPU="${GATE_GPU:-0}"
: > "$OUT"
echo "=== ACCEPTANCE GATE: $SKILL ===" | tee -a "$OUT"
echo "checkpoint: ${CKPT:-<newest glob match>}" | tee -a "$OUT"
echo "date: $(date -Is)" | tee -a "$OUT"

# SEEDS: one 5-round run is NOT a verdict - measured over 5 seeds x 5 rounds, ball_kick scores
# anywhere from 1/5 to 4/5 and velocity walk 0/5 to 3/5 (logs/seed_sweep_results.txt), so the gate
# aggregates 3 seeds (15 rounds) by default. Override with GATE_SEEDS.
SEEDS="${GATE_SEEDS:-3}"
echo "--- demonstration: $SEEDS seeds x 5 rounds ---" | tee -a "$OUT"
TOT=0; PAS=0
for SD in $(seq 0 $((SEEDS - 1))); do
  R=$(env CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=offline uv run python logs/skill_demo.py \
      --rounds 5 --only "$SKILL" --seed $SD ${CKPT:+--ckpt "$CKPT"} 2>&1)
  LINE=$(echo "$R" | grep -aE "(ALL PASS|PARTIAL|FAIL)" | tail -1)
  N=$(echo "$LINE" | grep -oE "[0-9]+/5" | head -1 | cut -d/ -f1)
  echo "  seed $SD: $(echo "$LINE" | sed 's/  */ /g')" | tee -a "$OUT"
  echo "    $(echo "$R" | grep -aE "^[a-z].*[0-9]:" | head -1 | sed 's/  */ /g')" | tee -a "$OUT"
  TOT=$((TOT + 5)); PAS=$((PAS + ${N:-0}))
done
echo "  AGGREGATE: $PAS/$TOT rounds ($((PAS * 100 / TOT))%)" | tee -a "$OUT"

if [ -n "$TASK" ]; then
  echo "--- tilt-keyed family evaluation ---" | tee -a "$OUT"
  env CUDA_VISIBLE_DEVICES=$GPU uv run python logs/family_eval.py "$TASK" "$CKPT" 256 2>&1 \
    | tail -1 | .venv/bin/python -c "
import json,sys
d=json.loads(sys.stdin.read())
ob=(d.get('recovery') or {}).get('BY ORIENTATION (tilt-keyed)') or {}
print('  recovery(tilt-keyed):', json.dumps(ob, ensure_ascii=False))
print('  terminal pose:', d.get('final_trunk_z_mm_median'), 'mm  g', d.get('final_gravity_z_median'))
print('  terminations:', d.get('terminations'))
" | tee -a "$OUT"
fi
AGG=$(grep -a "AGGREGATE" "$OUT" | tail -1)
RATE=$(echo "$AGG" | grep -oE "\([0-9]+%\)" | tr -d '()%')   # strip the percent or [ -ge ] fails
if [ -n "$RATE" ] && [ "$RATE" -ge 80 ]; then V="PASS"; elif [ -n "$RATE" ] && [ "$RATE" -ge 50 ]; then V="PARTIAL"; else V="FAIL"; fi
echo "=== VERDICT: $V  ${AGG#  AGGREGATE: } ===" | tee -a "$OUT"
