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
# SLUG names the output files; GATE_SLUG overrides it so two variants of the same row (e.g. the
# left-footed kick, which uses the same "ball_kick right" row spec with MICRODUCK_KICK_FOOT=left) do
# not silently overwrite each other's verdict file - measured 2026-09-28, when the left-foot run
# clobbered gate_ball_kick_right.txt.
SLUG="${GATE_SLUG:-$(echo "$SKILL" | tr ' +()' '____')}"
OUT="logs/gate_${SLUG}.txt"
GPU="${GATE_GPU:-0}"
: > "$OUT"
echo "=== ACCEPTANCE GATE: $SKILL ===" | tee -a "$OUT"
echo "checkpoint: ${CKPT:-<newest glob match>}" | tee -a "$OUT"
echo "date: $(date -Is)" | tee -a "$OUT"

# SEEDS: one 5-round run is NOT a verdict - measured over 5 seeds x 5 rounds, ball_kick scores
# anywhere from 1/5 to 4/5 and velocity walk 0/5 to 3/5 (logs/seed_sweep_results.txt).
#
# MARGINAL ROWS GET 5 SEEDS (25 rounds). Added 2026-09-27 after the simulator was measured to be
# chaotic and not bit-reproducible across resets: identical state plus identical actions diverge from
# step 1, and a task whose two behaviours are ~0.3 % apart in return flips whole episodes on that
# noise alone (logs/turn_tail_findings.md). At 3 seeds (15 rounds) the resolution is roughly +/-13 %,
# which is the same order as the effect being judged on exactly these rows:
#     velocity walk      0/5 .. 3/5 across seeds
#     velocity turn      2/6, 1/6, 0/6 across runs (all the same number at n=6)
#     walk+turn          same marginal region
#     ball_kick          1/5 .. 4/5 across seeds
# Everything else keeps 3 seeds. Either is overridden by an explicit GATE_SEEDS.
MARGINAL_RE='velocity walk|velocity turn|walk\+turn|ball_kick'
if [ -n "${GATE_SEEDS:-}" ]; then
  SEEDS="$GATE_SEEDS"; WHY="explicit GATE_SEEDS"
elif echo "$SKILL" | grep -qE "$MARGINAL_RE"; then
  SEEDS=5; WHY="marginal row (measured to flip on simulator noise; 3 seeds resolves only ~+/-13%)"
else
  SEEDS=3; WHY="default"
fi
echo "--- demonstration: $SEEDS seeds x 5 rounds ($WHY) ---" | tee -a "$OUT"
TOT=0; PAS=0
for SD in $(seq 0 $((SEEDS - 1))); do
  R=$(env CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=offline uv run python logs/skill_demo.py \
      --rounds 5 --only "$SKILL" --seed $SD ${CKPT:+--ckpt "$CKPT"} \
      --json-out "logs/gate_${SLUG}_seed${SD}.json" 2>&1)
  LINE=$(echo "$R" | grep -aE "(ALL PASS|PARTIAL|FAIL)" | tail -1)
  N=$(echo "$LINE" | grep -oE "[0-9]+/5" | head -1 | cut -d/ -f1)
  echo "  seed $SD: $(echo "$LINE" | sed 's/  */ /g')" | tee -a "$OUT"
  echo "    $(echo "$R" | grep -aE "^[a-z].*[0-9]:" | head -1 | sed 's/  */ /g')" | tee -a "$OUT"
  TOT=$((TOT + 5)); PAS=$((PAS + ${N:-0}))
done
echo "  AGGREGATE: $PAS/$TOT rounds ($((PAS * 100 / TOT))%)" | tee -a "$OUT"

# CONTINUOUS READOUTS next to the pass count (2026-09-28). A pass rate on a marginal row carries a
# +/-14-19 point interval at these episode counts; the physical readouts it is derived from do not, so
# report both. The metric per row kind comes from paired_eval.PRIMARY so gate and paired tool agree.
uv run python - "$SKILL" "$SEEDS" "logs/gate_${SLUG}" <<'PYEOF' | tee -a "$OUT"
import glob, json, statistics, sys
sys.path.insert(0, "logs")
from paired_eval import PRIMARY
skill, seeds, prefix = sys.argv[1], int(sys.argv[2]), sys.argv[3]
rows = []
for f in sorted(glob.glob(prefix + "_seed*.json")):
    rows += json.load(open(f))
if not rows:
    print("  continuous readouts: none (no --json-out data)"); raise SystemExit(0)
kind = rows[0]["kind"]
metric = PRIMARY.get(kind)
for r in rows:
    vals = [rd["readouts"].get(metric) for rd in r["rounds"] if rd["readouts"].get(metric) is not None]
    if vals:
        print(f"  seed {r['seed']}: median {metric} = {statistics.median(vals):.4g} (n={len(vals)})")
allv = [rd["readouts"][metric] for r in rows for rd in r["rounds"] if rd["readouts"].get(metric) is not None]
if len(allv) > 1:
    med = statistics.median(allv)
    sd = statistics.stdev(allv)
    hw = 1.96 * sd / len(allv) ** 0.5
    print(f"  CONTINUOUS: median {metric} over {len(allv)} rounds = {med:.4g} "
          f"(mean {statistics.fmean(allv):.4g} +/-{hw:.4g} 95%, SD {sd:.4g})")
PYEOF

# Optional paired comparison against a previous checkpoint, same seeds: the statistic that can see a
# one-round-per-seed shift (see logs/optimization_space_2026-09-28.md). Set GATE_PAIRED_WITH=<ckpt>.
if [ -n "${GATE_PAIRED_WITH:-}" ] && [ -n "$CKPT" ]; then
  echo "--- paired against a previous checkpoint (same seeds) ---" | tee -a "$OUT"
  env CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=offline uv run python logs/paired_eval.py \
      --skill "$SKILL" --a "$GATE_PAIRED_WITH" --b "$CKPT" --seeds "$SEEDS" --rounds 5 \
      --label-a before --label-b after 2>&1 | grep -aE "rounds/seed delta|delta :|seeds improved|UNPAIRED|before:|after:" \
      | tee -a "$OUT"
fi

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
