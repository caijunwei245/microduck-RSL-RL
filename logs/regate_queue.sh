#!/bin/bash
# Re-run the acceptance gate for every row (optimization_space_2026-09-28.md item 1, second half).
#
# Why: the files in logs/gate_*.txt mix eras - they predate the fix in the turn, the measured-latency
# default, and (for two of them) the verdict parser itself, so readings like "FAIL 15/15 (100 %)" sat
# next to a ledger that says otherwise. This refreshes them under the CURRENT recipe.
#
# Seed policy comes from acceptance_gate.sh itself: the marginal rows (velocity walk / turn / walk+turn
# / ball_kick) take 5 seeds = 25 rounds, everything else 3 seeds = 15 rounds. `sitstand` is pinned to 5
# as well because it is the weakest measured row (2/5 in the 5-round rotation). `velocity turn` is
# skipped: it was gated fresh this morning (21/25, logs/gate_velocity_turn.txt).
#
# Turn rows need the promoted policy explicitly, because the gate's default checkpoint for a row is the
# row's own glob, which still points at the deployed walking candidate.
cd /home/iisr/.unsloth/studio/sandbox/__LOCALID_W8vum7H/microduck_rl || exit 1
echo $$ > logs/regate.pid
TURN=logs/rsl_rl/velocity/2026-09-26_19-46-52_turn03_woboff/model_4998.pt

ROWS=(
  "velocity walk"
  "walk+turn"
  "ball_kick right"
  "sitstand"
  "ground_pick"
  "rollers (fast)"
  "swizzle"
  "roller_slope"
  "roller_standup"
  "roller_crouch"
  "standup floor flip"
  "velstand floor flip"
  "spin"
  "roulade"
)

run_row() {  # $1 = row, $2 = gpu
  local row="$1" gpu="$2" ck="" task="" seeds=""
  case "$row" in
    "walk+turn") ck="$TURN" ;;
    "sitstand") seeds=5 ;;
    "standup floor flip") task="Mjlab-StandUp-Flat-MicroDuck" ;;
    "velstand floor flip") task="Mjlab-VelStand-Flat-MicroDuck" ;;
  esac
  echo "[$(date +%H:%M:%S)] GPU$gpu START $row${ck:+ (ckpt $ck)}${seeds:+ (GATE_SEEDS=$seeds)}${task:+ (task $task)}"
  env GATE_GPU="$gpu" ${seeds:+GATE_SEEDS=$seeds} \
    bash logs/acceptance_gate.sh "$row" "$ck" "$task" > logs/regate_$(echo "$row" | tr ' +()' '____').log 2>&1
  echo "[$(date +%H:%M:%S)] GPU$gpu DONE  $row -> $(grep -a VERDICT "logs/gate_$(echo "$row" | tr ' +()' '____').txt" | tail -1)"
}

worker() {  # $1 = gpu, $2 = parity (0 = even indices, 1 = odd)
  local gpu="$1" par="$2" i
  for i in "${!ROWS[@]}"; do
    [ $((i % 2)) -eq "$par" ] || continue
    run_row "${ROWS[$i]}" "$gpu"
  done
}

worker 0 0 &
w0=$!
worker 1 1 &
w1=$!
wait $w0 $w1

{
  echo "############ acceptance gate, refreshed $(date -Is) ############"
  echo "(current recipe: measured latency 1-2 steps held; turn rows on the wobble-tax fix)"
  echo
  for i in "${!ROWS[@]}"; do
    row="${ROWS[$i]}"
    slug=$(echo "$row" | tr ' +()' '____')
    printf "%-22s %s\n" "$row" "$(grep -a VERDICT logs/gate_${slug}.txt 2>/dev/null | tail -1 | sed 's/=== VERDICT: //;s/ ===//')"
  done
  echo
  echo "velocity turn          $(grep -a VERDICT logs/gate_velocity_turn.txt | tail -1 | sed 's/=== VERDICT: //;s/ ===//')   (gated 2026-09-28 09:03)"
} > logs/regate_summary.txt 2>&1
cat logs/regate_summary.txt
echo REGATE_DONE
