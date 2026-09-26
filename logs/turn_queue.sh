#!/bin/bash
# Turn dead-zone experiment scheduler. Pre-registration: logs/turn_deadzone_plan.md
#
# Four arms, two GPUs, launched as slots free up, then the pre-registered 3-seed sweep.
# GPU 0 is released by the 15-row demonstration (logs/skill_demo_2rounds.pid) before use;
# GPU 1 is free from the start.
cd /home/iisr/.unsloth/studio/sandbox/__LOCALID_W8vum7H/microduck_rl || exit 1
echo $$ > logs/turn_queue.pid

ARMS="turn05_warm turn03_scratch turn00_scratch turn05_scratch"
declare -A SLOT=()

state() {  # 0 = running, 1 = finished, 2 = never started
  local f="logs/$1.pid"
  [ -f "$f" ] || return 2
  kill -0 "$(cat "$f")" 2>/dev/null && return 0 || return 1
}
demo_up() { [ -f logs/skill_demo_2rounds.pid ] && kill -0 "$(cat logs/skill_demo_2rounds.pid)" 2>/dev/null; }
gpu_busy() {
  local g=$1 a
  for a in $ARMS; do
    if [ "${SLOT[$a]:-}" = "$g" ] && state "$a"; then return 0; fi
  done
  return 1
}

launch() {
  local arm=$1 gpu=$2
  case $arm in
    turn05_warm)
      # The confound-fixed redo of step 3: same checkpoint, but WARM_START restarts the step-based
      # curricula at 0, so the pin is present from the start of the run instead of on top of a
      # fully-hardened recipe. Deployment command (0.5) pinned.
      setsid env CUDA_VISIBLE_DEVICES=$gpu WANDB_MODE=offline MICRODUCK_WARM_START=1 \
        MICRODUCK_YAW_RANGE=0.5 MICRODUCK_SUSTAINED_TURN=0.3 \
        uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096 \
        --agent.resume True --agent.load-run 2026-09-19_01-26-11_dc_long_0919_0125 \
        --agent.load-checkpoint model_67997.pt --agent.max_iterations 2000 \
        --agent.run-name $arm > logs/$arm.train.log 2>&1 < /dev/null & ;;
    turn03_scratch)
      # H1's test: from scratch, whole command distribution <= 0.3, 30 % of envs held at 0.3 in place.
      setsid env CUDA_VISIBLE_DEVICES=$gpu WANDB_MODE=offline \
        MICRODUCK_YAW_RANGE=0.3 MICRODUCK_SUSTAINED_TURN=0.3 \
        uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096 \
        --agent.max_iterations 5000 --agent.run-name $arm > logs/$arm.train.log 2>&1 < /dev/null & ;;
    turn00_scratch)
      # CONTROL: identical from-scratch run with the stock command mix (no pin).
      setsid env CUDA_VISIBLE_DEVICES=$gpu WANDB_MODE=offline \
        uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096 \
        --agent.max_iterations 5000 --agent.run-name $arm > logs/$arm.train.log 2>&1 < /dev/null & ;;
    turn05_scratch)
      # H1 at the deployment point.
      setsid env CUDA_VISIBLE_DEVICES=$gpu WANDB_MODE=offline \
        MICRODUCK_YAW_RANGE=0.5 MICRODUCK_SUSTAINED_TURN=0.5 \
        uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096 \
        --agent.max_iterations 5000 --agent.run-name $arm > logs/$arm.train.log 2>&1 < /dev/null & ;;
  esac
  echo $! > logs/$arm.pid
  SLOT[$arm]=$gpu
  echo "[$(date +%H:%M:%S)] LAUNCH $arm on GPU$gpu (pid $(cat logs/$arm.pid))"
}

echo "[$(date +%H:%M:%S)] scheduler up; demo_up=$(demo_up && echo yes || echo no)"
while :; do
  left=0
  for a in $ARMS; do state "$a" || left=$((left + 1)); done
  [ "$left" = 0 ] && break
  for a in $ARMS; do
    state "$a"; st=$?
    [ "$st" = 1 ] && continue          # finished
    if [ "$st" = 2 ]; then             # never started -> place it on a free GPU
      for g in 1 0; do
        gpu_busy $g && continue
        if [ "$g" = 0 ] && demo_up; then continue; fi
        launch "$a" "$g"
        break
      done
    fi
  done
  sleep 30
done
echo "[$(date +%H:%M:%S)] all arms finished"

# ---------------------------------------------------------------- pre-registered sweep
pick() { local d f; d=$(ls -d $1 2>/dev/null | tail -1); [ -z "$d" ] && return 1
  f=$(ls "$d"/model_*.pt 2>/dev/null | sed 's/.*model_//;s/\.pt$//' | sort -n | tail -1)
  [ -z "$f" ] && return 1; echo "$d/model_$f.pt"; }

mkdir -p logs/turn_eval
BASE=logs/rsl_rl/velocity/2026-09-19_01-26-11_dc_long_0919_0125/model_67997.pt
{
  echo "############ TURN DEAD ZONE: pre-registered sweep ############"
  echo "design: logs/turn_deadzone_plan.md   date: $(date '+%Y-%m-%d %H:%M')"
  echo "baseline checkpoint: $BASE"
  for arm in base turn05_warm turn03_scratch turn00_scratch turn05_scratch; do
    if [ "$arm" = base ]; then CK=$BASE; else
      CK=$(pick "logs/rsl_rl/velocity/*_$arm") || { echo "MISSING checkpoint for $arm"; continue; }
    fi
    echo
    echo "=== $arm : $CK"
    # Gait sanity first: a weak gait's yaw number says nothing about the dead zone.
    env CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline uv run python logs/skill_demo.py \
        --rounds 3 --only "velocity walk" --seed 0 --ckpt "$CK" \
        > logs/turn_eval/${arm}_walk.out 2>&1
    grep -aE "^velocity walk" logs/turn_eval/${arm}_walk.out || echo "  (walk row unavailable)"
    for T in 0.5 0.3; do
      for S in 0 1 2; do
        env CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline uv run python logs/skill_demo.py \
            --rounds 5 --only turn --turn $T --seed $S --ckpt "$CK" \
            > logs/turn_eval/${arm}_turn${T}_seed${S}.out 2>&1
        grep -aE "^(velocity turn|walk\+turn) " logs/turn_eval/${arm}_turn${T}_seed${S}.out \
          | sed "s/^/  [turn $T seed $S] /"
      done
    done
  done
  echo
  echo "############ TRAINING-SIDE METRICS (final 200 iterations) ############"
  for arm in turn05_warm turn03_scratch turn00_scratch turn05_scratch; do
    echo "--- $arm"
    grep -aoE "(Episode_Metrics/dc_turn_gain|Metrics/twist/error_vel_yaw|Episode_Reward/track_angular_velocity): [-0-9.]+" \
      logs/$arm.train.log 2>/dev/null | tail -3
  done
} > logs/turn_scratch_results.txt 2>&1

uv run python logs/turn_report.py >> logs/turn_scratch_results.txt 2>&1
echo "TURN_QUEUE_DONE"
