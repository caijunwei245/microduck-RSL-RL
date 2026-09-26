#!/bin/bash
# Turn-tail fix: does paying for the in-place turn (MICRODUCK_TURN_FOCUS) close the 0.32 % payoff gap?
# Pre-registration: logs/turn_focus_plan.md
#
# Three arms off the SAME promoted checkpoint, +2,000 iterations each, one knob apart:
#   turnfocus2 (2.0), turnfocus4 (4.0), turnfocus0 (0.0 = control, more iterations only)
cd /home/iisr/.unsloth/studio/sandbox/__LOCALID_W8vum7H/microduck_rl || exit 1
echo $$ > logs/turn_focus.pid

SRC_RUN=2026-09-26_19-46-52_turn03_woboff
SRC_CKPT=model_4998.pt
COMMON="MICRODUCK_YAW_RANGE=0.3 MICRODUCK_SUSTAINED_TURN=0.3 MICRODUCK_ANGULAR_WOBBLE=0"

for a in turn05_warm turn03_scratch turn03_woboff turn00_scratch turn00_ship turn03_woboff5k; do
  f="logs/$a.pid"
  if [ -f "$f" ] && kill -0 "$(cat "$f")" 2>/dev/null; then
    echo "ABORT: $a still running"; exit 1
  fi
done

launch() {  # $1 arm, $2 gpu, $3 focus weight
  setsid env CUDA_VISIBLE_DEVICES=$2 WANDB_MODE=offline $COMMON MICRODUCK_TURN_FOCUS=$3 \
    uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096 \
    --agent.resume True --agent.load-run "$SRC_RUN" --agent.load-checkpoint "$SRC_CKPT" \
    --agent.max_iterations 2000 --agent.run-name "$1" \
    > logs/$1.train.log 2>&1 < /dev/null &
  echo $! > logs/$1.pid
  echo "[$(date +%H:%M:%S)] LAUNCH $1 on GPU$2 (focus=$3, pid $(cat logs/$1.pid))"
}

launch turnfocus2 0 2.0
launch turnfocus4 1 4.0

# third arm (the control) takes whichever GPU frees first
while :; do
  for a in turnfocus2 turnfocus4; do
    kill -0 "$(cat logs/$a.pid)" 2>/dev/null || { g=$([ "$a" = turnfocus2 ] && echo 0 || echo 1)
      launch turnfocus0 "$g" 0.0; break 2; }
  done
  sleep 30
done

for a in turnfocus2 turnfocus4 turnfocus0; do
  while kill -0 "$(cat logs/$a.pid)" 2>/dev/null; do sleep 60; done
  echo "[$(date +%H:%M:%S)] $a finished"
done
sleep 30

pick() { local d f; d=$(ls -d $1 2>/dev/null | tail -1); [ -z "$d" ] && return 1
  f=$(ls "$d"/model_*.pt 2>/dev/null | sed 's/.*model_//;s/\.pt$//' | sort -n | tail -1)
  [ -z "$f" ] && return 1; echo "$d/model_$f.pt"; }

mkdir -p logs/turn_eval
{
  echo "############ TURN FOCUS: does the bonus close the tail? ############"
  echo "design: logs/turn_focus_plan.md   date: $(date '+%Y-%m-%d %H:%M')"
  echo "baseline (before any arm): logs/rsl_rl/velocity/2026-09-26_19-46-52_turn03_woboff/model_4998.pt"
  echo "    -> tail measured at cmd 0.3, seed 0: episodes 5-6 = 0.073 / 0.042 rad/s (stand-still)"
  for arm in turnfocus0 turnfocus2 turnfocus4; do
    CK=$(pick "logs/rsl_rl/velocity/*_$arm") || { echo "MISSING $arm"; continue; }
    echo
    echo "=== $arm : $CK"
    echo "--- the tail itself (6 episodes, cmd 0.3, seed 0; primary readout)"
    sed -n '/per-episode DR vs turn outcome/,/cell =/p' <(
      timeout 1200 uv run python logs/dr_tail_probe.py --rounds 6 --ckpts "$CK" 2>/dev/null) \
      | sed 's/^/    /'
    echo "--- walk row (must stay >= 0.20 m/s)"
    env CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline uv run python logs/skill_demo.py \
        --rounds 3 --only "velocity walk" --seed 0 --ckpt "$CK" 2>/dev/null \
      | grep -aE "^velocity walk" | sed 's/^/    /'
    echo "--- 3 seeds x 5 rounds sweep"
    for T in 0.5 0.3; do
      for S in 0 1 2; do
        env CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline uv run python logs/skill_demo.py \
            --rounds 5 --only turn --turn $T --seed $S --ckpt "$CK" \
            > logs/turn_eval/${arm}_turn${T}_seed${S}.out 2>&1
        grep -aE "^velocity turn " logs/turn_eval/${arm}_turn${T}_seed${S}.out | sed "s/^/    [$T s$S] /"
      done
    done
  done
} > logs/turn_focus_results.txt 2>&1

uv run python logs/turn_report.py >> logs/turn_focus_results.txt 2>&1
echo "TURN_FOCUS_DONE"
