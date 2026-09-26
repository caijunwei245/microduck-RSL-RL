#!/bin/bash
# Turn fix consolidation run. Pre-registration: logs/turn_ship_plan.md
# Follow-up to logs/turn_deadzone_verdict.md: the dead zone is the `angular_wobble` tax, so the
# remaining question is what the SHIPPABLE recipe is - untaxed with the pin, or untaxed alone.
#
#   GPU0: turn03_woboff CONTINUED 2000 -> 5000 iterations (same run dir, same settings, no
#         WARM_START because it is a resume of a run that already warm-started)
#   GPU1: turn00_ship              fresh warm start from the deployed checkpoint, WOBBLE OFF,
#                                  stock command mix (no pin) - tests whether the pin is needed at all
cd /home/iisr/.unsloth/studio/sandbox/__LOCALID_W8vum7H/microduck_rl || exit 1
echo $$ > logs/turn_ship.pid

SRC_RUN=2026-09-19_01-26-11_dc_long_0919_0125
SRC_CKPT=model_67997.pt
WOB_RUN=2026-09-26_16-26-03_turn03_woboff

for a in turn05_warm turn03_scratch turn03_woboff turn00_scratch; do
  f="logs/$a.pid"
  if [ -f "$f" ] && kill -0 "$(cat "$f")" 2>/dev/null; then
    echo "ABORT: $a is still running; refusing to share a GPU"; exit 1
  fi
done

echo "[$(date +%H:%M:%S)] launching the two ship arms"
setsid env CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline \
  MICRODUCK_YAW_RANGE=0.3 MICRODUCK_SUSTAINED_TURN=0.3 MICRODUCK_ANGULAR_WOBBLE=0 \
  uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096 \
  --agent.resume True --agent.load-run "$WOB_RUN" --agent.load-checkpoint model_1999.pt \
  --agent.max_iterations 3000 --agent.run-name turn03_woboff \
  > logs/turn03_woboff5k.train.log 2>&1 < /dev/null &
echo $! > logs/turn03_woboff5k.pid

setsid env CUDA_VISIBLE_DEVICES=1 WANDB_MODE=offline MICRODUCK_WARM_START=1 \
  MICRODUCK_ANGULAR_WOBBLE=0 \
  uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096 \
  --agent.resume True --agent.load-run "$SRC_RUN" --agent.load-checkpoint "$SRC_CKPT" \
  --agent.max_iterations 5000 --agent.run-name turn00_ship \
  > logs/turn00_ship.train.log 2>&1 < /dev/null &
echo $! > logs/turn00_ship.pid

for a in turn03_woboff5k turn00_ship; do
  echo "[$(date +%H:%M:%S)] $a pid $(cat logs/$a.pid)"
done

for a in turn03_woboff5k turn00_ship; do
  while kill -0 "$(cat logs/$a.pid)" 2>/dev/null; do sleep 60; done
  echo "[$(date +%H:%M:%S)] $a finished"
done
sleep 30

pick() { local d f; d=$(ls -d $1 2>/dev/null | tail -1); [ -z "$d" ] && return 1
  f=$(ls "$d"/model_*.pt 2>/dev/null | sed 's/.*model_//;s/\.pt$//' | sort -n | tail -1)
  [ -z "$f" ] && return 1; echo "$d/model_$f.pt"; }

W5=$(pick "logs/rsl_rl/velocity/*_turn03_woboff")
SH=$(pick "logs/rsl_rl/velocity/*_turn00_ship")
{
  echo "############ TURN FIX CONSOLIDATION: pre-registered sweep ############"
  echo "design: logs/turn_ship_plan.md   date: $(date '+%Y-%m-%d %H:%M')"
  echo "pin + wobble-off, continued to 5k : $W5"
  echo "wobble-off, NO pin, 5k            : $SH"
  for pair in "turn03_woboff5k:$W5" "turn00_ship:$SH"; do
    arm=${pair%%:*}; CK=${pair#*:}
    echo
    echo "=== $arm : $CK"
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
} > logs/turn_ship_results.txt 2>&1

uv run python logs/turn_report.py >> logs/turn_ship_results.txt 2>&1
echo "TURN_SHIP_DONE"
