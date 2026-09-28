#!/bin/bash
# Item 1 of the three follow-ups: give the two skating families an arm trained UNDER the measured
# latency envelope (1-2 steps held coherently), because their pre-change checkpoints lose 12-24 points
# when evaluated under it (`logs/regate_2026-09-28.md`). Same recipe as the runs they warm start from,
# only the envelope differs; MICRODUCK_WARM_START=1 restarts the step-based curricula so the policy
# re-adapts from the start of the schedule rather than inheriting a hardened one.
cd /home/iisr/.unsloth/studio/sandbox/__LOCALID_W8vum7H/microduck_rl || exit 1
echo $$ > logs/roller_lag12.pid

CRUN=2026-09-24_08-58-22_crouch_delta2
SSTALL=$(ls -d logs/rsl_rl/roller_standup/*_roller_standup_stall | tail -1)
SRUN=$(basename "$SSTALL")

# roller_crouch: the source run's recipe was MICRODUCK_CROUCH_DELTA=4.0
setsid env CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline MICRODUCK_WARM_START=1 MICRODUCK_CROUCH_DELTA=4.0 \
  uv run train Mjlab-RollerCrouch-Flat-MicroDuck --env.scene.num-envs 4096 \
  --agent.resume True --agent.load-run "$CRUN" --agent.load-checkpoint model_2498.pt \
  --agent.max_iterations 2000 --agent.run-name rollercrouch_lag12 \
  > logs/rollercrouch_lag12.train.log 2>&1 < /dev/null &
echo $! > logs/rollercrouch_lag12.pid

# roller_standup: the source run used the tilt-clause stall backstop
setsid env CUDA_VISIBLE_DEVICES=1 WANDB_MODE=offline MICRODUCK_WARM_START=1 MICRODUCK_STALL_TILT_G=-0.9 \
  uv run train Mjlab-RollerStandUp-Flat-MicroDuck --env.scene.num-envs 4096 \
  --agent.resume True --agent.load-run "$SRUN" --agent.load-checkpoint model_2498.pt \
  --agent.max_iterations 2000 --agent.run-name rollerstandup_lag12 \
  > logs/rollerstandup_lag12.train.log 2>&1 < /dev/null &
echo $! > logs/rollerstandup_lag12.pid

for a in rollercrouch_lag12 rollerstandup_lag12; do
  echo "[$(date +%H:%M:%S)] LAUNCH $a pid $(cat logs/$a.pid)"
done

for a in rollercrouch_lag12 rollerstandup_lag12; do
  while kill -0 "$(cat logs/$a.pid)" 2>/dev/null; do sleep 60; done
  echo "[$(date +%H:%M:%S)] $a finished"
done
sleep 20
{
  echo "############ roller families, arm trained under the measured envelope ############"
  echo "date: $(date -Is)"
  echo "source recipes kept; only the latency envelope differs (now the default: 1-2 steps held)"
  for a in rollercrouch_lag12 rollerstandup_lag12; do
    echo "--- $a: $(grep -aoE 'Learning iteration [0-9]+/[0-9]+' logs/$a.train.log | tail -1)"
  done
} > logs/roller_lag12_results.txt 2>&1

# pre-registered readout: the same 25-round gate as the comparison it is trying to beat
( env GATE_GPU=0 GATE_SEEDS=5 bash logs/acceptance_gate.sh "roller_crouch" "$(ls -d logs/rsl_rl/roller_crouch/*_rollercrouch_lag12/model_*.pt | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1 | xargs -I{} echo "$(ls -d logs/rsl_rl/roller_crouch/*_rollercrouch_lag12)/model_{}.pt")" "" >> logs/roller_lag12_results.txt 2>&1 ) &
( env GATE_GPU=1 GATE_SEEDS=5 bash logs/acceptance_gate.sh "roller_standup" "$(ls -d logs/rsl_rl/roller_standup/*_rollerstandup_lag12/model_*.pt | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1 | xargs -I{} echo "$(ls -d logs/rsl_rl/roller_standup/*_rollerstandup_lag12)/model_{}.pt")" "" >> logs/roller_lag12_results.txt 2>&1 ) &
wait
grep -aE "VERDICT|AGGREGATE" logs/roller_lag12_results.txt
echo ROLLER_LAG12_DONE
