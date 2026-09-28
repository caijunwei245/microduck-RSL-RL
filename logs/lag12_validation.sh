#!/bin/bash
# Validation for the 2026-09-27 latency default change (3-6 dithered -> 1-2 held coherently).
# 1) the tail over 12 episodes (historical default: 2 of 6 stood still)
# 2) the deployed rows under the new latency (regression check)
cd /home/iisr/.unsloth/studio/sandbox/__LOCALID_W8vum7H/microduck_rl || exit 1
TURN=logs/rsl_rl/velocity/2026-09-26_19-46-52_turn03_woboff/model_4998.pt
WALK=logs/rsl_rl/velocity/2026-09-19_01-26-11_dc_long_0919_0125/model_67997.pt
{
echo "############ latency default 1-2 steps, held coherently: validation ############"
echo "date: $(date -Is)"
echo
echo "=== the turn tail over 12 episodes (cmd 0.3, seed 0, new default latency) ==="
timeout 2400 uv run python logs/dr_tail_probe.py --rounds 12 --ckpts "$TURN" 2>/dev/null \
  | sed -n '/per-episode DR vs turn outcome/,$p' | grep -aE "^ *[0-9]+ +0\.|episodes below|<--"
echo
echo "=== deployed rows under the new latency (3 rounds each) ==="
for row in "velocity walk" "velocity turn" "walk+turn"; do
  CK=$WALK; [ "$row" != "velocity walk" ] && CK=$TURN
  env CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline timeout 900 uv run python logs/skill_demo.py \
      --rounds 3 --only "$row" --seed 0 --turn 0.5 --ckpt "$CK" 2>/dev/null \
    | grep -aE "^(velocity walk|velocity turn|walk\+turn) " | sed 's/^/  /'
done
} > logs/lag12_validation.out 2>&1
echo LAG12_VALIDATION_DONE
