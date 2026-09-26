#!/bin/bash
# Is the in-place turn dead zone a WOBBLE-TAX artifact? (2026-09-26, supplementary diagnostic)
#
# The dead zone: commanded in-place yaw 0.5 -> achieved 0.249, 0.4 -> 0.135, 0.3 -> 0.062,
# 0.2 -> 0.023 on the deployed policy. AGENTS.md already records that the angular-wobble cost
# (which prices the INSTANTANEOUS roll/pitch rate) is what holds turning back at cmd 0.5:
# 0.26-0.30 with it, 0.68-0.75 without. If that is also the mechanism at LOW commanded rates,
# the fix is a wobble-weight curriculum keyed on |commanded yaw| - not another pinned run.
#
# PRIMARY (matched pair - same source checkpoint `yawonly_lrbound@57999`, same code, 6000 iters,
# one env var apart):
#     yawwobble_long  wobble ON   (-3.0)
#     yawonly_long    wobble OFF  (MICRODUCK_ANGULAR_WOBBLE=0)
# SECONDARY (dose-response at cmd 0.3, confounded: these two use the `yaw05` recipe with xy
# tracking, not the yaw-only reward of the pair above):
#     yaw05_long      wobble -3.0
#     yaw05_wob15     wobble -1.5  (MICRODUCK_WOBBLE_WEIGHT=-1.5)
#
# Readout: median achieved |yaw| over 3 rounds, and its gain against the command.
cd /home/iisr/.unsloth/studio/sandbox/__LOCALID_W8vum7H/microduck_rl || exit 1
D=logs/rsl_rl/velocity
WOB=$D/2026-09-17_00-02-38_yawwobble_long/model_63998.pt
NOW=$D/2026-09-17_00-03-38_yawonly_long/model_63998.pt
W30=$D/2026-09-17_09-30-08_yaw05_long/model_69997.pt
W15=$D/2026-09-17_14-08-20_yaw05_wob15/model_66997.pt

sweep() {  # $1 tag, $2 ckpt, $3 command
  env CUDA_VISIBLE_DEVICES=1 WANDB_MODE=offline uv run python logs/skill_demo.py \
    --rounds 3 --only turn --turn "$3" --seed 0 --ckpt "$2" 2>/dev/null \
    | grep -aE "^velocity turn " | sed "s/^/cmd $3  $1  /"
}

{
  echo "############ WOBBLE TAX vs the in-place turn dead zone ############"
  echo "date: $(date -Is)"
  echo "PRIMARY matched pair (yawonly_lrbound@57999 + 6000 iters, one env var apart):"
  echo "  wobble ON : $WOB"
  echo "  wobble OFF: $NOW"
  for cmd in 0.5 0.3 0.2; do
    sweep "wobble_ON  " "$WOB" "$cmd"
    sweep "wobble_OFF " "$NOW" "$cmd"
  done
  echo
  echo "SECONDARY dose-response at cmd 0.3 (confounded: yaw05 recipe, xy tracking on):"
  echo "  wobble -3.0: $W30"
  echo "  wobble -1.5: $W15"
  sweep "wobble_-3.0" "$W30" 0.3
  sweep "wobble_-1.5" "$W15" 0.3
} > logs/wobble_deadzone.txt 2>&1
echo "WOBBLE_DEADZONE_DONE"
