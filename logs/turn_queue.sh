#!/bin/bash
# Turn dead-zone experiment scheduler. Pre-registration: logs/turn_deadzone_plan.md
#
# Arms are launched as GPU slots free up (at most one per GPU), then the pre-registered 3-seed
# sweep runs and the decision rules are applied.
#
# AMENDED 2026-09-26, after the wobble diagnostic (logs/wobble_deadzone.sh) came back:
#   the dead zone at cmd 0.3 is mostly the `angular_wobble` tax - on a matched pair (same source
#   checkpoint, one env var apart) the wobble-free arm tracks 0.3 at gain 1.1 while the wobble-on
#   arm sits at 0.36-0.50. So:
#     * `turn05_scratch` (the pre-registered droppable arm) is DROPPED;
#     * `turn03_woboff` replaces it: the recommended FIX (deployment ckpt + WARM_START + pin 0.3 +
#       MICRODUCK_ANGULAR_WOBBLE=0), i.e. the recipe the diagnostic says to try, on the deployed
#       policy rather than on a from-scratch gait;
#     * `ref_wobbleoff` is added to the sweep as a REFERENCE policy (the wobble-free arm of the
#       diagnostic pair) so the queue's own table contains the contrast measured under identical
#       conditions. Its absolute numbers are not comparable to the others - it is an older recipe.
#
# BUGFIX: the first version counted "arms that are not running" and broke at 0, which both fired
# early (all arms running - impossible here) and never fired at all (all arms finished). It is now
# "break when every arm has finished", which is what the eval block needs.
cd /home/iisr/.unsloth/studio/sandbox/__LOCALID_W8vum7H/microduck_rl || exit 1
echo $$ > logs/turn_queue.pid

ARMS="turn05_warm turn03_scratch turn03_woboff turn00_scratch"

declare -A SLOT=()
state() {  # 0 = running, 1 = finished, 2 = never started
  local f="logs/$1.pid"
  [ -f "$f" ] || return 2
  kill -0 "$(cat "$f")" 2>/dev/null && return 0 || return 1
}
# A restarted scheduler must not double-book a GPU an arm is ALREADY on: recover each running arm's
# slot from the process environment (the arm was started as `setsid env CUDA_VISIBLE_DEVICES=N ...`,
# and the pid survives the exec chain, so /proc/<pid>/environ still carries it).
gpu_of() { tr '\0' '\n' < "/proc/$(cat "logs/$1.pid")/environ" 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p'; }
for _a in $ARMS; do
  if state "$_a"; then SLOT[$_a]=$(gpu_of "$_a"); echo "[boot] $_a already running on GPU${SLOT[$_a]:-?}"; fi
done
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
      # Confound-fixed redo of optimization_plan step 3: same checkpoint, but WARM_START restarts the
      # step-based curricula at 0, so the pin is present from the start of the run instead of on top
      # of a fully-hardened recipe. Deployment command (0.5) pinned.
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
    turn03_woboff)
      # THE FIX, per the wobble diagnostic: pin 0.3 AND remove the tax that prices the scuffing a
      # slow pivot-step needs. Warm start + MICRODUCK_WARM_START so the curricula restart, since the
      # tax was part of the recipe the loaded policy grew up under.
      setsid env CUDA_VISIBLE_DEVICES=$gpu WANDB_MODE=offline MICRODUCK_WARM_START=1 \
        MICRODUCK_YAW_RANGE=0.3 MICRODUCK_SUSTAINED_TURN=0.3 MICRODUCK_ANGULAR_WOBBLE=0 \
        uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096 \
        --agent.resume True --agent.load-run 2026-09-19_01-26-11_dc_long_0919_0125 \
        --agent.load-checkpoint model_67997.pt --agent.max_iterations 2000 \
        --agent.run-name $arm > logs/$arm.train.log 2>&1 < /dev/null & ;;
    turn00_scratch)
      # CONTROL: identical from-scratch run with the stock command mix (no pin).
      setsid env CUDA_VISIBLE_DEVICES=$gpu WANDB_MODE=offline \
        uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096 \
        --agent.max_iterations 5000 --agent.run-name $arm > logs/$arm.train.log 2>&1 < /dev/null & ;;
  esac
  echo $! > logs/$arm.pid
  SLOT[$arm]=$gpu
  echo "[$(date +%H:%M:%S)] LAUNCH $arm on GPU$gpu (pid $(cat logs/$arm.pid))"
}

echo "[$(date +%H:%M:%S)] scheduler up; ARMS='$ARMS'"
while :; do
  pending=0
  for a in $ARMS; do
    state "$a"; [ "$?" = 1 ] || pending=$((pending + 1))   # running or not-yet-started
  done
  [ "$pending" = 0 ] && break                            # every arm finished
  for a in $ARMS; do
    state "$a"; st=$?
    [ "$st" = 1 ] && continue                            # finished
    if [ "$st" = 2 ]; then
      for g in 1 0; do
        gpu_busy $g && continue
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
REF=logs/rsl_rl/velocity/2026-09-17_00-03-38_yawonly_long/model_63998.pt   # wobble-free reference
{
  echo "############ TURN DEAD ZONE: pre-registered sweep ############"
  echo "design: logs/turn_deadzone_plan.md (+ the 2026-09-26 amendment in logs/turn_queue.sh)"
  echo "date: $(date '+%Y-%m-%d %H:%M')"
  echo "baseline checkpoint : $BASE"
  echo "wobble-free ref     : $REF   (older yaw-only recipe - contrast only, not comparable)"
  for arm in base turn05_warm turn03_scratch turn03_woboff turn00_scratch ref_wobbleoff; do
    case $arm in
      base) CK=$BASE ;;
      ref_wobbleoff) CK=$REF ;;
      *) CK=$(pick "logs/rsl_rl/velocity/*_$arm") || { echo "MISSING checkpoint for $arm"; continue; } ;;
    esac
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
  echo "############ TRAINING-SIDE METRICS (final lines of each arm) ############"
  for arm in turn05_warm turn03_scratch turn03_woboff turn00_scratch; do
    echo "--- $arm"
    grep -aoE "(Episode_Metrics/dc_turn_gain|Metrics/twist/error_vel_yaw|Episode_Reward/angular_wobble|Episode_Reward/track_angular_velocity): [-0-9.]+" \
      logs/$arm.train.log 2>/dev/null | tail -4
  done
} > logs/turn_scratch_results.txt 2>&1

uv run python logs/turn_report.py >> logs/turn_scratch_results.txt 2>&1
echo "TURN_QUEUE_DONE"
