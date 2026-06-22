#!/bin/bash
# Evaluare completa conform planului din paper.
# Ruleaza secvential toate modelele pe: KITTI clean, KITTI-C, Weather (fog/rain/snow)
# cu si fara TTA. Toate rezultatele loggate in wandb project "tinydepth".

set -euo pipefail

WORKDIR="/home/ubuntu/TinyDepth"
cd "$WORKDIR"

source /home/ubuntu/anaconda3/etc/profile.d/conda.sh
conda activate tinydepth

PYTHON="CUDA_VISIBLE_DEVICES=0 python"
WANDB="--use_wandb --wandb_project tinydepth"
DATA="--data_path $WORKDIR --png"
BASE_ARGS="--eval_mono --scales 0 $DATA $WANDB"

# -------------------------------------------------------
# Checkpoints
# -------------------------------------------------------
B6="$WORKDIR/models/Tiny-Depth-b6/models/weights_49"
UNCERT="$WORKDIR/models/Tiny-Depth-Basic-Uncertainty-Head-2/models/weights_49"
FEATSUPP="$WORKDIR/models/Tiny-Depth-Weather-Robust-Feature-Supression/models/weights_49"
URW_S2="$WORKDIR/models/URW-Depth-S2/models/weights_14"
URW_HIRES="$WORKDIR/models/URW-Depth-HiRes-S2/models/weights_7"

KITTI_C_PATH="$WORKDIR/kitti_c/kitti_c"
TTA_STEPS=5
TTA_LR=1e-4

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# -------------------------------------------------------
# 1. KITTI CLEAN — evaluate_depth.py
# -------------------------------------------------------
log "=== KITTI CLEAN (no TTA) ==="

log "b6"
eval $PYTHON evaluate_depth.py --load_weights_folder $B6 \
  --height 192 --width 640 $BASE_ARGS \
  --wandb_run_name "eval-kitti-b6"

log "uncertainty"
eval $PYTHON evaluate_depth.py --load_weights_folder $UNCERT \
  --height 192 --width 640 $BASE_ARGS \
  --wandb_run_name "eval-kitti-uncertainty"

log "featsupp"
eval $PYTHON evaluate_depth.py --load_weights_folder $FEATSUPP \
  --height 192 --width 640 --use_feature_suppression $BASE_ARGS \
  --wandb_run_name "eval-kitti-featsupp"

log "urw-s2"
eval $PYTHON evaluate_depth.py --load_weights_folder $URW_S2 \
  --height 192 --width 640 --use_feature_suppression $BASE_ARGS \
  --wandb_run_name "eval-kitti-urw-s2"

log "urw-hires"
eval $PYTHON evaluate_depth.py --load_weights_folder $URW_HIRES \
  --height 384 --width 1280 --use_feature_suppression $BASE_ARGS \
  --wandb_run_name "eval-kitti-urw-hires"

# -------------------------------------------------------
# 2. KITTI CLEAN + TTA — tta_evaluate.py
# -------------------------------------------------------
log "=== KITTI CLEAN (TTA) ==="

log "uncertainty + TTA"
eval $PYTHON tta_evaluate.py --load_weights_folder $UNCERT \
  --height 192 --width 640 $BASE_ARGS \
  --tta_steps $TTA_STEPS --tta_lr $TTA_LR \
  --wandb_run_name "eval-kitti-tta-uncertainty"

log "featsupp + TTA"
eval $PYTHON tta_evaluate.py --load_weights_folder $FEATSUPP \
  --height 192 --width 640 --use_feature_suppression $BASE_ARGS \
  --tta_steps $TTA_STEPS --tta_lr $TTA_LR \
  --wandb_run_name "eval-kitti-tta-featsupp"

log "urw-s2 + TTA"
eval $PYTHON tta_evaluate.py --load_weights_folder $URW_S2 \
  --height 192 --width 640 --use_feature_suppression $BASE_ARGS \
  --tta_steps $TTA_STEPS --tta_lr $TTA_LR \
  --wandb_run_name "eval-kitti-tta-urw-s2"

log "urw-hires + TTA"
eval $PYTHON tta_evaluate.py --load_weights_folder $URW_HIRES \
  --height 384 --width 1280 --use_feature_suppression $BASE_ARGS \
  --tta_steps $TTA_STEPS --tta_lr $TTA_LR \
  --wandb_run_name "eval-kitti-tta-urw-hires"

# -------------------------------------------------------
# 3. KITTI-C — evaluate_kitti_c.py (no TTA, prea lent)
# -------------------------------------------------------
log "=== KITTI-C (no TTA) ==="

log "kitti-c b6"
eval $PYTHON evaluate_kitti_c.py --load_weights_folder $B6 \
  --kitti_c_path $KITTI_C_PATH --corruptions all --eval_mono \
  --height 192 --width 640 \
  $WANDB --wandb_run_name "eval-kitti-c-b6"

log "kitti-c featsupp"
eval $PYTHON evaluate_kitti_c.py --load_weights_folder $FEATSUPP \
  --kitti_c_path $KITTI_C_PATH --corruptions all --eval_mono \
  --height 192 --width 640 --use_feature_suppression \
  $WANDB --wandb_run_name "eval-kitti-c-featsupp"

log "kitti-c urw-s2"
eval $PYTHON evaluate_kitti_c.py --load_weights_folder $URW_S2 \
  --kitti_c_path $KITTI_C_PATH --corruptions all --eval_mono \
  --height 192 --width 640 --use_feature_suppression \
  $WANDB --wandb_run_name "eval-kitti-c-urw-s2"

log "kitti-c urw-hires"
eval $PYTHON evaluate_kitti_c.py --load_weights_folder $URW_HIRES \
  --kitti_c_path $KITTI_C_PATH --corruptions all --eval_mono \
  --height 384 --width 1280 --use_feature_suppression \
  $WANDB --wandb_run_name "eval-kitti-c-urw-hires"

# -------------------------------------------------------
# 4. WEATHER (fog / rain / snow) — fara TTA
# -------------------------------------------------------
log "=== WEATHER no TTA ==="

for WEATHER in fog rain snow; do
  for MODEL_NAME in b6 uncertainty featsupp urw-s2 urw-hires; do
    case $MODEL_NAME in
      b6)        WF="$B6";       HH=192; WW=640; FS="" ;;
      uncertainty) WF="$UNCERT"; HH=192; WW=640; FS="" ;;
      featsupp)  WF="$FEATSUPP"; HH=192; WW=640; FS="--use_feature_suppression" ;;
      urw-s2)    WF="$URW_S2";   HH=192; WW=640; FS="--use_feature_suppression" ;;
      urw-hires) WF="$URW_HIRES"; HH=384; WW=1280; FS="--use_feature_suppression" ;;
    esac
    log "weather $WEATHER $MODEL_NAME"
    eval $PYTHON evaluate_weather.py --load_weights_folder $WF \
      --height $HH --width $WW $FS $BASE_ARGS \
      --weather_type $WEATHER --weather_severity moderate \
      --wandb_run_name "eval-${WEATHER}-${MODEL_NAME}"
  done
done

# -------------------------------------------------------
# 5. WEATHER + TTA (fog / rain / snow)
# -------------------------------------------------------
log "=== WEATHER + TTA ==="

for WEATHER in fog rain snow; do
  for MODEL_NAME in uncertainty featsupp urw-s2 urw-hires; do
    case $MODEL_NAME in
      uncertainty) WF="$UNCERT"; HH=192; WW=640; FS="" ;;
      featsupp)  WF="$FEATSUPP"; HH=192; WW=640; FS="--use_feature_suppression" ;;
      urw-s2)    WF="$URW_S2";   HH=192; WW=640; FS="--use_feature_suppression" ;;
      urw-hires) WF="$URW_HIRES"; HH=384; WW=1280; FS="--use_feature_suppression" ;;
    esac
    log "weather $WEATHER $MODEL_NAME TTA"
    eval $PYTHON evaluate_weather.py --load_weights_folder $WF \
      --height $HH --width $WW $FS $BASE_ARGS \
      --weather_type $WEATHER --weather_severity moderate \
      --use_tta --tta_steps $TTA_STEPS --tta_lr $TTA_LR \
      --wandb_run_name "eval-${WEATHER}-tta-${MODEL_NAME}"
  done
done

log "=== EVALUARE COMPLETA ==="
