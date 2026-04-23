#!/bin/bash
# TTA evaluation script
# Folosire:
#   bash tta_evaluate.sh           -> weights_latest
#   bash tta_evaluate.sh 49        -> weights_49
#   bash tta_evaluate.sh 49 10 5e-5 -> weights_49, 10 pasi TTA, lr=5e-5

MODEL_NAME="Tiny-Depth-Weather-Robust-Feature-Supression"
LOG_DIR="models"
WEIGHTS_DIR="$LOG_DIR/$MODEL_NAME/models"

TTA_STEPS=${2:-3}
TTA_LR=${3:-1e-5}

if [ -n "$1" ]; then
    WEIGHTS="$WEIGHTS_DIR/weights_$1"
    if [ ! -d "$WEIGHTS" ]; then
        echo "Eroare: nu exista checkpoint weights_$1 in $WEIGHTS_DIR"
        exit 1
    fi
    echo "-> TTA evaluare checkpoint: weights_$1"
else
    if [ ! -L "$WEIGHTS_DIR/weights_latest" ]; then
        echo "Eroare: nu exista weights_latest in $WEIGHTS_DIR"
        exit 1
    fi
    WEIGHTS=$(readlink "$WEIGHTS_DIR/weights_latest")
    echo "-> TTA evaluare checkpoint: $(basename $WEIGHTS)"
fi

CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/TinyDepth/tta_evaluate.py \
  --load_weights_folder "$WEIGHTS" \
  --eval_mono \
  --height 192 \
  --width 640 \
  --scales 0 \
  --data_path /home/ubuntu/TinyDepth \
  --png \
  --eval_split eigen \
  --tta_steps $TTA_STEPS \
  --tta_lr $TTA_LR \
  --use_feature_suppression \
  --use_wandb \
  --wandb_project tinydepth \
  --wandb_run_name "tta-$MODEL_NAME-steps${TTA_STEPS}-$(basename $WEIGHTS)"
