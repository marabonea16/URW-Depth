#!/bin/bash
# Script de evaluare TinyDepth
# Folosire:
#   Ultimul checkpoint:  bash evaluate.sh
#   Checkpoint specific: bash evaluate.sh 42

MODEL_NAME="Tiny-Depth-Weather-Robust-Feature-Supression"
LOG_DIR="models"
WEIGHTS_DIR="$LOG_DIR/$MODEL_NAME/models"

# daca e dat un numar de epoca ca argument, foloseste acel checkpoint
if [ -n "$1" ]; then
    WEIGHTS="$WEIGHTS_DIR/weights_$1"
    if [ ! -d "$WEIGHTS" ]; then
        echo "Eroare: nu exista checkpoint weights_$1 in $WEIGHTS_DIR"
        exit 1
    fi
    echo "-> Evaluare checkpoint: weights_$1"
else
    # altfel foloseste weights_latest
    if [ ! -L "$WEIGHTS_DIR/weights_latest" ]; then
        echo "Eroare: nu exista weights_latest in $WEIGHTS_DIR"
        echo "Specifica un checkpoint: bash evaluate.sh <epoca>"
        exit 1
    fi
    WEIGHTS=$(readlink "$WEIGHTS_DIR/weights_latest")
    echo "-> Evaluare checkpoint: $(basename $WEIGHTS)"
fi

CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/TinyDepth/evaluate_depth.py \
  --load_weights_folder "$WEIGHTS" \
  --eval_mono \
  --height 192 \
  --width 640 \
  --scales 0 \
  --data_path /home/ubuntu/TinyDepth \
  --png \
  --eval_split eigen \
  --use_feature_suppression \
  --use_wandb \
  --wandb_project tinydepth \
  --wandb_run_name "$MODEL_NAME-$(basename $WEIGHTS)"
