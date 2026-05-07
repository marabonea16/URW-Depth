#!/bin/bash
# Antrenare URW-Depth cu Laplacian NLL pentru sigma calibrat
# Folosire:
#   Pornire de la zero:  bash train_laplacian.sh
#   Resume automat:      bash train_laplacian.sh --resume

SESSION="train_nll"
LOG_DIR="models"
MODEL_NAME="Tiny-Depth-Laplacian-NLL"
WEIGHTS_DIR="$LOG_DIR/$MODEL_NAME/models"

BASE_CMD="CUDA_VISIBLE_DEVICES=0 python train.py \
  --model_name $MODEL_NAME \
  --split eigen_zhou \
  --height 192 \
  --width 640 \
  --scales 0 \
  --png \
  --batch_size 4 \
  --learning_rate 1e-4 \
  --num_epochs 50 \
  --scheduler_step_size 24 \
  --log_dir $LOG_DIR \
  --data_path /home/ubuntu/TinyDepth \
  --num_workers 8 \
  --use_weather_aug \
  --use_feature_suppression \
  --laplacian_nll \
  --use_wandb \
  --wandb_project tinydepth \
  --wandb_run_name tinydepth-laplacian-nll"

# detecteaza daca trebuie resume
if [ "$1" == "--resume" ] && [ -L "$WEIGHTS_DIR/weights_latest" ]; then
    LATEST=$(readlink "$WEIGHTS_DIR/weights_latest")
    EPOCH=$(basename "$LATEST" | sed 's/weights_//')
    START_EPOCH=$((EPOCH + 1))
    echo "Resuming from epoch $START_EPOCH (checkpoint: weights_$EPOCH)"
    CMD="$BASE_CMD --load_weights_folder $LATEST --start_epoch $START_EPOCH"
else
    echo "Starting training from scratch"
    CMD="$BASE_CMD"
fi

# porneste sau reatazeaza sesiunea tmux
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Sesiunea tmux '$SESSION' exista deja. Reatasare..."
    tmux attach -t "$SESSION"
else
    echo "Creez sesiunea tmux '$SESSION'..."
    tmux new-session -d -s "$SESSION" "bash -c '$CMD; echo TRAINING DONE - apasa Enter; read'"
    tmux attach -t "$SESSION"
fi
