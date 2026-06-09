#!/bin/bash
# Two-stage training pentru URW-Depth:
#   Stage 1: 50 epoci antrenare clean (fara weather aug) -> maximizeaza clean performance
#   Stage 2: 15 epoci fine-tune cu weather aug (LR=1e-5) -> adauga robustete
#
# Folosire:
#   De la zero:    bash train_twostage.sh
#   Resume S1:     bash train_twostage.sh --resume-s1
#   Resume S2:     bash train_twostage.sh --resume-s2

SESSION="train_ts"
LOG_DIR="models"
S1_NAME="URW-Depth-S1"
S2_NAME="URW-Depth-S2"
S1_DIR="$LOG_DIR/$S1_NAME/models"
S2_DIR="$LOG_DIR/$S2_NAME/models"
WORKDIR="/home/ubuntu/TinyDepth"

BASE_ARGS="--split eigen_zhou --height 192 --width 640 --scales 0 --png \
  --batch_size 4 --scheduler_step_size 24 \
  --log_dir $LOG_DIR --data_path $WORKDIR --num_workers 8 \
  --use_feature_suppression --use_wandb --wandb_project tinydepth"

S1_CMD="python train.py --model_name $S1_NAME $BASE_ARGS \
  --learning_rate 1e-4 --num_epochs 50 \
  --wandb_run_name urw-depth-s1-clean"

S2_CMD="python train.py --model_name $S2_NAME $BASE_ARGS \
  --learning_rate 1e-5 --num_epochs 15 --scheduler_step_size 8 \
  --use_weather_aug \
  --wandb_run_name urw-depth-s2-weather \
  --load_weights_folder $WORKDIR/$S1_DIR/weights_49 --start_epoch 0"

# --- logica resume ---
if [ "$1" == "--resume-s1" ] && [ -L "$S1_DIR/weights_latest" ]; then
    LATEST=$(readlink "$S1_DIR/weights_latest")
    EPOCH=$(basename "$LATEST" | sed 's/weights_//')
    START=$((EPOCH + 1))
    echo "Resuming Stage 1 from epoch $START"
    S1_CMD="$S1_CMD --load_weights_folder $WORKDIR/$LATEST --start_epoch $START"
    SKIP_S1=false
elif [ "$1" == "--resume-s2" ] && [ -L "$S2_DIR/weights_latest" ]; then
    LATEST=$(readlink "$S2_DIR/weights_latest")
    EPOCH=$(basename "$LATEST" | sed 's/weights_//')
    START=$((EPOCH + 1))
    echo "Resuming Stage 2 from epoch $START"
    S2_CMD="python train.py --model_name $S2_NAME $BASE_ARGS \
      --learning_rate 1e-5 --num_epochs 15 --scheduler_step_size 8 \
      --use_weather_aug \
      --wandb_run_name urw-depth-s2-weather \
      --load_weights_folder $WORKDIR/$LATEST --start_epoch $START"
    SKIP_S1=true
else
    echo "Starting two-stage training from scratch"
    SKIP_S1=false
fi

FULL_CMD="source /home/ubuntu/anaconda3/etc/profile.d/conda.sh && conda activate tinydepth && cd $WORKDIR"

if [ "$SKIP_S1" = false ]; then
    FULL_CMD="$FULL_CMD && echo '=== STAGE 1: Clean training ===' && CUDA_VISIBLE_DEVICES=0 $S1_CMD"
fi

FULL_CMD="$FULL_CMD && echo '=== STAGE 2: Weather fine-tuning ===' && CUDA_VISIBLE_DEVICES=0 $S2_CMD && echo 'TWO-STAGE TRAINING COMPLETE'"

# porneste in tmux cu inhibitor de shutdown
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session $SESSION exista deja. Reatasare..."
    tmux attach -t "$SESSION"
else
    tmux new-session -d -s "$SESSION" "bash -c 'systemd-inhibit --what=shutdown --who=URW-Depth --why=training --mode=block bash -c \"$FULL_CMD\"; echo DONE - apasa Enter; read'"
    echo "Pornit in sesiunea tmux '$SESSION'"
    tmux attach -t "$SESSION"
fi
