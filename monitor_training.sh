#!/bin/bash
# Monitorizeaza antrenarea Tiny-Depth-Laplacian-NLL si o reia daca s-a oprit.
# Rulat de cron la fiecare ora.

set -euo pipefail

WORKDIR="/home/ubuntu/TinyDepth"
LOG="/home/ubuntu/monitor_training.log"
HC_URL="https://hc-ping.com/f2d24240-e53d-4315-bf2d-c9bae9778766"

cd "$WORKDIR"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

# Ping healthchecks.io so it knows the VM is alive
wget -qO- "$HC_URL" > /dev/null 2>&1 || true

# -------------------------------------------------------
# Two-stage (URW-Depth-S1 / S2) - prioritate maxima
# -------------------------------------------------------

# -------------------------------------------------------
# HiRes two-stage (URW-Depth-HiRes-S1 / S2) - prioritate maxima
# -------------------------------------------------------

HRS1_DIR="$WORKDIR/models/URW-Depth-HiRes-S1/models"
HRS2_DIR="$WORKDIR/models/URW-Depth-HiRes-S2/models"
SESSION_TS="train_ts"

HIRES_BASE="--split eigen_zhou --height 384 --width 1280 --scales 0 --png \
  --batch_size 1 --accum_steps 2 --scheduler_step_size 24 \
  --log_dir models --data_path $WORKDIR --num_workers 8 \
  --use_feature_suppression --use_wandb --wandb_project tinydepth \
  --use_amp --use_activation_checkpoint"

# Daca HiRes-S2 e complet, gata
if [ -d "$HRS2_DIR/weights_14" ]; then
    echo "$(timestamp) HiRes two-stage complete. Nothing to do." >> "$LOG"
    exit 0
fi

# Daca ruleaza deja HiRes, lasa-l
if pgrep -f "train.py.*(URW-Depth-HiRes)" > /dev/null 2>&1; then
    echo "$(timestamp) HiRes training running. OK." >> "$LOG"
    exit 0
fi

# Daca HiRes-S1 e complet (30 epochs), reia/porneste HiRes-S2
if [ -d "$HRS1_DIR/weights_29" ]; then
    if [ -L "$HRS2_DIR/weights_latest" ]; then
        LATEST=$(readlink -f "$HRS2_DIR/weights_latest")
        EPOCH=$(basename "$LATEST" | sed 's/weights_//')
        START=$((EPOCH + 1))
        echo "$(timestamp) Resuming HiRes-S2 from epoch $START" >> "$LOG"
        CMD="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 python train.py --model_name URW-Depth-HiRes-S2 $HIRES_BASE \
          --learning_rate 5e-6 --num_epochs 15 --scheduler_step_size 8 \
          --use_weather_aug --use_corruption_aug --wandb_run_name urw-depth-hires-s2-corruption \
          --load_weights_folder $LATEST --start_epoch $START"
    else
        echo "$(timestamp) Starting HiRes-S2 from HiRes-S1/weights_29" >> "$LOG"
        CMD="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 python train.py --model_name URW-Depth-HiRes-S2 $HIRES_BASE \
          --learning_rate 5e-6 --num_epochs 15 --scheduler_step_size 8 \
          --use_weather_aug --use_corruption_aug --wandb_run_name urw-depth-hires-s2-corruption \
          --load_weights_folder $HRS1_DIR/weights_29 --start_epoch 0"
    fi
    tmux kill-session -t "$SESSION_TS" 2>/dev/null || true
    tmux new-session -d -s "$SESSION_TS" "bash -c 'cd $WORKDIR && source /home/ubuntu/anaconda3/etc/profile.d/conda.sh && conda activate tinydepth && systemd-inhibit --what=shutdown --who=URW-Depth --why=training --mode=block bash -c \"$CMD\"; echo DONE; read'"
    echo "$(timestamp) Launched HiRes-S2 in tmux '$SESSION_TS'." >> "$LOG"
    exit 0
fi

# HiRes-S1 nu e complet - reia/porneste
if [ -L "$HRS1_DIR/weights_latest" ]; then
    LATEST=$(readlink -f "$HRS1_DIR/weights_latest")
    EPOCH=$(basename "$LATEST" | sed 's/weights_//')
    START=$((EPOCH + 1))
    echo "$(timestamp) Resuming HiRes-S1 from epoch $START" >> "$LOG"
    S1_EXTRA="--load_weights_folder $LATEST --start_epoch $START"
else
    echo "$(timestamp) Starting HiRes-S1 from scratch" >> "$LOG"
    S1_EXTRA=""
fi

HRS1_CMD="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 python train.py --model_name URW-Depth-HiRes-S1 $HIRES_BASE \
  --learning_rate 5e-5 --num_epochs 30 --wandb_run_name urw-depth-hires-s1-clean $S1_EXTRA"
HRS2_CMD="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 python train.py --model_name URW-Depth-HiRes-S2 $HIRES_BASE \
  --learning_rate 5e-6 --num_epochs 15 --scheduler_step_size 8 \
  --use_weather_aug --use_corruption_aug --wandb_run_name urw-depth-hires-s2-corruption \
  --load_weights_folder $HRS1_DIR/weights_29 --start_epoch 0"

tmux kill-session -t "$SESSION_TS" 2>/dev/null || true
tmux new-session -d -s "$SESSION_TS" "bash -c 'cd $WORKDIR && source /home/ubuntu/anaconda3/etc/profile.d/conda.sh && conda activate tinydepth && systemd-inhibit --what=shutdown --who=URW-Depth --why=training --mode=block bash -c \"$HRS1_CMD && $HRS2_CMD\"; echo DONE; read'"
echo "$(timestamp) Launched HiRes two-stage in tmux '$SESSION_TS'." >> "$LOG"
