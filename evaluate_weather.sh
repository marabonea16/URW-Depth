#!/bin/bash
# Evaluare completa pe vreme adversa: fog / rain / snow
# cu si fara TTA, pentru modelul specificat.
#
# Folosire:
#   bash evaluate_weather.sh                    -> weights_latest, toate tipurile
#   bash evaluate_weather.sh 49                 -> weights_49
#   bash evaluate_weather.sh 49 fog             -> doar fog
#   bash evaluate_weather.sh 49 all nottta      -> fara TTA

MODEL_NAME="Tiny-Depth-Weather-Robust-Feature-Supression"
LOG_DIR="models"
WEIGHTS_DIR="$LOG_DIR/$MODEL_NAME/models"

EPOCH=${1:-"latest"}
WEATHER=${2:-"all"}     # fog | rain | snow | all
TTA_MODE=${3:-"both"}   # tta | nottta | both

if [ "$EPOCH" == "latest" ]; then
    if [ ! -L "$WEIGHTS_DIR/weights_latest" ]; then
        echo "Eroare: nu exista weights_latest"
        exit 1
    fi
    WEIGHTS=$(readlink "$WEIGHTS_DIR/weights_latest")
else
    WEIGHTS="$WEIGHTS_DIR/weights_$EPOCH"
    if [ ! -d "$WEIGHTS" ]; then
        echo "Eroare: nu exista weights_$EPOCH"
        exit 1
    fi
fi

echo "-> Model: $MODEL_NAME"
echo "-> Weights: $(basename $WEIGHTS)"
echo "-> Weather: $WEATHER | TTA: $TTA_MODE"
echo ""

BASE_CMD="CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/TinyDepth/evaluate_weather.py \
  --load_weights_folder $WEIGHTS \
  --eval_mono --height 192 --width 640 --scales 0 \
  --data_path /home/ubuntu/TinyDepth --png --eval_split eigen \
  --use_feature_suppression \
  --weather_severity moderate \
  --use_wandb --wandb_project tinydepth"

# determina ce tipuri de vreme de rulat
if [ "$WEATHER" == "all" ]; then
    WEATHER_TYPES="fog rain snow"
else
    WEATHER_TYPES="$WEATHER"
fi

for W in $WEATHER_TYPES; do
    echo "=============================="
    echo "Weather: $W"
    echo "=============================="

    if [ "$TTA_MODE" == "nottta" ] || [ "$TTA_MODE" == "both" ]; then
        echo "-- Fara TTA --"
        eval "$BASE_CMD \
          --weather_type $W \
          --wandb_run_name $MODEL_NAME-$W-noTTA-$(basename $WEIGHTS)"
    fi

    if [ "$TTA_MODE" == "tta" ] || [ "$TTA_MODE" == "both" ]; then
        echo "-- Cu TTA --"
        eval "$BASE_CMD \
          --weather_type $W \
          --use_tta --tta_steps 3 --tta_lr 1e-5 \
          --wandb_run_name $MODEL_NAME-$W-TTA-$(basename $WEIGHTS)"
    fi

    echo ""
done

echo "-> Toate evaluarile terminate."
