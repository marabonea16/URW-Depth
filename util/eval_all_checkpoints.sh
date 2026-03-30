#!/usr/bin/env bash
set -euo pipefail

# Evaluate all weights_* checkpoints and print a metrics leaderboard.
# Usage:
#   bash util/eval_all_checkpoints.sh \
#     /home/ubuntu/TinyDepth/models/Tiny-Depth/models \
#     /home/ubuntu/TinyDepth \
#     --png
#
# Positional args:
#   1) checkpoint root dir (contains weights_*)
#   2) KITTI data_path
# Remaining args are forwarded to evaluate_depth.py.

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <checkpoint_root> <data_path> [extra evaluate_depth args]"
  exit 1
fi

CKPT_ROOT="$1"
DATA_PATH="$2"
shift 2
EXTRA_ARGS=("$@")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_SCRIPT="${REPO_ROOT}/evaluate_depth.py"
GT_FILE="${REPO_ROOT}/splits/eigen/gt_depths.npz"

if [[ ! -f "${EVAL_SCRIPT}" ]]; then
  echo "Cannot find evaluate_depth.py at ${EVAL_SCRIPT}"
  exit 1
fi

if [[ ! -d "${CKPT_ROOT}" ]]; then
  echo "Checkpoint root not found: ${CKPT_ROOT}"
  exit 1
fi

if [[ ! -f "${GT_FILE}" ]]; then
  echo "Missing ${GT_FILE}"
  echo "Run: python ${REPO_ROOT}/export_gt_depth.py --data_path ${DATA_PATH} --split eigen"
  exit 1
fi

mapfile -t CKPTS < <(find "${CKPT_ROOT}" -maxdepth 1 -type d -name "weights_*" | sort -V)

if [[ ${#CKPTS[@]} -eq 0 ]]; then
  echo "No weights_* folders found in ${CKPT_ROOT}"
  exit 1
fi

RESULTS_FILE="$(mktemp)"
trap 'rm -f "${RESULTS_FILE}"' EXIT

echo "Evaluating ${#CKPTS[@]} checkpoints..."
echo

for ckpt in "${CKPTS[@]}"; do
  ckpt_name="$(basename "${ckpt}")"
  log_file="$(mktemp)"
  echo "[${ckpt_name}] running eval..."
  if CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python "${EVAL_SCRIPT}" \
    --load_weights_folder "${ckpt}" \
    --eval_mono \
    --height 192 --width 640 --scales 0 \
    --data_path "${DATA_PATH}" \
    --eval_split eigen \
    --num_workers 8 \
    "${EXTRA_ARGS[@]}" >"${log_file}" 2>&1; then
    metrics_line="$(grep -E '^&' "${log_file}" | tail -n 1 || true)"
    if [[ -z "${metrics_line}" ]]; then
      echo "[${ckpt_name}] failed to parse metrics"
      echo "[${ckpt_name}] parse_error" >> "${RESULTS_FILE}"
    else
      # Split the LaTeX-style line "& 0.123 & ..."
      abs_rel="$(echo "${metrics_line}" | awk -F'&' '{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}')"
      sq_rel="$(echo "${metrics_line}" | awk -F'&' '{gsub(/^[ \t]+|[ \t]+$/, "", $3); print $3}')"
      rmse="$(echo "${metrics_line}" | awk -F'&' '{gsub(/^[ \t]+|[ \t]+$/, "", $4); print $4}')"
      rmse_log="$(echo "${metrics_line}" | awk -F'&' '{gsub(/^[ \t]+|[ \t]+$/, "", $5); print $5}')"
      a1="$(echo "${metrics_line}" | awk -F'&' '{gsub(/^[ \t]+|[ \t]+$/, "", $6); print $6}')"
      a2="$(echo "${metrics_line}" | awk -F'&' '{gsub(/^[ \t]+|[ \t]+$/, "", $7); print $7}')"
      a3="$(echo "${metrics_line}" | awk -F'&' '{gsub(/^[ \t]+|[ \t]+$/, "", $8); sub(/\\\\$/, "", $8); print $8}')"
      echo "${ckpt_name} ${abs_rel} ${sq_rel} ${rmse} ${rmse_log} ${a1} ${a2} ${a3}" >> "${RESULTS_FILE}"
      echo "[${ckpt_name}] abs_rel=${abs_rel} sq_rel=${sq_rel} rmse=${rmse} rmse_log=${rmse_log} a1=${a1}"
    fi
  else
    echo "[${ckpt_name}] eval_failed"
    echo "[${ckpt_name}] eval_failed" >> "${RESULTS_FILE}"
  fi
  rm -f "${log_file}"
done

echo
echo "Leaderboard (sorted by abs_rel ascending):"
printf "%-12s %-8s %-8s %-8s %-10s %-8s %-8s %-8s\n" \
  "checkpoint" "abs_rel" "sq_rel" "rmse" "rmse_log" "a1" "a2" "a3"
grep -E '^weights_[0-9]+' "${RESULTS_FILE}" | sort -g -k2 | while read -r c a b d e f g h; do
  printf "%-12s %-8s %-8s %-8s %-10s %-8s %-8s %-8s\n" "$c" "$a" "$b" "$d" "$e" "$f" "$g" "$h"
done
