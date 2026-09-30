#!/usr/bin/env bash
# Train nnU-Net v2 on the three per-sequence NPC datasets.
#
# Config choices and why:
#   3d_fullres      nnU-Net planned anisotropic kernels for this data (5.5 mm
#                   through-plane vs 0.45 mm in-plane), which a 2d config throws away.
#   250 epochs      the default 1000 would cost days per fold. 250 is a documented
#                   nnU-Net variant and is ample for 275 cases.
#   fold 0 first    one fold per sequence gives a usable model fastest; further
#                   folds are appended afterwards to build the ensemble for the
#                   uncertainty analysis.
set -u
source "$(dirname "$0")/../nnunet/env.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TRAINER=nnUNetTrainer_250epochs
CONFIG=3d_fullres

# Wait for preprocessing to finish before touching the preprocessed folder.
while pgrep -f "nnUNetv2_plan_and_preprocess" > /dev/null; do sleep 60; done
echo "[$(date +%H:%M)] preprocessing finished, starting training"

for D in 501 502 503; do
  for FOLD in "$@"; do
    echo "=============================================================="
    echo "[$(date +%H:%M)] dataset $D  fold $FOLD  $CONFIG  $TRAINER"
    echo "=============================================================="
    nnUNetv2_train "$D" "$CONFIG" "$FOLD" -tr "$TRAINER" --npz
    echo "[$(date +%H:%M)] dataset $D fold $FOLD exit=$?"
  done
done
echo "[$(date +%H:%M)] ALL TRAINING DONE"
