#!/bin/bash
# Second training run of the T1WI 3D full-resolution nnU-Net, to estimate how
# far median Dice moves when only the random initialisation and data order
# change. Identical plans, trainer, split and epochs as the original; only the
# results directory differs, so the original model is never touched.
#
# Resumable: re-running continues from the latest checkpoint.
#
#     setsid nohup bash scripts/run_seed2_3d.sh > logs_seed2.txt 2>&1 &

set -u
cd "$(dirname "$0")/.."
source nnunet/env.sh
export nnUNet_results="$PWD/nnunet/nnUNet_results_seed2"
mkdir -p "$nnUNet_results"

GPU="${NPC_GPU:-1}"
MIN_FREE_MB="${MIN_FREE_MB:-20000}"
export CUDA_VISIBLE_DEVICES="$GPU"

free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
if [ "${free:-0}" -lt "$MIN_FREE_MB" ]; then
  echo "$(date '+%F %T')  GPU $GPU has ${free} MiB free, need ${MIN_FREE_MB}; not starting"
  exit 1
fi

TR=nnUNetTrainer_250epochs
FOLD_DIR="$nnUNet_results/Dataset501_NPCT1WI/${TR}__nnUNetPlans__3d_fullres/fold_0"

if [ ! -f "$FOLD_DIR/checkpoint_final.pth" ]; then
  cont=""
  [ -f "$FOLD_DIR/checkpoint_latest.pth" ] && cont="--c"
  echo "$(date '+%F %T')  training on GPU $GPU ${cont:+(continuing)}"
  nnUNetv2_train 501 3d_fullres 0 -tr "$TR" $cont || { echo "training failed"; exit 1; }
fi

PRED="$PWD/nnunet/multiref/pred_T1WI_3d_fullres_seed2"
mkdir -p "$PRED"
if ! ls "$PRED"/*.nii.gz >/dev/null 2>&1; then
  echo "$(date '+%F %T')  predicting held-out patients"
  nnUNetv2_predict -i "$PWD/nnunet/multiref/in_T1WI" -o "$PRED" -d 501 \
    -c 3d_fullres -f 0 -tr "$TR" -device cuda || { echo "prediction failed"; exit 1; }
fi
echo "$(date '+%F %T')  done: $(ls "$PRED"/*.nii.gz | wc -l) predictions"
