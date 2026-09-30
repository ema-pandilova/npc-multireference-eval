#!/usr/bin/env bash
# Train the 2D nnU-Net configuration on T1WI (Dataset501), used as a scale reference.
set -u
source "$(dirname "$0")/../nnunet/env.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
nnUNetv2_train 501 2d 0 -tr nnUNetTrainer_250epochs --npz
