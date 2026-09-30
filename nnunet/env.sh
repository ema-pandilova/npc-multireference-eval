# Source before any nnU-Net command.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export nnUNet_raw="$ROOT/nnUNet_raw"
export nnUNet_preprocessed="$ROOT/nnUNet_preprocessed"
export nnUNet_results="$ROOT/nnUNet_results"
export nnUNet_n_proc_DA=8
