# Multi-reference evaluation of nasopharyngeal carcinoma MRI segmentation

Analysis code for measuring how per-sequence reference standards (T1WI, T2WI,
CE-T1WI) differ in a public nasopharyngeal carcinoma MRI dataset, and how that
difference moves the reported score of a trained nnU-Net.

## Data

The dataset is not included. Download it from Zenodo, record 13131827
(CC BY 4.0), or fetch it with:

    python3 scripts/00_fetch_dataset.py --which all --dest ../dataset_v2

Dataset locations are set in `configs/paths.yaml`.

## Installation

    pip install -r requirements.txt
    pip install nnunetv2==2.1

## Running

Scripts are numbered and run in order:

- `01`–`08`: dataset audit, fixed split, and the registration-independent volume analyses
- `11`: builds the nnU-Net datasets
- `16`, `19`, `20`: multi-reference evaluation of the trained models
- `13`, `18`, `21`, `22`, `27`: registration, resampling and reader analyses
- `23`: collects all summary numbers into `results/tables/manuscript_numbers.json`

nnU-Net training:

    source nnunet/env.sh
    nnUNetv2_plan_and_preprocess -d 501 502 503 --verify_dataset_integrity
    # restore the fixed split so the held-out patients match the study
    for d in 501_NPCT1WI 502_NPCT2WI 503_NPCCET1WI; do
      cp data/splits/nnunet/Dataset${d}_splits_final.json nnunet/nnUNet_preprocessed/Dataset$d/splits_final.json
    done
    bash scripts/run_nnunet_training.sh 0

## Fixed split

`data/splits/patient_splits.csv` is the patient-level split used by every
analysis. All 30 double-annotated patients are held out.

## Licence

Code: MIT (see `LICENSE`). Dataset: CC BY 4.0, by its original authors.
