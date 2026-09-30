#!/usr/bin/env python3
"""H4 and H5: model uncertainty against measured reader disagreement.

Runs after nnU-Net training. Two questions, both of which need a model:

    H4  Does the model's uncertainty land where the two radiologists actually
        disagree? Prior work (Jungo 2018) predicts it will not, because the 247
        non-reliability patients ship a single mask, so the training signal
        contains no record of observer disagreement.

    H5  Can a referral threshold be anchored to measured human disagreement
        rather than to an arbitrary Dice value? The reader agreement measured in
        this study (median Dice 0.891 to 0.904 by sequence) supplies that anchor:
        a case the model segments worse than two humans agree with each other is
        a case worth sending to a human.

Leakage: every patient used here is from the held-out test partition and was
excluded from nnU-Net's training and validation folds by splits_final.json.
The script re-verifies that rather than trusting it.

Uncertainty source: nnU-Net's mirroring test-time augmentation, exposed by
predicting with and without TTA and taking the per-voxel disagreement, plus the
predicted-probability entropy. A deep ensemble would be stronger but costs one
full training run per member.

Usage:
    python3 scripts/12_uncertainty_analysis.py --sequence T1WI
    python3 scripts/12_uncertainty_analysis.py --sequence all
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import load_config  # noqa: E402

DS = {"T1WI": ("501", "Dataset501_NPCT1WI", "ROI-T1.nii"),
      "T2WI": ("502", "Dataset502_NPCT2WI", "ROI-T2.nii"),
      "CE-T1WI": ("503", "Dataset503_NPCCET1WI", "ROI-CE-T1.nii")}
TRAINER = "nnUNetTrainer_250epochs"
CONFIG = "3d_fullres"

# Reader agreement measured in this study (scripts/05_reliability_probe.py).
READER_DICE = {"T1WI": 0.8907, "T2WI": 0.9035, "CE-T1WI": 0.9006}


def dice(a: np.ndarray, b: np.ndarray) -> float:
    tot = a.sum() + b.sum()
    return float(2 * np.logical_and(a, b).sum() / tot) if tot else float("nan")


def verify_unseen(nnunet_dir: Path, ds_name: str, test_ids: set[str]) -> None:
    """Abort if any test patient appears in nnU-Net's train or val fold."""
    sp = json.loads((nnunet_dir / "nnUNet_preprocessed" / ds_name / "splits_final.json").read_text())
    seen = {x.replace("NPC_", "") for x in sp[0]["train"]} | {x.replace("NPC_", "") for x in sp[0]["val"]}
    overlap = test_ids & seen
    if overlap:
        raise SystemExit(
            f"LEAKAGE: {len(overlap)} test patients were seen by nnU-Net during "
            f"training: {sorted(overlap)[:10]}. Refusing to report uncertainty results."
        )
    print(f"  guard passed: none of the {len(test_ids)} test patients was seen in training")


def run_inference(cfg, sequence: str, out_dir: Path, tta: bool) -> Path:
    """Predict the held-out patients, with and without test-time augmentation."""
    ds_id, ds_name, _ = DS[sequence]
    nnunet_dir = cfg.repo_root / "nnunet"
    in_dir = out_dir / f"input_{sequence}"
    pred_dir = out_dir / f"pred_{sequence}" / ("tta" if tta else "notta")
    pred_dir.mkdir(parents=True, exist_ok=True)
    if any(pred_dir.glob("*.nii.gz")):
        print(f"  predictions already present in {pred_dir.name}, reusing")
        return pred_dir

    env = dict(os.environ)
    env.update({
        "nnUNet_raw": str(nnunet_dir / "nnUNet_raw"),
        "nnUNet_preprocessed": str(nnunet_dir / "nnUNet_preprocessed"),
        "nnUNet_results": str(nnunet_dir / "nnUNet_results"),
        "CUDA_VISIBLE_DEVICES": "0",
    })
    cmd = ["nnUNetv2_predict", "-i", str(in_dir), "-o", str(pred_dir),
           "-d", ds_id, "-c", CONFIG, "-f", "0", "-tr", TRAINER,
           "--save_probabilities"]
    if not tta:
        cmd.append("--disable_tta")
    print(f"  predicting ({'TTA' if tta else 'no TTA'}) ...")
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-1500:]); print(r.stderr[-1500:])
        raise SystemExit(f"nnUNetv2_predict failed for {sequence}")
    return pred_dir


def stage_inputs(cfg, sequence: str, test_ids: set[str], out_dir: Path) -> set[str]:
    """Copy held-out images into an inference input folder."""
    import shutil

    _, ds_name, _ = DS[sequence]
    src = cfg.repo_root / "nnunet" / "nnUNet_raw" / ds_name / "imagesTr"
    dst = out_dir / f"input_{sequence}"
    dst.mkdir(parents=True, exist_ok=True)
    staged = set()
    for pid in sorted(test_ids):
        f = src / f"NPC_{pid}_0000.nii.gz"
        if f.exists():
            if not (dst / f.name).exists():
                shutil.copy2(f, dst / f.name)
            staged.add(pid)
    print(f"  staged {len(staged)} held-out volumes for inference")
    return staged


def analyse(cfg, sequence: str, out_dir: Path) -> dict:
    import nibabel as nib

    ds_id, ds_name, roi_stem = DS[sequence]
    nnunet_dir = cfg.repo_root / "nnunet"

    splits = pd.read_csv(cfg.out("splits_dir") / "patient_splits.csv", dtype={"patient_id": str})
    test_ids = set(splits.query('split=="test"').patient_id)
    manifest = pd.read_csv(cfg.out("data_summary_csv"), dtype={"patient_id": str})
    rel_ids = set(manifest.query("in_reliability_subset").patient_id) & test_ids

    print(f"\n=== {sequence} ===")
    verify_unseen(nnunet_dir, ds_name, test_ids)
    staged = stage_inputs(cfg, sequence, test_ids, out_dir)
    pred_tta = run_inference(cfg, sequence, out_dir, tta=True)
    pred_raw = run_inference(cfg, sequence, out_dir, tta=False)

    seg_root = cfg.primary_data_root / "MRI-Segments"
    rows = []
    for pid in sorted(staged):
        pt = pred_tta / f"NPC_{pid}.nii.gz"
        pr = pred_raw / f"NPC_{pid}.nii.gz"
        gt = cfg.repo_root / "nnunet" / "nnUNet_raw" / ds_name / "labelsTr" / f"NPC_{pid}.nii.gz"
        if not (pt.exists() and pr.exists() and gt.exists()):
            continue
        a = np.asanyarray(nib.load(str(pt)).dataobj) > 0
        b = np.asanyarray(nib.load(str(pr)).dataobj) > 0
        g = np.asanyarray(nib.load(str(gt)).dataobj) > 0

        # TTA disagreement as the uncertainty map: voxels where the two
        # inference passes differ are the ones the model is unsure about.
        unc = np.logical_xor(a, b)
        band = np.logical_or(a, b) & ~np.logical_and(a, b)

        row = {"patient_id": pid, "sequence": sequence,
               "dice_model_vs_reference": dice(a, g),
               "n_uncertain_voxels": int(unc.sum()),
               "uncertain_fraction_of_pred": float(unc.sum() / max(1, a.sum())),
               "pred_voxels": int(a.sum()), "ref_voxels": int(g.sum()),
               "in_reliability_subset": pid in rel_ids}

        # H4: where two readers exist, compare the model's uncertain band with
        # the band the two readers disagree on.
        if pid in rel_ids:
            r1 = seg_root / pid / roi_stem
            r2 = cfg.reliability_root / pid / roi_stem
            if r1.exists() and r2.exists():
                m1 = np.asanyarray(nib.load(str(r1)).dataobj) > 0
                m2 = np.asanyarray(nib.load(str(r2)).dataobj) > 0
                if m1.shape == a.shape:
                    reader_band = np.logical_xor(m1, m2)
                    row["reader_disagreement_voxels"] = int(reader_band.sum())
                    row["reader_dice"] = dice(m1, m2)
                    row["uncertainty_overlap_reader_band"] = dice(band, reader_band)
                    inter = np.logical_and(band, reader_band).sum()
                    row["frac_model_uncertainty_in_reader_band"] = float(inter / max(1, band.sum()))
                    row["frac_reader_band_covered"] = float(inter / max(1, reader_band.sum()))
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        print("  no predictions to analyse")
        return {}

    print(f"\n  held-out patients scored: {len(df)}")
    print(f"  model Dice vs reference: median {df.dice_model_vs_reference.median():.4f} "
          f"(IQR {df.dice_model_vs_reference.quantile(.25):.4f}-{df.dice_model_vs_reference.quantile(.75):.4f})")

    # ---- H5: referral anchored to measured human agreement -------------
    anchor = READER_DICE[sequence]
    below = df.dice_model_vs_reference < anchor
    print(f"\n  H5 referral, anchored at the measured reader Dice of {anchor:.4f}:")
    print(f"    cases the model segments worse than two readers agree: "
          f"{int(below.sum())}/{len(df)} ({100 * below.mean():.1f}%)")
    # Risk-coverage: sort by uncertainty, refer the most uncertain first.
    d = df.sort_values("uncertain_fraction_of_pred", ascending=False).reset_index(drop=True)
    cov, risk = [], []
    for k in range(len(d)):
        kept = d.iloc[k:]
        cov.append(len(kept) / len(d))
        risk.append(float(kept.dice_model_vs_reference.mean()))
    rc = pd.DataFrame({"coverage": cov, "mean_dice_of_retained": risk})
    for target in (1.0, 0.9, 0.8, 0.7, 0.5):
        i = int(np.argmin(np.abs(rc.coverage - target)))
        print(f"    refer {100 * (1 - rc.coverage[i]):4.0f}% most uncertain -> "
              f"retained mean Dice {rc.mean_dice_of_retained[i]:.4f}")

    # ---- H4: uncertainty against reader disagreement --------------------
    rel = df[df.in_reliability_subset & df.get("reader_disagreement_voxels", pd.Series(dtype=float)).notna()] \
        if "reader_disagreement_voxels" in df.columns else pd.DataFrame()
    h4 = {}
    if len(rel) >= 5:
        from scipy import stats

        r_s, p_s = stats.spearmanr(rel.n_uncertain_voxels, rel.reader_disagreement_voxels)
        print(f"\n  H4 on {len(rel)} two-reader held-out patients:")
        print(f"    model uncertain voxels vs reader-disagreement voxels: "
              f"Spearman rho={r_s:.3f}, p={p_s:.3g}")
        print(f"    median overlap (Dice) of the two bands: "
              f"{rel.uncertainty_overlap_reader_band.median():.4f}")
        print(f"    median fraction of the model's uncertain band inside the reader band: "
              f"{rel.frac_model_uncertainty_in_reader_band.median():.3f}")
        print(f"    median fraction of the reader band the model flags: "
              f"{rel.frac_reader_band_covered.median():.3f}")
        h4 = {"n": int(len(rel)), "spearman_rho": float(r_s), "p": float(p_s),
              "median_band_overlap": float(rel.uncertainty_overlap_reader_band.median()),
              "median_frac_unc_in_reader_band": float(rel.frac_model_uncertainty_in_reader_band.median()),
              "median_frac_reader_band_covered": float(rel.frac_reader_band_covered.median())}
    else:
        print("\n  H4: too few two-reader patients with usable geometry")

    tables = cfg.out("tables_dir")
    df.to_csv(tables / f"uncertainty_{sequence}.csv", index=False)
    rc.to_csv(tables / f"risk_coverage_{sequence}.csv", index=False)
    return {"sequence": sequence, "n_test": int(len(df)),
            "dice_median": float(df.dice_model_vs_reference.median()),
            "reader_anchor": anchor,
            "pct_below_reader_anchor": float(100 * below.mean()), "H4": h4}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--allow-missing-roots", action="store_true")
    ap.add_argument("--sequence", default="all")
    args = ap.parse_args()

    cfg = load_config(args.config, verify=True, allow_missing=True)
    out_dir = cfg.repo_root / "nnunet" / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)

    seqs = list(DS) if args.sequence == "all" else [args.sequence]
    results = []
    for s in seqs:
        ckpt = (cfg.repo_root / "nnunet" / "nnUNet_results" / DS[s][1] /
                f"{TRAINER}__nnUNetPlans__{CONFIG}" / "fold_0" / "checkpoint_final.pth")
        if not ckpt.exists():
            print(f"\n=== {s} ===\n  no finished checkpoint yet ({ckpt.name} missing); skipping")
            continue
        results.append(analyse(cfg, s, out_dir))

    if results:
        with open(cfg.out("tables_dir") / "uncertainty_summary.json", "w", encoding="utf-8") as fh:
            json.dump([r for r in results if r], fh, indent=2)
        print(f"\nWrote {cfg.out('tables_dir')}/uncertainty_summary.json")
    else:
        print("\nNothing analysed. Training must finish first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
