#!/usr/bin/env python3
"""Score a trained model against every per-sequence reference, not just its own.

This is the experiment that turns the reference-standard confound from a property
of the annotations into a property of a reported result. A model trained and
predicting on one sequence produces exactly one segmentation per patient. That
single, fixed prediction is then scored three times: against its own sequence's
reference, and against the other two sequences' references brought into its space
by rigid within-session registration.

Nothing about the model changes between those three numbers. Any difference is
attributable to the choice of reference standard alone.

Reported per model and per reference:
    Dice, volume error, and the reference sensitivity of each patient, defined as
    max_r Dice_r  minus  min_r Dice_r. That spread is what a single reported
    figure hides.

All patients scored here are from the held-out partition and were excluded from
nnU-Net's training and validation folds. The script verifies that and aborts
rather than trusting it.

Usage:
    python3 scripts/16_multireference_evaluation.py --model T1WI
    python3 scripts/16_multireference_evaluation.py --model all
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
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
SEQ_ROI = {"T1WI": "ROI-T1.nii", "T2WI": "ROI-T2.nii", "CE-T1WI": "ROI-CE-T1.nii"}
TRAINER = "nnUNetTrainer_250epochs"


def dice(a, b):
    t = a.sum() + b.sum()
    return float(2 * np.logical_and(a, b).sum() / t) if t else float("nan")


def read_series(sitk, folder: Path):
    r = sitk.ImageSeriesReader()
    ids = r.GetGDCMSeriesIDs(str(folder))
    if not ids:
        return None
    best, n = None, -1
    for sid in ids:
        f = r.GetGDCMSeriesFileNames(str(folder), sid)
        if len(f) > n:
            best, n = sid, len(f)
    r.SetFileNames(r.GetGDCMSeriesFileNames(str(folder), best))
    try:
        return r.Execute()
    except Exception:  # noqa: BLE001
        return None


def register(sitk, fixed, moving):
    ff, mf = sitk.Cast(fixed, sitk.sitkFloat32), sitk.Cast(moving, sitk.sitkFloat32)
    init = sitk.CenteredTransformInitializer(
        ff, mf, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.10, seed=20260727)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsRegularStepGradientDescent(1.0, 1e-4, 200, gradientMagnitudeTolerance=1e-6)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2, 1, 0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    R.SetInitialTransform(init, inPlace=False)
    try:
        return R.Execute(ff, mf)
    except Exception:  # noqa: BLE001
        return None


def verify_unseen(nnunet: Path, ds_name: str, test_ids: set[str]) -> None:
    sp = json.loads((nnunet / "nnUNet_preprocessed" / ds_name / "splits_final.json").read_text())
    seen = {x.replace("NPC_", "") for x in sp[0]["train"]} | {x.replace("NPC_", "") for x in sp[0]["val"]}
    bad = test_ids & seen
    if bad:
        raise SystemExit(f"LEAKAGE: {len(bad)} evaluation patients were seen in training: "
                         f"{sorted(bad)[:8]}. Refusing to report.")
    print(f"    leakage guard passed ({len(test_ids)} held-out patients, none seen)")


def predict(cfg, model: str, test_ids: set[str], out_dir: Path,
            config_name: str = "3d_fullres", device: str = "cuda") -> Path:
    ds_id, ds_name, _ = DS[model]
    nnunet = cfg.repo_root / "nnunet"
    in_dir, pred_dir = out_dir / f"in_{model}", out_dir / f"pred_{model}_{config_name}"
    in_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    if any(pred_dir.glob("*.nii.gz")):
        print(f"    reusing {len(list(pred_dir.glob('*.nii.gz')))} existing predictions")
        return pred_dir

    src = nnunet / "nnUNet_raw" / ds_name / "imagesTr"
    n = 0
    for pid in sorted(test_ids):
        f = src / f"NPC_{pid}_0000.nii.gz"
        if f.exists():
            if not (in_dir / f.name).exists():
                shutil.copy2(f, in_dir / f.name)
            n += 1
    print(f"    staged {n} volumes")

    env = dict(os.environ)
    env.update({"nnUNet_raw": str(nnunet / "nnUNet_raw"),
                "nnUNet_preprocessed": str(nnunet / "nnUNet_preprocessed"),
                "nnUNet_results": str(nnunet / "nnUNet_results")})
    if device == "cpu":
        # Keep the GPU untouched, and cap threads so a long CPU run does not
        # monopolise a shared machine.
        env["CUDA_VISIBLE_DEVICES"] = ""
        env.setdefault("OMP_NUM_THREADS", "8")
        env.setdefault("MKL_NUM_THREADS", "8")
    else:
        env["CUDA_VISIBLE_DEVICES"] = os.environ.get("NPC_GPU", "0")
    cmd = ["nnUNetv2_predict", "-i", str(in_dir), "-o", str(pred_dir), "-d", ds_id,
           "-c", config_name, "-f", "0", "-tr", TRAINER, "-device", device]
    if device == "cpu":
        cmd += ["-npp", "2", "-nps", "2"]
    print("    predicting ...", flush=True)
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-1200:]); print(r.stderr[-1200:])
        raise SystemExit("nnUNetv2_predict failed")
    return pred_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--allow-missing-roots", action="store_true")
    ap.add_argument("--model", default="all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="inference device; cpu leaves the GPU untouched")
    ap.add_argument("--config-name", default="3d_fullres",
                    help="nnU-Net configuration to evaluate (3d_fullres or 2d)")
    args = ap.parse_args()

    import SimpleITK as sitk

    cfg = load_config(args.config, verify=True, allow_missing=True)
    nnunet = cfg.repo_root / "nnunet"
    seg_root = cfg.primary_data_root / "MRI-Segments"
    out_dir = nnunet / "multiref"
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = pd.read_csv(cfg.out("splits_dir") / "patient_splits.csv", dtype={"patient_id": str})
    test_ids = set(splits.query('split=="test"').patient_id)

    models = list(DS) if args.model == "all" else [args.model]
    rows = []

    for model in models:
        ds_id, ds_name, own_roi = DS[model]
        ckpt = (nnunet / "nnUNet_results" / ds_name /
                f"{TRAINER}__nnUNetPlans__{args.config_name}" / "fold_0" / "checkpoint_final.pth")
        print(f"\n=== model trained on {model} ===")
        if not ckpt.exists():
            print("    no finished checkpoint; skipping")
            continue
        verify_unseen(nnunet, ds_name, test_ids)
        pred_dir = predict(cfg, model, test_ids, out_dir, args.config_name, args.device)

        pids = sorted(p.stem.replace(".nii", "").replace("NPC_", "")
                      for p in pred_dir.glob("*.nii.gz"))
        if args.limit:
            pids = pids[: args.limit]
        print(f"    scoring {len(pids)} patients against all three references")

        for k, pid in enumerate(pids, 1):
            pf = pred_dir / f"NPC_{pid}.nii.gz"
            own_img_dir = seg_root / pid / model
            if not own_img_dir.is_dir():
                continue
            pred = sitk.ReadImage(str(pf))
            P = sitk.GetArrayFromImage(pred) > 0
            fixed = read_series(sitk, own_img_dir)
            if fixed is None:
                continue

            rec = {"model": f"{model}-{args.config_name}", "patient_id": pid,
                   "pred_voxels": int(P.sum())}
            for ref_seq, roi in SEQ_ROI.items():
                rp = seg_root / pid / roi
                if not rp.exists():
                    continue
                try:
                    ref = sitk.ReadImage(str(rp))
                except Exception:  # noqa: BLE001
                    continue
                if ref_seq == model:
                    if ref.GetSize() != pred.GetSize():
                        continue
                    R = sitk.GetArrayFromImage(ref) > 0
                else:
                    mv = read_series(sitk, seg_root / pid / ref_seq)
                    if mv is None or mv.GetSize() != ref.GetSize():
                        continue
                    tx = register(sitk, fixed, mv)
                    if tx is None:
                        continue
                    R = sitk.GetArrayFromImage(
                        sitk.Resample(ref, fixed, tx, sitk.sitkNearestNeighbor, 0,
                                      ref.GetPixelID())) > 0
                rec[f"dice_vs_{ref_seq}"] = round(dice(P, R), 4)
                rec[f"refvox_{ref_seq}"] = int(R.sum())
            rows.append(rec)
            if k % 20 == 0 or k == len(pids):
                print(f"      {k}/{len(pids)}", flush=True)

    if not rows:
        print("\nNothing scored.")
        return 1

    df = pd.DataFrame(rows)
    cols = [f"dice_vs_{s}" for s in SEQ_ROI]
    have = [c for c in cols if c in df.columns]
    df["reference_sensitivity"] = df[have].max(axis=1) - df[have].min(axis=1)

    print("\n" + "=" * 78)
    print("MULTI-REFERENCE EVALUATION")
    print("=" * 78)
    print(f"\n{'model':<10}" + "".join(f"{'vs ' + s:>14}" for s in SEQ_ROI)
          + f"{'ref sensitivity':>18}")
    summary = {}
    for model, g in df.groupby("model"):
        line = f"{model:<10}"
        entry = {}
        for s in SEQ_ROI:
            c = f"dice_vs_{s}"
            v = g[c].median() if c in g.columns and g[c].notna().any() else float("nan")
            line += f"{v:>14.4f}"
            entry[s] = None if np.isnan(v) else round(float(v), 4)
        line += f"{g.reference_sensitivity.median():>18.4f}"
        entry["reference_sensitivity_median"] = round(float(g.reference_sensitivity.median()), 4)
        entry["n"] = int(len(g))
        summary[model] = entry
        print(line)

    print("\n  Every row is the SAME set of predictions scored three ways. The spread")
    print("  is attributable to the reference standard, not to the model.")

    # Does the apparent best model depend on which reference is used?
    if len(summary) > 1:
        print("\n  Which model looks best, by reference:")
        flips = set()
        for s in SEQ_ROI:
            vals = {m: summary[m][s] for m in summary if summary[m].get(s) is not None}
            if vals:
                best = max(vals, key=vals.get)
                flips.add(best)
                print(f"    scored against {s:<9} -> {best} ({vals[best]:.4f})")
        print(f"    {'RANKING IS STABLE' if len(flips) == 1 else 'RANKING FLIPS with the reference'}")

    tables = cfg.out("tables_dir")
    suffix = "" if args.config_name == "3d_fullres" else f"_{args.config_name}"
    df.to_csv(tables / f"multireference_evaluation{suffix}.csv", index=False)
    with open(tables / f"multireference_summary{suffix}.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nWrote {tables}/multireference_evaluation{suffix}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
