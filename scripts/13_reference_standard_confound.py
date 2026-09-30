#!/usr/bin/env python3
"""Quantify how much a Dice score depends on which sequence supplied the reference.

This is the segmentation-evaluation core of the study. A model is scored by
agreement with a reference standard. On multi-sequence data that reference is
delineated separately per sequence, so the target itself moves. We measure how
far it moves, in the same units the field reports.

    E7  The achievable ceiling. Two readers on the same sequence agree at Dice
        0.891 to 0.904. A model scoring above that is reproducing one reader's
        style, not recovering an observer-independent boundary, so published
        scores should be read against this ceiling rather than against 1.0.

    E8  The reference-swap penalty. For each patient we rigidly register one
        sequence to another within the same imaging session and transfer the
        mask, then compute Dice between the two per-sequence references. That
        value is what a perfect model trained on sequence X would score if it
        were evaluated against sequence Y's reference. It is the cost of
        reference choice alone, with model quality held perfect.

Registration is rigid and within-session, which is the appropriate model for the
same patient scanned minutes apart. It is not exact: the sequences have different
fields of view and resolutions, and a residual misregistration inflates apparent
disagreement. We therefore report the registration metric alongside every case
and treat E8 as an upper bound on agreement, that is a lower bound on the
penalty, rather than a point estimate.

Usage:
    python3 scripts/13_reference_standard_confound.py --limit 30   # quick check
    python3 scripts/13_reference_standard_confound.py              # all patients
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import load_config  # noqa: E402

SEQ = {"T1WI": "ROI-T1.nii", "T2WI": "ROI-T2.nii", "CE-T1WI": "ROI-CE-T1.nii"}
READER_DICE = {"T1WI": 0.8907, "T2WI": 0.9035, "CE-T1WI": 0.9006}


def dice(a: np.ndarray, b: np.ndarray) -> float:
    tot = a.sum() + b.sum()
    return float(2 * np.logical_and(a, b).sum() / tot) if tot else float("nan")


def read_series(sitk, folder: Path):
    reader = sitk.ImageSeriesReader()
    ids = reader.GetGDCMSeriesIDs(str(folder))
    if not ids:
        return None
    best, n = None, -1
    for sid in ids:
        files = reader.GetGDCMSeriesFileNames(str(folder), sid)
        if len(files) > n:
            best, n = sid, len(files)
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(folder), best))
    try:
        return reader.Execute()
    except Exception:  # noqa: BLE001
        return None


def register(sitk, fixed, moving):
    """Rigid registration of moving onto fixed, Mattes mutual information.

    Same patient, same session, so rigid is the right transform class; anything
    deformable would absorb genuine boundary disagreement into the warp and make
    the sequences look more similar than they are.
    """
    fixed_f = sitk.Cast(fixed, sitk.sitkFloat32)
    moving_f = sitk.Cast(moving, sitk.sitkFloat32)
    init = sitk.CenteredTransformInitializer(
        fixed_f, moving_f, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.10, seed=20260727)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-4, numberOfIterations=200,
        gradientMagnitudeTolerance=1e-6)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2, 1, 0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    R.SetInitialTransform(init, inPlace=False)
    try:
        tx = R.Execute(fixed_f, moving_f)
        return tx, float(R.GetMetricValue())
    except Exception as exc:  # noqa: BLE001
        return None, float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--allow-missing-roots", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N patients")
    args = ap.parse_args()

    import SimpleITK as sitk

    cfg = load_config(args.config, verify=True, allow_missing=True)
    seg_root = cfg.primary_data_root / "MRI-Segments"
    tables = cfg.out("tables_dir")

    print("=" * 78)
    print("REFERENCE STANDARD CONFOUND")
    print("=" * 78)

    # ---- E7: the achievable ceiling -----------------------------------
    print("\nE7  Achievable Dice ceiling, set by inter-reader agreement")
    for s, d in READER_DICE.items():
        print(f"    {s:<9} two readers agree at Dice {d:.4f}")
    print(f"    mean ceiling across sequences: {np.mean(list(READER_DICE.values())):.4f}")
    print("    A model scoring above this is reproducing one reader's delineation")
    print("    style rather than an observer-independent boundary.")

    # ---- E8: reference-swap penalty ------------------------------------
    print("\nE8  Reference-swap penalty (rigid within-session registration)")
    pids = sorted(p.name for p in seg_root.iterdir() if p.is_dir())
    if args.limit:
        pids = pids[: args.limit]
    print(f"    patients: {len(pids)}   pairs per patient: 3\n")

    rows = []
    for i, pid in enumerate(pids, 1):
        pdir = seg_root / pid
        imgs, masks = {}, {}
        for s, roi in SEQ.items():
            f = pdir / s
            m = pdir / roi
            if not (f.is_dir() and m.exists()):
                continue
            im = read_series(sitk, f)
            if im is None:
                continue
            try:
                mk = sitk.ReadImage(str(m))
            except Exception:  # noqa: BLE001
                continue
            if im.GetSize() != mk.GetSize():
                continue
            imgs[s], masks[s] = im, mk

        for a, b in itertools.combinations(sorted(imgs), 2):
            tx, metric = register(sitk, imgs[a], imgs[b])
            if tx is None:
                rows.append({"patient_id": pid, "fixed": a, "moving": b,
                             "dice": np.nan, "status": "registration failed"})
                continue
            moved = sitk.Resample(masks[b], imgs[a], tx, sitk.sitkNearestNeighbor,
                                  0, masks[b].GetPixelID())
            A = sitk.GetArrayFromImage(masks[a]) > 0
            B = sitk.GetArrayFromImage(moved) > 0
            d = dice(A, B)
            rows.append({"patient_id": pid, "fixed": a, "moving": b, "dice": d,
                         "reg_metric": round(metric, 4),
                         "vox_fixed": int(A.sum()), "vox_moving": int(B.sum()),
                         "status": "ok"})
        if i % 20 == 0 or i == len(pids):
            print(f"    {i}/{len(pids)} patients", flush=True)

    df = pd.DataFrame(rows)
    ok = df[df.status == "ok"].dropna(subset=["dice"])
    if ok.empty:
        print("\n    no successful registrations")
        return 1

    print(f"\n    successful pairs: {len(ok)} / {len(df)}")
    print(f"\n    {'sequence pair':<24}{'n':>5}{'median Dice':>14}{'IQR':>18}")
    summary = []
    for (a, b), g in ok.groupby(["fixed", "moving"]):
        print(f"    {a + ' vs ' + b:<24}{len(g):>5}{g.dice.median():>14.4f}"
              f"{f'{g.dice.quantile(.25):.3f}-{g.dice.quantile(.75):.3f}':>18}")
        summary.append({"pair": f"{a} vs {b}", "n": int(len(g)),
                        "median_dice": round(float(g.dice.median()), 4),
                        "q1": round(float(g.dice.quantile(.25)), 4),
                        "q3": round(float(g.dice.quantile(.75)), 4)})

    overall = float(ok.dice.median())
    ceiling = float(np.mean(list(READER_DICE.values())))
    print(f"\n    overall median cross-sequence reference agreement: {overall:.4f}")
    print(f"    same-sequence inter-reader ceiling:                 {ceiling:.4f}")
    print(f"    penalty for scoring against the wrong sequence:     {ceiling - overall:.4f} Dice")
    print("\n    Interpretation: a model that perfectly reproduced one sequence's")
    print("    reference would score about this much lower if evaluated against")
    print("    another sequence's reference, with model quality held perfect.")
    print("    Registration residual makes this an upper bound on agreement, so")
    print("    the true penalty is at least this large.")

    df.to_csv(tables / "reference_swap_pairs.csv", index=False)
    out = {"ceiling_by_sequence": READER_DICE, "mean_ceiling": round(ceiling, 4),
           "cross_sequence_median_dice": round(overall, 4),
           "penalty_dice": round(ceiling - overall, 4),
           "by_pair": summary, "n_pairs_ok": int(len(ok)), "n_pairs_total": int(len(df))}
    with open(tables / "reference_standard_confound.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {tables}/reference_standard_confound.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
