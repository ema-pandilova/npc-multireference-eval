#!/usr/bin/env python3
"""Boundary-sensitive metrics for the multi-reference evaluation.

The multi-reference comparison had rested on the Dice coefficient alone. Dice is
a volumetric overlap index and is relatively insensitive to where the boundary
sits, which is awkward for a paper whose subject is precisely that the boundary
moves between references. This script recomputes the same comparison with
metrics that respond to boundary placement, using the predictions already on
disk. No retraining is involved.

    Surface Dice   overlap of the two surfaces within a tolerance, reported at
                   1 mm and at 2 mm.
    HD95           95th percentile Hausdorff distance, in millimetres.
    ASSD           average symmetric surface distance, in millimetres.
    RVE            relative volume error, signed, as a percentage.

Distances use the true voxel spacing of the fixed image, not voxel counts, so
the anisotropy of these acquisitions is respected.

Outputs:
    results/tables/surface_metrics.csv
    results/tables/surface_metrics_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import load_config  # noqa: E402

SEQ_ROI = {"T1WI": "ROI-T1.nii", "T2WI": "ROI-T2.nii", "CE-T1WI": "ROI-CE-T1.nii"}


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


def surface_distances(sitk, a: np.ndarray, b: np.ndarray, spacing) -> tuple:
    """Symmetric surface distances in millimetres, both directions."""
    from scipy.ndimage import binary_erosion, distance_transform_edt

    if not a.any() or not b.any():
        return None
    st = np.zeros((3, 3, 3), bool); st[1, :, :] = True; st[:, 1, :] = True; st[:, :, 1] = True
    sa = a & ~binary_erosion(a, st)
    sb = b & ~binary_erosion(b, st)
    if not sa.any() or not sb.any():
        return None
    # spacing is (x, y, z) from SimpleITK; arrays are (z, y, x)
    sp = (spacing[2], spacing[1], spacing[0])
    da = distance_transform_edt(~sb, sampling=sp)[sa]
    db = distance_transform_edt(~sa, sampling=sp)[sb]
    return da, db


def metrics(sitk, pred: np.ndarray, ref: np.ndarray, spacing) -> dict:
    inter = np.logical_and(pred, ref).sum()
    tot = pred.sum() + ref.sum()
    out = {"dice": round(float(2 * inter / tot), 4) if tot else np.nan}
    vp, vr = pred.sum() * np.prod(spacing), ref.sum() * np.prod(spacing)
    out["rve_pct"] = round(float((vp - vr) / vr * 100), 2) if vr else np.nan

    sd = surface_distances(sitk, pred.astype(bool), ref.astype(bool), spacing)
    if sd is None:
        out.update({"hd95_mm": np.nan, "assd_mm": np.nan,
                    "surface_dice_1mm": np.nan, "surface_dice_2mm": np.nan})
        return out
    da, db = sd
    both = np.concatenate([da, db])
    out["hd95_mm"] = round(float(np.percentile(both, 95)), 2)
    out["assd_mm"] = round(float(both.mean()), 2)
    for tol in (1.0, 2.0):
        num = (da <= tol).sum() + (db <= tol).sum()
        out[f"surface_dice_{int(tol)}mm"] = round(float(num / (len(da) + len(db))), 4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--allow-missing-roots", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import SimpleITK as sitk

    cfg = load_config(args.config, verify=True, allow_missing=True)
    tables = cfg.out("tables_dir")
    seg_root = cfg.primary_data_root / "MRI-Segments"
    pred_dir = cfg.repo_root / "nnunet" / "multiref" / "pred_T1WI"

    pids = sorted(p.name.replace("NPC_", "").replace(".nii.gz", "")
                  for p in pred_dir.glob("NPC_*.nii.gz"))
    if args.limit:
        pids = pids[: args.limit]
    print(f"scoring {len(pids)} held-out patients against three references\n")

    rows = []
    for k, pid in enumerate(pids, 1):
        pf = pred_dir / f"NPC_{pid}.nii.gz"
        own = seg_root / pid / "T1WI"
        if not own.is_dir():
            continue
        pred_img = sitk.ReadImage(str(pf))
        P = sitk.GetArrayFromImage(pred_img) > 0
        spacing = pred_img.GetSpacing()
        fixed = read_series(sitk, own)
        if fixed is None:
            continue
        rec = {"patient_id": pid}
        for ref_seq, roi in SEQ_ROI.items():
            rp = seg_root / pid / roi
            if not rp.exists():
                continue
            try:
                ref = sitk.ReadImage(str(rp))
            except Exception:  # noqa: BLE001
                continue
            if ref_seq == "T1WI":
                if ref.GetSize() != pred_img.GetSize():
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
            for m, v in metrics(sitk, P, R, spacing).items():
                rec[f"{m}__{ref_seq}"] = v
        rows.append(rec)
        if k % 20 == 0 or k == len(pids):
            print(f"  {k}/{len(pids)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(tables / "surface_metrics.csv", index=False)

    # Restrict to the complete-case set before aggregating anything. Taking a
    # column-wise median instead lets pandas skip NaN per column, which silently
    # gives a different population to each column and readmits patients the
    # patient-flow table declares excluded.
    ref_cols = [f"dice__{s}" for s in SEQ_ROI if f"dice__{s}" in df.columns]
    dfc = df.dropna(subset=ref_cols) if ref_cols else df
    dropped = sorted(set(df.patient_id) - set(dfc.patient_id))
    if dropped:
        print(f"\nexcluded from aggregation, incomplete against all three references: "
              f"{', '.join(dropped)}")

    print("\n" + "=" * 78)
    print("BOUNDARY-SENSITIVE METRICS, ONE MODEL SCORED AGAINST EACH REFERENCE")
    print("=" * 78)
    summary = {}
    names = [("dice", "Dice", "%.3f"), ("surface_dice_1mm", "Surface Dice 1mm", "%.3f"),
             ("surface_dice_2mm", "Surface Dice 2mm", "%.3f"), ("hd95_mm", "HD95 (mm)", "%.2f"),
             ("assd_mm", "ASSD (mm)", "%.2f"), ("rve_pct", "RVE (%)", "%+.1f")]
    print(f"\n{'metric':<20}" + "".join(f"{s:>14}" for s in SEQ_ROI) + f"{'spread':>12}")
    for key, label, fmt in names:
        cols = [f"{key}__{s}" for s in SEQ_ROI if f"{key}__{s}" in dfc.columns]
        if not cols:
            continue
        meds = [dfc[c].median() for c in cols]
        line = f"{label:<20}" + "".join(f"{(fmt % m):>14}" for m in meds)
        spread = max(meds) - min(meds)
        line += f"{(fmt % spread):>12}"
        print(line)
        summary[key] = {s: (None if np.isnan(m) else round(float(m), 4))
                        for s, m in zip(SEQ_ROI, meds)}
        summary[key]["spread_of_medians"] = round(float(spread), 4)
        per_pt = (dfc[cols].max(axis=1) - dfc[cols].min(axis=1)).median()
        summary[key]["reference_sensitivity_median"] = round(float(per_pt), 4)
        summary[key]["n"] = int(len(dfc))

    print("\n  Every column is the same fixed set of predictions. Only the reference changes.")
    with open(tables / "surface_metrics_summary.json", "w", encoding="utf-8") as fh:
        json.dump({"n_scored": int(len(df)), "n_complete": int(len(dfc)),
                   "excluded": dropped, "metrics": summary}, fh, indent=2)
    print(f"\nWrote {tables}/surface_metrics.csv  "
          f"(scored {len(df)}, aggregated over {len(dfc)} complete cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
