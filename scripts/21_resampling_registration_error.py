#!/usr/bin/env python3
"""Separate grid transfer and registration from cross-sequence reference disagreement.

Comparing two per-sequence references needs two steps that the inter-reader
comparison never needs. The sequences are registered, and the moving mask is
resampled by nearest neighbour onto the fixed sequence's grid, whose pixel size
and slice positions differ. Both steps lower Dice even when the two masks
describe the same boundary. This script measures each step so that the observed
cross-sequence discrepancy can be read against them.

For every patient and sequence pair, with the pair order of script 13:

  observed      Dice(A, T(B)), the cross-sequence discrepancy as published, with
                T the rigid transfer of B onto A's grid. Recomputed here as a
                reproduction check.
  floor         Dice(A, T(redraw_b(A))). A is re-delineated on B's grid as the
                same physical shape, then brought back through the pipeline with
                the exact inverse transform. Registration error is zero by
                construction, so the loss is grid transfer alone. The re-delineation
                interpolates a signed distance map between slices (shape-based
                interpolation) and samples it at B's voxel centres.
  floor_nn      the cruder nearest-neighbour round trip A -> grid B -> grid A.
  affine        Dice(A, T_affine(B)), the one alternative registration, identical
                to the rigid pipeline except for the transform class.
  header        Dice(A, B moved by scanner coordinates alone, no registration).
  rigid_vs_aff  agreement between the rigid and affine transfers of the same mask,
                and the distance between their tumour centroids in mm.
  correction_mm how far the rigid registration moved the tumour centroid relative
                to scanner coordinates.

For the 30 patients annotated by both radiologists it adds the
resampling-matched reader comparison. The second reader's mask is re-delineated
on B's grid and transferred back exactly like a cross-sequence reference, so the
reader contrast carries the same grid-transfer loss and differs from the
cross-sequence contrast only by the sequence itself and by registration error.

Outputs:
    results/tables/resampling_registration_pairs.csv
    results/tables/resampling_matched_reader.csv
    results/tables/resampling_registration_summary.json

Usage:
    python3 scripts/21_resampling_registration_error.py --limit 3 --workers 3
    python3 scripts/21_resampling_registration_error.py --workers 12
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import load_config  # noqa: E402

SEQ = {"T1WI": "ROI-T1.nii", "T2WI": "ROI-T2.nii", "CE-T1WI": "ROI-CE-T1.nii"}
SEED = 20260727
THREADS = 2
CROP_MARGIN_MM = 25.0


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


def register(sitk, fixed, moving, kind: str = "rigid"):
    """The pipeline's registration (scripts 13 and 16); only the class may change."""
    fixed_f = sitk.Cast(fixed, sitk.sitkFloat32)
    moving_f = sitk.Cast(moving, sitk.sitkFloat32)
    base = sitk.Euler3DTransform() if kind == "rigid" else sitk.AffineTransform(3)
    init = sitk.CenteredTransformInitializer(
        fixed_f, moving_f, base, sitk.CenteredTransformInitializerFilter.GEOMETRY)
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.10, seed=SEED)
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
        return R.Execute(fixed_f, moving_f)
    except Exception:  # noqa: BLE001
        return None


def transfer(sitk, mask, grid, tx):
    """The pipeline step: nearest-neighbour resampling of a mask onto another grid."""
    return sitk.Resample(mask, grid, tx, sitk.sitkNearestNeighbor, 0, mask.GetPixelID())


def signed_distance(sitk, mask):
    """Signed distance in mm whose zero level lies midway between voxel centres.

    Negative inside, positive outside. Computed on a crop around the lesion,
    which keeps the geometry of the full image, so resampling it is exact.
    """
    from scipy.ndimage import distance_transform_edt

    arr = sitk.GetArrayFromImage(mask) > 0
    sp = mask.GetSpacing()                        # x, y, z
    idx = np.argwhere(arr)                        # z, y, x
    lo, hi = idx.min(0), idx.max(0) + 1
    marg = np.ceil(CROP_MARGIN_MM / np.array(sp[::-1])).astype(int)
    lo = np.maximum(lo - marg, 0)
    hi = np.minimum(hi + marg, arr.shape)
    sub = arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    samp = sp[::-1]
    phi = (distance_transform_edt(~sub, sampling=samp)
           - distance_transform_edt(sub, sampling=samp)).astype(np.float32)
    img = sitk.GetImageFromArray(phi)
    img.SetSpacing(sp)
    img.SetDirection(mask.GetDirection())
    img.SetOrigin(mask.TransformIndexToPhysicalPoint([int(lo[2]), int(lo[1]), int(lo[0])]))
    return img


def redraw(sitk, mask, grid, tx_grid_to_mask):
    """The same physical shape as `mask`, delineated on `grid`."""
    phi = sitk.Resample(signed_distance(sitk, mask), grid, tx_grid_to_mask,
                        sitk.sitkLinear, 1e4, sitk.sitkFloat32)
    return sitk.Cast(phi <= 0, sitk.sitkUInt8)


def centroid_mm(sitk, mask):
    st = sitk.LabelShapeStatisticsImageFilter()
    st.Execute(sitk.Cast(mask > 0, sitk.sitkUInt8))
    return np.array(st.GetCentroid(1)) if st.HasLabel(1) else None


def arr(sitk, img) -> np.ndarray:
    return sitk.GetArrayFromImage(img) > 0


def process(job):
    pid, seg_root, rel_root = job
    import SimpleITK as sitk
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(THREADS)

    pdir = seg_root / pid
    imgs, masks, second = {}, {}, {}
    for s, roi in SEQ.items():
        f, m = pdir / s, pdir / roi
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
        # Mask headers are used as released, exactly as in scripts 13 and 16.
        imgs[s], masks[s] = im, sitk.Cast(mk > 0, sitk.sitkUInt8)
        r2 = rel_root / pid / roi
        if r2.exists():
            m2 = sitk.ReadImage(str(r2))
            if m2.GetSize() == im.GetSize():
                second[s] = sitk.Cast(m2 > 0, sitk.sitkUInt8)

    pair_rows, reader_rows = [], []
    identity = sitk.Transform(3, sitk.sitkIdentity)
    for a, b in itertools.combinations(sorted(imgs), 2):
        rec = {"patient_id": pid, "fixed": a, "moving": b}
        A, B = masks[a], masks[b]
        if not (arr(sitk, A).any() and arr(sitk, B).any()):
            rec["status"] = "empty mask"
            pair_rows.append(rec)
            continue
        tx = register(sitk, imgs[a], imgs[b], "rigid")
        if tx is None:
            rec["status"] = "rigid failed"
            pair_rows.append(rec)
            continue
        tx_inv = tx.GetInverse()
        A_np = arr(sitk, A)

        moved = transfer(sitk, B, imgs[a], tx)
        rec["observed"] = dice(A_np, arr(sitk, moved))
        rec["header"] = dice(A_np, arr(sitk, transfer(sitk, B, imgs[a], identity)))
        rec["floor"] = dice(A_np, arr(sitk, transfer(sitk, redraw(sitk, A, imgs[b], tx_inv),
                                                     imgs[a], tx)))
        rec["floor_nn"] = dice(A_np, arr(sitk, transfer(
            sitk, transfer(sitk, A, imgs[b], tx_inv), imgs[a], tx)))

        cb = centroid_mm(sitk, B)
        if cb is not None:
            rec["correction_mm"] = float(np.linalg.norm(
                np.array(tx_inv.TransformPoint(tuple(cb))) - cb))

        txa = register(sitk, imgs[a], imgs[b], "affine")
        if txa is not None:
            moved_a = transfer(sitk, B, imgs[a], txa)
            rec["affine"] = dice(A_np, arr(sitk, moved_a))
            rec["rigid_vs_affine_dice"] = dice(arr(sitk, moved), arr(sitk, moved_a))
            if cb is not None:
                rec["rigid_vs_affine_mm"] = float(np.linalg.norm(
                    np.array(tx_inv.TransformPoint(tuple(cb)))
                    - np.array(txa.GetInverse().TransformPoint(tuple(cb)))))
        rec["status"] = "ok"
        pair_rows.append(rec)

        # Resampling-matched reader comparison, two-reader patients only.
        if a in second and b in second:
            As, Bs = second[a], second[b]
            As_np = arr(sitk, As)
            if As_np.any() and arr(sitk, Bs).any():
                matched = np.mean([
                    dice(A_np, arr(sitk, transfer(sitk, redraw(sitk, As, imgs[b], tx_inv),
                                                  imgs[a], tx))),
                    dice(As_np, arr(sitk, transfer(sitk, redraw(sitk, A, imgs[b], tx_inv),
                                                   imgs[a], tx)))])
                seq_second = dice(As_np, arr(sitk, transfer(sitk, Bs, imgs[a], tx)))
                reader_rows.append({
                    "patient_id": pid, "fixed": a, "moving": b,
                    "cross_seq_primary": rec["observed"],
                    "cross_seq_second": seq_second,
                    "cross_seq": float(np.mean([rec["observed"], seq_second])),
                    "reader_native": dice(A_np, As_np),
                    "reader_matched": float(matched)})
    return pair_rows, reader_rows


def boot_ci(x, stat=np.median, n=10000, seed=20260928):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    b = [stat(rng.choice(x, len(x), replace=True)) for _ in range(n)]
    return [round(float(v), 4) for v in np.percentile(b, [2.5, 97.5])]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--allow-missing-roots", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    from scipy.stats import wilcoxon

    cfg = load_config(args.config, verify=True, allow_missing=True)
    seg_root = cfg.primary_data_root / "MRI-Segments"
    rel_root = cfg.reliability_root
    tables = cfg.out("tables_dir")

    pids = sorted(p.name for p in seg_root.iterdir() if p.is_dir())
    if args.limit:
        pids = pids[: args.limit]
    jobs = [(p, seg_root, rel_root) for p in pids]

    pair_rows, reader_rows = [], []
    with Pool(args.workers) as pool:
        for k, (pr, rr) in enumerate(pool.imap_unordered(process, jobs), 1):
            pair_rows += pr
            reader_rows += rr
            if k % 20 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} patients", flush=True)

    pairs = pd.DataFrame(pair_rows).sort_values(["patient_id", "fixed", "moving"])
    pairs.to_csv(tables / "resampling_registration_pairs.csv", index=False,
                 float_format="%.4f")
    ok = pairs[pairs.status == "ok"].copy()

    # Reproduction check against the published per-pair Dice.
    pub = pd.read_csv(tables / "reference_swap_pairs.csv", dtype={"patient_id": str})
    pub["patient_id"] = pub.patient_id.str.zfill(3)
    chk = ok.merge(pub[["patient_id", "fixed", "moving", "dice"]],
                   on=["patient_id", "fixed", "moving"], how="inner")
    repro = {"n_pairs_matched": int(len(chk)),
             "published_median": round(float(chk.dice.median()), 4),
             "recomputed_median": round(float(chk.observed.median()), 4),
             "median_abs_difference": round(float((chk.dice - chk.observed).abs().median()), 4),
             "max_abs_difference": round(float((chk.dice - chk.observed).abs().max()), 4)}

    def med(c):
        return round(float(ok[c].median()), 4)

    summary = {
        "n_pairs": int(len(ok)), "n_patients": int(ok.patient_id.nunique()),
        "reproduction": repro,
        "median_dice": {c: med(c) for c in
                        ["observed", "floor", "floor_nn", "affine", "header"]},
        "iqr_dice": {c: [round(float(ok[c].quantile(q)), 4) for q in (.25, .75)]
                     for c in ["observed", "floor", "floor_nn", "affine", "header"]},
        "floor_by_pair": {f"{a} vs {b}": round(float(g.floor.median()), 4)
                          for (a, b), g in ok.groupby(["fixed", "moving"])},
        "observed_by_pair": {f"{a} vs {b}": round(float(g.observed.median()), 4)
                             for (a, b), g in ok.groupby(["fixed", "moving"])},
        "affine_by_pair": {f"{a} vs {b}": round(float(g.affine.median()), 4)
                           for (a, b), g in ok.groupby(["fixed", "moving"])},
        "grid_transfer_loss_median": round(float((1 - ok.floor).median()), 4),
        "observed_loss_median": round(float((1 - ok.observed).median()), 4),
        "share_of_observed_loss_from_grid_transfer_median": round(float(
            ((1 - ok.floor) / (1 - ok.observed)).median()), 4),
        "affine_minus_rigid_median": round(float((ok.affine - ok.observed).median()), 4),
        "rigid_vs_affine_dice_median": med("rigid_vs_affine_dice"),
        "rigid_vs_affine_mm_median": med("rigid_vs_affine_mm"),
        "rigid_vs_affine_mm_p90": round(float(ok.rigid_vs_affine_mm.quantile(.9)), 3),
        "correction_mm_median": med("correction_mm"),
        "correction_mm_p90": round(float(ok.correction_mm.quantile(.9)), 3),
    }

    if reader_rows:
        rd = pd.DataFrame(reader_rows)
        rd.to_csv(tables / "resampling_matched_reader.csv", index=False, float_format="%.4f")
        per = rd.groupby("patient_id").agg(
            cross_seq=("cross_seq", "median"),
            cross_seq_primary=("cross_seq_primary", "median"),
            reader_matched=("reader_matched", "median"))
        ir = pd.read_csv(tables / "inter_rater_agreement.csv", dtype={"patient_id": str})
        ir["patient_id"] = ir.patient_id.str.zfill(3)
        per = per.join(ir.groupby("patient_id").dice.median().rename("reader_native"))
        per = per.dropna()
        diff_nat = per.reader_native - per.cross_seq_primary
        diff_mat = per.reader_matched - per.cross_seq
        summary["reader_comparison"] = {
            "n_patients": int(len(per)),
            "reader_native_median": round(float(per.reader_native.median()), 4),
            "reader_matched_median": round(float(per.reader_matched.median()), 4),
            "cross_seq_primary_median": round(float(per.cross_seq_primary.median()), 4),
            "cross_seq_both_readers_median": round(float(per.cross_seq.median()), 4),
            "unmatched_gap_median": round(float(diff_nat.median()), 4),
            "unmatched_gap_ci": boot_ci(diff_nat),
            "matched_gap_median": round(float(diff_mat.median()), 4),
            "matched_gap_ci": boot_ci(diff_mat),
            "matched_gap_wilcoxon_p": float(wilcoxon(per.reader_matched, per.cross_seq).pvalue),
            "n_matched_reader_higher": int((diff_mat > 0).sum()),
        }
        per.round(4).to_csv(tables / "resampling_matched_reader_per_patient.csv")

    with open(tables / "resampling_registration_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
