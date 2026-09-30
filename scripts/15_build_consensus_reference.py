#!/usr/bin/env python3
"""Build a cross-sequence consensus reference and an nnU-Net dataset that uses it.

Motivation, which comes from this study's own measurements. Per-sequence
references disagree with each other by roughly 0.14 Dice once registered into a
common space. Training a model against a single sequence's reference therefore
teaches it that sequence's delineation idiosyncrasy along with the anatomy. A
model supervised by the agreement across sequences should learn the part that is
common to all of them.

Method. For each patient the three ROI masks are brought into the space of a
chosen target sequence by rigid within-session registration of the corresponding
images, then fused. Two fusion rules are produced:

    majority   a voxel is tumour when at least 2 of 3 registered references
               call it tumour. Conservative, close to the intersection.
    staple     STAPLE (Warfield et al.), which weights each reference by its
               estimated reliability rather than counting votes equally.

The consensus is written as a new nnU-Net dataset over the target sequence's
images, so that training against it is a controlled swap: identical images,
identical splits, only the supervision changes.

Honest limits. Rigid registration leaves residual misalignment, which makes the
consensus slightly tighter than truth near the boundary. Fusing across sequences
is not the same as fusing across readers, because the sequences genuinely show
different tissue; the consensus is the part they agree on, which is a defensible
target but not the only defensible one. Both points belong in the paper.

Usage:
    python3 scripts/15_build_consensus_reference.py --target T1WI --limit 12
    python3 scripts/15_build_consensus_reference.py --target T1WI
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

SEQ = {"T1WI": "ROI-T1.nii", "T2WI": "ROI-T2.nii", "CE-T1WI": "ROI-CE-T1.nii"}
DATASET_ID = {"T1WI": 504, "T2WI": 505, "CE-T1WI": 506}


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
    fixed_f, moving_f = sitk.Cast(fixed, sitk.sitkFloat32), sitk.Cast(moving, sitk.sitkFloat32)
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
        return R.Execute(fixed_f, moving_f), float(R.GetMetricValue())
    except Exception:  # noqa: BLE001
        return None, float("nan")


def dice(a, b):
    t = a.sum() + b.sum()
    return float(2 * np.logical_and(a, b).sum() / t) if t else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--allow-missing-roots", action="store_true")
    ap.add_argument("--target", default="T1WI", choices=list(SEQ))
    ap.add_argument("--fusion", default="staple", choices=["staple", "majority"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import SimpleITK as sitk

    cfg = load_config(args.config, verify=True, allow_missing=True)
    seg_root = cfg.primary_data_root / "MRI-Segments"
    ds_dir = cfg.repo_root / "nnunet" / "nnUNet_raw" / f"Dataset{DATASET_ID[args.target]}_NPC{args.target.replace('-', '')}CONS"
    img_dir, lbl_dir = ds_dir / "imagesTr", ds_dir / "labelsTr"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"CONSENSUS REFERENCE  target={args.target}  fusion={args.fusion}")
    print("=" * 78)

    pids = sorted(p.name for p in seg_root.iterdir() if p.is_dir())
    if args.limit:
        pids = pids[: args.limit]

    rows = []
    for i, pid in enumerate(pids, 1):
        pdir = seg_root / pid
        tgt_dir, tgt_roi = pdir / args.target, pdir / SEQ[args.target]
        if not (tgt_dir.is_dir() and tgt_roi.exists()):
            continue
        fixed = read_series(sitk, tgt_dir)
        if fixed is None:
            continue
        try:
            m_tgt = sitk.ReadImage(str(tgt_roi))
        except Exception:  # noqa: BLE001
            continue
        if fixed.GetSize() != m_tgt.GetSize():
            rows.append({"patient_id": pid, "status": "target size mismatch"})
            continue

        # STAPLE requires byte-identical geometry. The target mask and the
        # resampled ones differ in the last float digit of origin and spacing
        # (e.g. 5.5000005 vs 5.4999996), which trips ITK's tolerance check, so
        # every member of the stack is stamped with the target image's geometry.
        def stamp(binary_img):
            out = sitk.Cast(binary_img, sitk.sitkUInt8)
            out.CopyInformation(fixed)
            return out

        stack = [stamp(m_tgt > 0)]
        moved_names = [args.target]
        for s, roi in SEQ.items():
            if s == args.target:
                continue
            f, m = pdir / s, pdir / roi
            if not (f.is_dir() and m.exists()):
                continue
            mv = read_series(sitk, f)
            if mv is None:
                continue
            try:
                mk = sitk.ReadImage(str(m))
            except Exception:  # noqa: BLE001
                continue
            if mv.GetSize() != mk.GetSize():
                continue
            tx, _ = register(sitk, fixed, mv)
            if tx is None:
                continue
            res = sitk.Resample(mk, fixed, tx, sitk.sitkNearestNeighbor, 0, mk.GetPixelID())
            stack.append(stamp(res > 0))
            moved_names.append(s)

        if len(stack) < 2:
            rows.append({"patient_id": pid, "status": "too few references"})
            continue

        if args.fusion == "staple":
            fused = sitk.STAPLE(stack, 1.0)
            cons = sitk.Cast(fused > 0.5, sitk.sitkUInt8)
        else:
            acc = sum(sitk.Cast(s, sitk.sitkUInt16) for s in stack)
            cons = sitk.Cast(acc >= max(2, (len(stack) + 1) // 2), sitk.sitkUInt8)
        cons.CopyInformation(fixed)

        A = sitk.GetArrayFromImage(stack[0]) > 0
        C = sitk.GetArrayFromImage(cons) > 0
        if C.sum() == 0:
            rows.append({"patient_id": pid, "status": "empty consensus"})
            continue

        sitk.WriteImage(sitk.Cast(fixed, sitk.sitkFloat32), str(img_dir / f"NPC_{pid}_0000.nii.gz"), True)
        sitk.WriteImage(cons, str(lbl_dir / f"NPC_{pid}.nii.gz"), True)
        rows.append({"patient_id": pid, "status": "ok", "n_references": len(stack),
                     "references": "|".join(moved_names),
                     "vox_target_ref": int(A.sum()), "vox_consensus": int(C.sum()),
                     "dice_consensus_vs_target": round(dice(C, A), 4),
                     "consensus_over_target_ratio": round(float(C.sum()) / max(1, A.sum()), 4)})
        if i % 20 == 0 or i == len(pids):
            print(f"  {i}/{len(pids)}", flush=True)

    df = pd.DataFrame(rows)
    ok = df[df.status == "ok"]
    print(f"\n  built consensus for {len(ok)} patients, skipped {len(df) - len(ok)}")
    if ok.empty:
        return 1
    print(f"  consensus vs the target sequence's own reference: "
          f"median Dice {ok.dice_consensus_vs_target.median():.4f}")
    print(f"  consensus volume relative to target reference:     "
          f"median ratio {ok.consensus_over_target_ratio.median():.3f}")
    print("  (a ratio below 1 means the consensus is tighter, which is expected")
    print("   because it keeps only what the sequences agree on)")

    with open(ds_dir / "dataset.json", "w", encoding="utf-8") as fh:
        json.dump({"channel_names": {"0": "MRI"},
                   "labels": {"background": 0, "tumour": 1},
                   "numTraining": int(len(ok)), "file_ending": ".nii.gz",
                   "description": f"NPC {args.target} images with cross-sequence "
                                  f"{args.fusion} consensus reference"}, fh, indent=2)
    df.to_csv(cfg.out("tables_dir") / f"consensus_reference_{args.target}.csv", index=False)
    print(f"\n  dataset: {ds_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
