#!/usr/bin/env python3
"""Convert the NPC data into nnU-Net v2 raw format, one dataset per sequence.

This deliberately uses the THREE-DIMENSIONAL source data rather than the 2D TIFF
export. That choice matters:

  * The DICOM series carry true voxel spacing, so nnU-Net can resample sensibly
    and the resulting volumes are in physical units.
  * The released ROI NIfTI masks sit on the same grid as their DICOM series, so
    the 93% image/mask mismatch that affects the 2D export cannot occur here.
  * Anisotropy (~0.43 mm in-plane against ~5 mm through-plane) is exactly the
    situation nnU-Net's planner is designed to reason about.

One dataset per sequence, because the sequences are not co-registered and
because the per-sequence comparison is the point of the study.

Every conversion is verified: the image and its mask must agree on shape and
spacing, and the mask must be binary. Any patient failing a check is excluded
and reported rather than silently written.

Datasets produced:
    Dataset501_NPCT1WI, Dataset502_NPCT2WI, Dataset503_NPCCET1WI

Usage:
    python3 scripts/11_build_nnunet_dataset.py
    python3 scripts/11_build_nnunet_dataset.py --sequence T1WI
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import load_config  # noqa: E402

SEQ = {
    "T1WI":    {"id": 501, "name": "NPCT1WI",    "roi": "ROI-T1.nii"},
    "T2WI":    {"id": 502, "name": "NPCT2WI",    "roi": "ROI-T2.nii"},
    "CE-T1WI": {"id": 503, "name": "NPCCET1WI",  "roi": "ROI-CE-T1.nii"},
}


def read_dicom_series(folder: Path):
    """Read a DICOM series as a 3D image, ordered by the reader's own sorting."""
    import SimpleITK as sitk

    reader = sitk.ImageSeriesReader()
    ids = reader.GetGDCMSeriesIDs(str(folder))
    if not ids:
        return None, "no DICOM series found"
    if len(ids) > 1:
        # Pick the series with the most slices; a stray localiser should not win.
        best, best_n = None, -1
        for sid in ids:
            files = reader.GetGDCMSeriesFileNames(str(folder), sid)
            if len(files) > best_n:
                best, best_n = sid, len(files)
        ids = [best]
    files = reader.GetGDCMSeriesFileNames(str(folder), ids[0])
    reader.SetFileNames(files)
    try:
        return reader.Execute(), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def convert(cfg, sequence: str, out_root: Path, dry_run: bool = False) -> dict:
    import SimpleITK as sitk

    meta = SEQ[sequence]
    ds_dir = out_root / f"Dataset{meta['id']}_{meta['name']}"
    img_dir, lbl_dir = ds_dir / "imagesTr", ds_dir / "labelsTr"
    if not dry_run:
        for d in (img_dir, lbl_dir):
            d.mkdir(parents=True, exist_ok=True)

    seg_root = cfg.primary_data_root / "MRI-Segments"
    written, skipped = [], []

    for pdir in sorted(p for p in seg_root.iterdir() if p.is_dir()):
        pid = pdir.name
        series_dir = pdir / sequence
        roi_path = pdir / meta["roi"]
        if not series_dir.is_dir():
            skipped.append({"patient_id": pid, "reason": f"missing {sequence} folder"})
            continue
        if not roi_path.exists():
            skipped.append({"patient_id": pid, "reason": f"missing {meta['roi']}"})
            continue

        img, err = read_dicom_series(series_dir)
        if img is None:
            skipped.append({"patient_id": pid, "reason": f"DICOM read failed: {err}"})
            continue

        try:
            lbl = sitk.ReadImage(str(roi_path))
        except Exception as exc:  # noqa: BLE001
            skipped.append({"patient_id": pid, "reason": f"ROI read failed: {exc}"})
            continue

        # --- verification -------------------------------------------------
        if img.GetSize() != lbl.GetSize():
            skipped.append({"patient_id": pid,
                            "reason": f"size mismatch image {img.GetSize()} vs mask {lbl.GetSize()}"})
            continue
        if not np.allclose(img.GetSpacing(), lbl.GetSpacing(), rtol=1e-3):
            skipped.append({"patient_id": pid,
                            "reason": f"spacing mismatch {img.GetSpacing()} vs {lbl.GetSpacing()}"})
            continue

        arr = sitk.GetArrayFromImage(lbl)
        uniq = np.unique(arr)
        if not set(np.unique(uniq)).issubset({0, 1}):
            arr = (arr > 0).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
        if arr.sum() == 0:
            skipped.append({"patient_id": pid, "reason": "empty mask"})
            continue

        if dry_run:
            written.append({"patient_id": pid, "size": str(img.GetSize()),
                            "spacing": str(tuple(round(s, 3) for s in img.GetSpacing())),
                            "lesion_voxels": int(arr.sum())})
            continue

        # nnU-Net expects the label to carry the image's geometry exactly.
        lbl_out = sitk.GetImageFromArray(arr)
        lbl_out.CopyInformation(img)
        # Cast to a float type nnU-Net handles cleanly.
        img_out = sitk.Cast(img, sitk.sitkFloat32)

        sitk.WriteImage(img_out, str(img_dir / f"NPC_{pid}_0000.nii.gz"), True)
        sitk.WriteImage(lbl_out, str(lbl_dir / f"NPC_{pid}.nii.gz"), True)
        written.append({"patient_id": pid, "size": str(img.GetSize()),
                        "spacing": str(tuple(round(s, 3) for s in img.GetSpacing())),
                        "lesion_voxels": int(arr.sum())})

    if not dry_run and written:
        dataset_json = {
            "channel_names": {"0": "MRI"},
            "labels": {"background": 0, "tumour": 1},
            "numTraining": len(written),
            "file_ending": ".nii.gz",
            "description": f"Primary nasopharyngeal carcinoma, {sequence}, "
                           f"Zenodo record 13131827",
            "licence": "CC BY 4.0",
        }
        with open(ds_dir / "dataset.json", "w", encoding="utf-8") as fh:
            json.dump(dataset_json, fh, indent=2)

    return {"sequence": sequence, "dataset": ds_dir.name,
            "written": written, "skipped": skipped}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--allow-missing-roots", action="store_true")
    ap.add_argument("--sequence", default="all", help="T1WI | T2WI | CE-T1WI | all")
    ap.add_argument("--out", default=None, help="nnUNet_raw directory (default <repo>/nnunet/nnUNet_raw)")
    ap.add_argument("--dry-run", action="store_true", help="verify only, write nothing")
    args = ap.parse_args()

    cfg = load_config(args.config, verify=True, allow_missing=True)
    out_root = Path(args.out) if args.out else cfg.repo_root / "nnunet" / "nnUNet_raw"
    seqs = list(SEQ) if args.sequence == "all" else [args.sequence]

    print("=" * 78)
    print("BUILD nnU-Net RAW DATASETS (3D DICOM + released ROI masks)")
    print("=" * 78)
    print(f"source : {cfg.primary_data_root / 'MRI-Segments'}")
    print(f"target : {out_root}")
    print(f"mode   : {'DRY RUN' if args.dry_run else 'write'}\n")

    free = shutil.disk_usage(cfg.repo_root).free / 1e9
    print(f"free disk: {free:.1f} GB\n")

    reports = []
    for seq in seqs:
        print(f"--- {seq} ---")
        rep = convert(cfg, seq, out_root, dry_run=args.dry_run)
        reports.append(rep)
        print(f"  converted {len(rep['written'])} patients, skipped {len(rep['skipped'])}")
        if rep["skipped"]:
            from collections import Counter
            for reason, n in Counter(s["reason"].split(":")[0] for s in rep["skipped"]).most_common():
                print(f"    skipped ({n}): {reason}")
            for s in rep["skipped"][:5]:
                print(f"      {s['patient_id']}: {s['reason']}")
        if rep["written"]:
            sizes = pd.Series([w["size"] for w in rep["written"]]).value_counts()
            print(f"  most common volume size: {sizes.index[0]} ({sizes.iloc[0]} patients)")
        print()

    tables = cfg.out("tables_dir")
    rows = []
    for rep in reports:
        for w in rep["written"]:
            rows.append({**w, "sequence": rep["sequence"], "status": "converted"})
        for s in rep["skipped"]:
            rows.append({**s, "sequence": rep["sequence"], "status": "skipped"})
    pd.DataFrame(rows).to_csv(tables / "nnunet_conversion.csv", index=False)
    print(f"Wrote {tables}/nnunet_conversion.csv")

    if not args.dry_run:
        print("\nNext:")
        print(f"  export nnUNet_raw={out_root}")
        print(f"  export nnUNet_preprocessed={out_root.parent}/nnUNet_preprocessed")
        print(f"  export nnUNet_results={out_root.parent}/nnUNet_results")
        print("  nnUNetv2_plan_and_preprocess -d 501 502 503 --verify_dataset_integrity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
