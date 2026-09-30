#!/usr/bin/env python3
"""Score the second training run of the T1WI 3D model against all three references.

Identical to the scoring in script 16 (same registration, same transfer, same
patients), applied to the predictions of the repeated training run produced by
scripts/run_seed2_3d.sh. The difference between the two runs estimates training
variability for one configuration.

Output: results/tables/multireference_evaluation_seed2.csv

    python3 scripts/25_score_seed2.py --workers 8
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import load_config  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "s21", Path(__file__).resolve().parent / "21_resampling_registration_error.py")
s21 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s21)

SEQS = ["T1WI", "T2WI", "CE-T1WI"]
ROI = {"T1WI": "ROI-T1.nii", "T2WI": "ROI-T2.nii", "CE-T1WI": "ROI-CE-T1.nii"}


def score(job):
    pid, pred_dir, seg_root = job
    import SimpleITK as sitk
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(s21.THREADS)
    arr = lambda im: sitk.GetArrayFromImage(im) > 0  # noqa: E731
    fixed = s21.read_series(sitk, seg_root / pid / "T1WI")
    pf = pred_dir / f"NPC_{pid}.nii.gz"
    if fixed is None or not pf.exists():
        return None
    P = arr(sitk.ReadImage(str(pf)))
    rec = {"patient_id": pid}
    own = sitk.ReadImage(str(seg_root / pid / ROI["T1WI"]))
    if own.GetSize() != fixed.GetSize():
        return None
    rec["dice_vs_T1WI"] = s21.dice(P, arr(own))
    for b in ["T2WI", "CE-T1WI"]:
        ref = sitk.ReadImage(str(seg_root / pid / ROI[b]))
        mv = s21.read_series(sitk, seg_root / pid / b)
        if mv is None or mv.GetSize() != ref.GetSize():
            continue
        tx = s21.register(sitk, fixed, mv, "rigid")
        if tx is not None:
            rec[f"dice_vs_{b}"] = s21.dice(P, arr(s21.transfer(sitk, ref, fixed, tx)))
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--allow-missing-roots", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config(args.config, verify=True, allow_missing=True)
    tables = cfg.out("tables_dir")
    seg_root = cfg.primary_data_root / "MRI-Segments"
    pred_dir = cfg.repo_root / "nnunet" / "multiref" / "pred_T1WI_3d_fullres_seed2"

    pub = pd.read_csv(tables / "multireference_evaluation.csv", dtype={"patient_id": str})
    pub["patient_id"] = pub.patient_id.str.zfill(3)
    pub["model_seq"] = pub.model.str.replace("-3d_fullres", "", regex=False)
    cols = [f"dice_vs_{s}" for s in SEQS]
    complete = None
    for s in SEQS:
        ok = set(pub[pub.model_seq == s].dropna(subset=cols).patient_id)
        complete = ok if complete is None else complete & ok

    jobs = [(p, pred_dir, seg_root) for p in sorted(complete)]
    with Pool(args.workers) as pool:
        rows = [r for r in pool.imap_unordered(score, jobs) if r is not None]
    df = pd.DataFrame(rows).sort_values("patient_id")
    df.to_csv(tables / "multireference_evaluation_seed2.csv", index=False, float_format="%.4f")
    print(df[cols].median().round(4).to_string())
    print(f"{len(df)} patients scored")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
