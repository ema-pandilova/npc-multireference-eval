#!/usr/bin/env python3
"""Registration quality control for the model-level evaluation.

Every model-level score against a transferred reference rests on one rigid
registration (fixed: the model's own sequence; moving: the other sequence). This
script repeats each of those registrations exactly as script 16 ran them and
records, without using any Dice value, how far the registration moved the tumour
centroid away from its position in scanner coordinates. Within one imaging session
the genuine correction is a few millimetres and in these data never more than about
17 mm, so a displacement above 20 mm marks a registration that converged to a wrong
alignment. The score obtained with scanner coordinates alone is recorded as well, so
that a failure can be told apart from a genuine disagreement.

Output: results/tables/model_registration_qc.csv

    python3 scripts/27_model_registration_qc.py --workers 6
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import load_config  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "s21", Path(__file__).resolve().parent / "21_resampling_registration_error.py")
s21 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s21)

SEQS = ["T1WI", "T2WI", "CE-T1WI"]
ROI = {"T1WI": "ROI-T1.nii", "T2WI": "ROI-T2.nii", "CE-T1WI": "ROI-CE-T1.nii"}
PRED_DIRS = {"T1WI": "pred_T1WI", "T2WI": "pred_T2WI_3d_fullres",
             "CE-T1WI": "pred_CE-T1WI_3d_fullres"}


def qc(job):
    pid, model, pred_dir, seg_root = job
    import SimpleITK as sitk
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(s21.THREADS)
    arr = lambda im: sitk.GetArrayFromImage(im) > 0  # noqa: E731
    fixed = s21.read_series(sitk, seg_root / pid / model)
    pf = pred_dir / f"NPC_{pid}.nii.gz"
    if fixed is None or not pf.exists():
        return []
    P = arr(sitk.ReadImage(str(pf)))
    identity = sitk.Transform(3, sitk.sitkIdentity)
    rows = []
    for b in SEQS:
        if b == model:
            continue
        ref = sitk.ReadImage(str(seg_root / pid / ROI[b]))
        mv = s21.read_series(sitk, seg_root / pid / b)
        if mv is None or mv.GetSize() != ref.GetSize():
            continue
        tx = s21.register(sitk, fixed, mv, "rigid")
        if tx is None:
            continue
        cb = s21.centroid_mm(sitk, ref)
        corr = (float(np.linalg.norm(np.array(tx.GetInverse().TransformPoint(tuple(cb))) - cb))
                if cb is not None else float("nan"))
        rows.append({"patient_id": pid, "model": model, "reference": b,
                     "correction_mm": corr,
                     "dice_rigid": s21.dice(P, arr(s21.transfer(sitk, ref, fixed, tx))),
                     "dice_scanner": s21.dice(P, arr(s21.transfer(sitk, ref, fixed, identity)))})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--allow-missing-roots", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    cfg = load_config(args.config, verify=True, allow_missing=True)
    tables = cfg.out("tables_dir")
    seg_root = cfg.primary_data_root / "MRI-Segments"
    multiref = cfg.repo_root / "nnunet" / "multiref"

    pub = pd.read_csv(tables / "multireference_evaluation.csv", dtype={"patient_id": str})
    pub["patient_id"] = pub.patient_id.str.zfill(3)
    pub["model_seq"] = pub.model.str.replace("-3d_fullres", "", regex=False)
    cols = [f"dice_vs_{s}" for s in SEQS]
    complete = None
    for s in SEQS:
        ok = set(pub[pub.model_seq == s].dropna(subset=cols).patient_id)
        complete = ok if complete is None else complete & ok

    jobs = [(p, m, multiref / PRED_DIRS[m], seg_root) for p in sorted(complete) for m in SEQS]
    rows = []
    with Pool(args.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(qc, jobs), 1):
            rows += r
            if k % 30 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)}", flush=True)
    df = pd.DataFrame(rows).sort_values(["model", "reference", "patient_id"])
    df.to_csv(tables / "model_registration_qc.csv", index=False, float_format="%.4f")
    fail = df[df.correction_mm > 20]
    print(f"{len(df)} registrations, {len(fail)} with correction above 20 mm:")
    print(fail.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
