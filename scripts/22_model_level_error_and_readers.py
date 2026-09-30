#!/usr/bin/env python3
"""Model-level checks on reference sensitivity: grid transfer, estimator, readers.

Script 16 scored each trained 3D model against the three per-sequence references.
This script revisits those scores with the same predictions and the same rigid
registration, and answers four questions.

  1. Grid transfer. How much of a model's drop from its own reference to another
     sequence's reference is the cost of moving a mask between grids? The own
     reference is re-delineated on the other sequence's grid as the same physical
     shape and brought back through the pipeline with the exact inverse
     transform (script 21 describes the construction). The score against that
     transferred copy carries the grid-transfer loss and nothing else.

  2. Matched estimator. The published comparison set the difference between the
     two registered references (two values) against the best-to-worst range over
     all three. A range over three values is larger than a difference between two
     even without any effect, so the comparison is restated with every term a
     pairwise absolute difference.

  3. One alternative registration. The model trained on T1WI is re-scored with
     the affine variant of the pipeline registration.

  4. Readers. For the 30 held-out patients annotated by both radiologists, each
     model is also scored against the second reader's mask on every sequence.
     On the model's own sequence nothing is registered. On the other two the
     second reader's mask goes through the same transform as the first reader's,
     so a change of reader is compared with a change of sequence while the
     registration is held fixed.

Outputs:
    results/tables/model_level_scores.csv
    results/tables/model_level_summary.json

Usage:
    python3 scripts/22_model_level_error_and_readers.py --workers 12
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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
AFFINE_MODEL = "T1WI"


def process(job):
    pid, model, pred_dir, seg_root, rel_root = job
    import SimpleITK as sitk
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(s21.THREADS)
    arr = lambda im: sitk.GetArrayFromImage(im) > 0  # noqa: E731

    fixed = s21.read_series(sitk, seg_root / pid / model)
    pf = pred_dir / f"NPC_{pid}.nii.gz"
    if fixed is None or not pf.exists():
        return None
    P = arr(sitk.ReadImage(str(pf)))
    if P.shape != arr(fixed).shape:
        return None

    rec = {"patient_id": pid, "model": model}
    prim, sec = {}, {}
    for s in SEQS:
        m = seg_root / pid / ROI[s]
        if m.exists():
            prim[s] = sitk.Cast(sitk.ReadImage(str(m)) > 0, sitk.sitkUInt8)
        r2 = rel_root / pid / ROI[s]
        if r2.exists():
            sec[s] = sitk.Cast(sitk.ReadImage(str(r2)) > 0, sitk.sitkUInt8)
    if model not in prim or prim[model].GetSize() != fixed.GetSize():
        return None

    rec["dice_p_" + model] = s21.dice(P, arr(prim[model]))
    if model in sec and sec[model].GetSize() == fixed.GetSize():
        rec["dice_s_" + model] = s21.dice(P, arr(sec[model]))

    for b in SEQS:
        if b == model or b not in prim:
            continue
        mv = s21.read_series(sitk, seg_root / pid / b)
        if mv is None or mv.GetSize() != prim[b].GetSize():
            continue
        tx = s21.register(sitk, fixed, mv, "rigid")
        if tx is None:
            continue
        rec["dice_p_" + b] = s21.dice(P, arr(s21.transfer(sitk, prim[b], fixed, tx)))
        rec["floor_via_" + b] = s21.dice(P, arr(s21.transfer(
            sitk, s21.redraw(sitk, prim[model], mv, tx.GetInverse()), fixed, tx)))
        if b in sec and sec[b].GetSize() == mv.GetSize():
            rec["dice_s_" + b] = s21.dice(P, arr(s21.transfer(sitk, sec[b], fixed, tx)))
        if model == AFFINE_MODEL:
            txa = s21.register(sitk, fixed, mv, "affine")
            if txa is not None:
                rec["affine_p_" + b] = s21.dice(P, arr(s21.transfer(sitk, prim[b], fixed, txa)))
                # Distance between the tumour centroids placed by the two registrations,
                # used to recognise an affine optimisation that diverged.
                cb = s21.centroid_mm(sitk, prim[b])
                if cb is not None:
                    rec["affine_shift_mm_" + b] = float(np.linalg.norm(
                        np.array(tx.GetInverse().TransformPoint(tuple(cb)))
                        - np.array(txa.GetInverse().TransformPoint(tuple(cb)))))
    return rec


def boot_ratio(num, den, n=10000, seed=20260928):
    rng = np.random.default_rng(seed)
    num, den = np.asarray(num, float), np.asarray(den, float)
    idx = rng.integers(0, len(num), (n, len(num)))
    r = np.median(num[idx], axis=1) / np.median(den[idx], axis=1)
    return [round(float(v), 6) for v in np.percentile(r, [2.5, 97.5])]


def boot_median(x, n=10000, seed=20260928):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    b = np.median(x[rng.integers(0, len(x), (n, len(x)))], axis=1)
    return [round(float(v), 6) for v in np.percentile(b, [2.5, 97.5])]


def r4(v):
    # Six decimals despite the name: stored values are rounded once, when printed.
    return round(float(v), 6)


def wilcoxon_p(x, y):
    from scipy.stats import wilcoxon
    try:
        return float(wilcoxon(x, y).pvalue)
    except ValueError:
        return float("nan")


def matched_estimator(df_pub: pd.DataFrame) -> dict:
    """Every term a pairwise absolute difference, on the published per-patient scores."""
    out = {}
    for m in SEQS:
        g = df_pub[df_pub.model_seq == m]
        own = g[f"dice_vs_{m}"]
        b1, b2 = [g[f"dice_vs_{s}"] for s in SEQS if s != m]
        pairs = pd.concat([(own - b1).abs(), (own - b2).abs(), (b1 - b2).abs()], axis=1)
        out[m] = {
            "n": int(len(g)),
            "range_over_three": r4((pd.concat([own, b1, b2], axis=1).max(axis=1)
                                    - pd.concat([own, b1, b2], axis=1).min(axis=1)).median()),
            "mean_pairwise": r4(pairs.mean(axis=1).median()),
            "native_vs_registered": r4(pairs.iloc[:, :2].mean(axis=1).median()),
            "registered_pair": r4((b1 - b2).abs().median()),
            "own_advantage_signed": r4((own - (b1 + b2) / 2).median()),
        }
        out[m]["registered_over_native"] = r4(out[m]["registered_pair"] / out[m]["native_vs_registered"])
    for k in ["range_over_three", "mean_pairwise", "native_vs_registered",
              "registered_pair", "own_advantage_signed", "registered_over_native"]:
        out.setdefault("median_across_models", {})[k] = r4(np.median([out[m][k] for m in SEQS]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--allow-missing-roots", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only-two-reader", action="store_true",
                    help="score only the two-reader patients (a first, faster pass)")
    args = ap.parse_args()

    cfg = load_config(args.config, verify=True, allow_missing=True)
    tables = cfg.out("tables_dir")
    seg_root = cfg.primary_data_root / "MRI-Segments"
    rel_root = cfg.reliability_root
    multiref = cfg.repo_root / "nnunet" / "multiref"

    # The analysis population of the published matrix: scored by all three models
    # against all three references.
    pub = pd.read_csv(tables / "multireference_evaluation.csv", dtype={"patient_id": str})
    pub["patient_id"] = pub.patient_id.str.zfill(3)
    pub["model_seq"] = pub.model.str.replace("-3d_fullres", "", regex=False)
    cols = [f"dice_vs_{s}" for s in SEQS]
    complete = None
    for s in SEQS:
        ok = set(pub[pub.model_seq == s].dropna(subset=cols).patient_id)
        complete = ok if complete is None else complete & ok
    pids = sorted(complete)
    if args.limit:
        pids = pids[: args.limit]
    two_reader = {p.name for p in rel_root.iterdir() if p.is_dir()}
    if args.only_two_reader:
        pids = [p for p in pids if p in two_reader]

    # Rows already scored are reused, so an interrupted or partial run resumes.
    cache = tables / "model_level_scores.csv"
    rows, done = [], set()
    if cache.exists():
        old = pd.read_csv(cache, dtype={"patient_id": str})
        rows = old.to_dict(orient="records")
        done = set(zip(old.patient_id, old.model))
    jobs = [(p, m, multiref / PRED_DIRS[m], seg_root, rel_root)
            for p in pids for m in SEQS if (p, m) not in done]
    print(f"  {len(done)} scored rows reused, {len(jobs)} to score", flush=True)
    with Pool(args.workers) as pool:
        for k, rec in enumerate(pool.imap_unordered(process, jobs), 1):
            if rec is not None:
                rows.append(rec)
            if k % 10 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)}", flush=True)
                pd.DataFrame(rows).sort_values(["model", "patient_id"]).to_csv(
                    cache, index=False, float_format="%.4f")
    everything = pd.DataFrame(rows).sort_values(["model", "patient_id"])
    everything.to_csv(cache, index=False, float_format="%.4f")
    df = everything[everything.patient_id.isin(pids)]

    summary = {"n_patients": len(pids)}

    # 0. Reproduction of the published matrix.
    rep = {}
    for m in SEQS:
        g = df[df.model == m]
        rep[m] = {s: r4(g[f"dice_p_{s}"].median()) for s in SEQS}
    summary["reproduced_matrix"] = rep
    chk = df.merge(pub[["patient_id", "model_seq"] + cols].rename(columns={"model_seq": "model"}),
                   on=["patient_id", "model"], how="inner")
    summary["reproduction_max_abs_difference"] = r4(max(
        (chk[f"dice_p_{s}"] - chk[f"dice_vs_{s}"]).abs().max() for s in SEQS))

    # 1. Grid transfer at model level.
    gt = {}
    for m in SEQS:
        g = df[df.model == m]
        own = g[f"dice_p_{m}"]
        e = {}
        for b in SEQS:
            if b == m:
                continue
            fl, rg = g[f"floor_via_{b}"], g[f"dice_p_{b}"]
            e[b] = {"own_native": r4(own.median()),
                    "own_through_grid": r4(fl.median()),
                    "grid_transfer_loss": r4((own - fl).median()),
                    "grid_transfer_loss_mean": r4((own - fl).mean()),
                    "pct_loss_over_002": r4(100 * ((own - fl) > 0.02).mean()),
                    "reference_b": r4(rg.median()),
                    "observed_drop": r4((own - rg).median()),
                    "drop_beyond_grid_transfer": r4((fl - rg).median()),
                    "drop_beyond_grid_transfer_ci": boot_median((fl - rg).dropna())}
        gt[m] = e
    summary["grid_transfer_model_level"] = gt

    # 2. Matched estimator on the published per-patient scores.
    summary["matched_estimator"] = matched_estimator(pub[pub.patient_id.isin(pids)])

    # 3. Affine re-scoring of the T1WI model. An affine optimisation that moved the
    # tumour more than 10 mm away from the rigid solution has diverged (the rigid
    # correction itself rarely exceeds a few millimetres), and such patients are
    # counted and set aside rather than scored.
    g = df[df.model == AFFINE_MODEL].copy()
    regs = [s for s in SEQS if s != AFFINE_MODEL]
    div = pd.concat([g[f"affine_shift_mm_{b}"] > 10 for b in regs], axis=1).any(axis=1)
    okg = g[~div]
    rig = okg[[f"dice_p_{s}" for s in SEQS]]
    aff = pd.concat([okg[f"dice_p_{AFFINE_MODEL}"]] + [okg[f"affine_p_{b}"] for b in regs], axis=1)
    s_rig, s_aff = rig.max(axis=1) - rig.min(axis=1), aff.max(axis=1) - aff.min(axis=1)
    summary["affine_T1WI_model"] = {
        "n_patients": int(len(g)), "n_diverged": int(div.sum()), "n_converged": int(len(okg)),
        "median_rigid": {s: r4(okg[f"dice_p_{s}"].median()) for s in SEQS},
        "median_affine": {b: r4(okg[f"affine_p_{b}"].median()) for b in regs},
        "sensitivity_rigid": r4(s_rig.median()), "sensitivity_rigid_ci": boot_median(s_rig),
        "sensitivity_affine": r4(s_aff.median()), "sensitivity_affine_ci": boot_median(s_aff),
        "shift_mm_median": {b: r4(okg[f"affine_shift_mm_{b}"].median()) for b in regs},
    }

    # 4. Reader-controlled comparison on the two-reader patients.
    rc = {}
    for m in SEQS:
        g = df[(df.model == m) & df.patient_id.isin(two_reader)].copy()
        need = [f"dice_{r}_{s}" for r in "ps" for s in SEQS]
        g = g.dropna(subset=need)
        regs = [s for s in SEQS if s != m]
        reader_all = np.mean([(g[f"dice_p_{s}"] - g[f"dice_s_{s}"]).abs() for s in SEQS], axis=0)
        reader_native = (g[f"dice_p_{m}"] - g[f"dice_s_{m}"]).abs().to_numpy()
        reader_reg = np.mean([(g[f"dice_p_{s}"] - g[f"dice_s_{s}"]).abs() for s in regs], axis=0)
        seq_all = np.mean([(g[f"dice_{r}_{a}"] - g[f"dice_{r}_{b}"]).abs()
                           for r in "ps" for a, b in [(SEQS[0], SEQS[1]), (SEQS[0], SEQS[2]),
                                                       (SEQS[1], SEQS[2])]], axis=0)
        seq_reg = np.mean([(g[f"dice_{r}_{regs[0]}"] - g[f"dice_{r}_{regs[1]}"]).abs()
                           for r in "ps"], axis=0)
        rc[m] = {
            "n": int(len(g)),
            "median_dice_primary_reader": {s: r4(g[f"dice_p_{s}"].median()) for s in SEQS},
            "median_dice_second_reader": {s: r4(g[f"dice_s_{s}"].median()) for s in SEQS},
            "reader_effect_all": r4(np.median(reader_all)),
            "reader_effect_native": r4(np.median(reader_native)),
            "reader_effect_registered": r4(np.median(reader_reg)),
            "sequence_effect_all": r4(np.median(seq_all)),
            "sequence_effect_registered": r4(np.median(seq_reg)),
            "ratio_all": round(float(np.median(seq_all) / np.median(reader_all)), 6),
            "ratio_all_ci": boot_ratio(seq_all, reader_all),
            "wilcoxon_all_p": wilcoxon_p(seq_all, reader_all),
            "n_sequence_larger_all": int((seq_all > reader_all).sum()),
            "ratio_registered": round(float(np.median(seq_reg) / np.median(reader_reg)), 6),
            "ratio_registered_ci": boot_ratio(seq_reg, reader_reg),
            "wilcoxon_registered_p": wilcoxon_p(seq_reg, reader_reg),
            "n_sequence_larger_registered": int((seq_reg > reader_reg).sum()),
        }
    summary["reader_controlled"] = rc

    with open(tables / "model_level_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
