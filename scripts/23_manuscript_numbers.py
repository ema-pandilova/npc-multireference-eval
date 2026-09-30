#!/usr/bin/env python3
"""Every number the revised manuscript cites, recomputed from the stored tables.

The manuscript quotes values that earlier scripts printed but did not save (the
mixed-model coefficients, the Friedman and Holm tests on model scores, the
configuration comparison, the 2D export counts). This script recomputes each of
them from the per-patient tables under results/tables, together with the new
registration and resampling analyses (scripts 21 and 22), and writes one file
that the manuscript is checked against.

    python3 scripts/23_manuscript_numbers.py            # uses cached 2D counts
    python3 scripts/23_manuscript_numbers.py --tiff     # rescans the 2D export

Output: results/tables/manuscript_numbers.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import load_config  # noqa: E402

SEQS = ["T1WI", "T2WI", "CE-T1WI"]
MORPH = {"T1WI": "T1", "T2WI": "T2", "CE-T1WI": "CET1"}
SEED = 20260928
QC_MM = 20.0   # registration displacing the tumour further than this has failed


def r(v, k=6):
    """Round half up on the shortest decimal representation.

    Python's round() works on the binary value, so a median of exactly 0.9035
    becomes 0.903. Manuscripts round the decimal value, which gives 0.904.
    """
    from decimal import Decimal, ROUND_HALF_UP
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    # Twelve significant digits removes binary noise such as 0.9034999999999999.
    q = Decimal(f"{float(v):.12g}").quantize(Decimal(1).scaleb(-k), rounding=ROUND_HALF_UP)
    return float(q)


def med_iqr(x, k=6):
    x = pd.Series(x).dropna()
    return {"median": r(x.median(), k), "q1": r(x.quantile(.25), k), "q3": r(x.quantile(.75), k),
            "n": int(len(x))}


def boot_median_ci(x, n=10000, seed=SEED):
    x = np.asarray(pd.Series(x).dropna(), float)
    rng = np.random.default_rng(seed)
    b = np.median(x[rng.integers(0, len(x), (n, len(x)))], axis=1)
    return [r(v) for v in np.percentile(b, [2.5, 97.5])]


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def grid_mismatch_2d(images_root: Path) -> dict:
    """Image against mask matrix in the distributed 2D TIFF export."""
    from PIL import Image
    tot, bad, per_seq, mask_sizes, pats = 0, 0, {}, {}, set()
    pat = re.compile(r"^(T1WI|T2WI|CE-T1WI)_(\d{3})\.tif$")
    for pdir in sorted(p for p in images_root.iterdir() if p.is_dir()):
        for f in sorted(pdir.iterdir()):
            m = pat.match(f.name)
            if not m:
                continue
            mk = pdir / f"{m.group(1)}_{m.group(2)}_mask.tif"
            if not mk.exists():
                continue
            with Image.open(f) as a, Image.open(mk) as b:
                sa, sb = a.size, b.size
            tot += 1
            s = per_seq.setdefault(m.group(1), [0, 0])
            s[0] += 1
            if sa != sb:
                bad += 1
                s[1] += 1
                pats.add(pdir.name)
                mask_sizes[sb[0]] = mask_sizes.get(sb[0], 0) + 1
    return {"pairs": tot, "mismatched": bad, "pct": r(100 * bad / tot, 1),
            "patients_affected": len(pats),
            "pct_by_sequence": {k: r(100 * v[1] / v[0], 1) for k, v in per_seq.items()},
            "mask_matrix_sizes": sorted(mask_sizes)}


def slice_geometry(cfg, T: Path) -> dict:
    """Whether the two sequences of a pair share slice planes, from the mask headers.

    A pair counts as coincident when the slice normals agree within 0.5 degrees, the
    slice spacing is equal and the planes lie within 0.5 mm of each other. Header
    geometry needs no registration, so this separates acquisitions on common planes
    from acquisitions on offset or differently spaced slices.
    """
    import SimpleITK as sitk
    seg = cfg.primary_data_root / "MRI-Segments"
    roi = {"T1WI": "ROI-T1.nii", "T2WI": "ROI-T2.nii", "CE-T1WI": "ROI-CE-T1.nii"}
    pr = pd.read_csv(T / "resampling_registration_pairs.csv", dtype={"patient_id": str})
    pr = pr[pr.status == "ok"].copy()
    cache = {}

    def geo(pid, s):
        if (pid, s) not in cache:
            rd = sitk.ImageFileReader()
            rd.SetFileName(str(seg / pid / roi[s]))
            rd.ReadImageInformation()
            cache[(pid, s)] = (np.array(rd.GetOrigin()), np.array(rd.GetSpacing()),
                               np.array(rd.GetDirection()).reshape(3, 3))
        return cache[(pid, s)]

    coinc = []
    for _, x in pr.iterrows():
        oa, sa, da = geo(x.patient_id, x.fixed)
        ob, sb, db = geo(x.patient_id, x.moving)
        ang = np.degrees(np.arccos(np.clip(abs(np.dot(da[:, 2], db[:, 2])), 0, 1)))
        off = abs(np.dot(oa - ob, da[:, 2])) % sb[2]
        off = min(off, sb[2] - off)
        coinc.append(bool(ang < 0.5 and off < 0.5 and abs(sa[2] - sb[2]) < 0.01))
    pr["coincident"] = coinc
    c, o = pr[pr.coincident], pr[~pr.coincident]
    return {"n_coincident": int(len(c)), "n_offset": int(len(o)),
            "floor_coincident": med_iqr(c.floor), "floor_offset": med_iqr(o.floor),
            "observed_coincident": r(c.observed.median()), "observed_offset": r(o.observed.median()),
            "pct_offset_by_pair": {f"{a} vs {b}": r(100 * (~g.coincident).mean(), 1)
                                   for (a, b), g in pr.groupby(["fixed", "moving"])}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--allow-missing-roots", action="store_true")
    ap.add_argument("--tiff", action="store_true", help="rescan the 2D TIFF export")
    args = ap.parse_args()

    from scipy.stats import friedmanchisquare, wilcoxon, kruskal

    cfg = load_config(args.config, verify=True, allow_missing=True)
    T = cfg.out("tables_dir")
    N: dict = {}

    # ---------------------------------------------------------------- dataset
    ds = json.loads((T / "dataset_summary.json").read_text())
    N["dataset"] = {"patients": ds["n_patient_dirs"], "slices": ds["total_slices"],
                    "masks": ds["total_masks"],
                    "absent_masks": ds["total_slices"] - ds["total_masks"],
                    "patients_with_absent_masks": ds["n_pairing_problems"],
                    "two_reader_patients": ds["n_reliability"]}
    sc = pd.read_csv(T / "scanner_distribution.csv")
    by_field = sc.groupby("field_t").n_patients.sum()
    N["scanners"] = {"models": sc[["manufacturer", "model", "field_t", "n_patients"]]
                     .to_dict(orient="records"),
                     "n_models": int(len(sc)), "n_vendors": int(sc.manufacturer.nunique()),
                     "by_field": {str(k): {"n": int(v), "pct": r(100 * v / 277, 1)}
                                  for k, v in by_field.items()}}
    meta = pd.read_csv(T / "scanner_metadata.csv")
    meta = meta[meta.status == "OK"]
    N["acquisition"] = {"series": int(len(meta)),
                        "median_inplane_mm": r(meta.pixel_spacing_x_mm.median(), 2),
                        "median_slice_thickness_mm": r(meta.SliceThickness.median(), 1)}
    geo = pd.read_csv(T / "reliability_geometry.csv", dtype={"patient_id": str})
    shp = geo.pivot(index="patient_id", columns="sequence", values="shape")
    inpl = shp.apply(lambda s: s.str.rsplit("x", n=1).str[0])
    N["reliability_archive"] = {"masks": int(len(geo)),
                                "masks_256": int((geo["shape"].str.startswith("256x256")).sum()),
                                "t1_t2_different_inplane_grid": int((inpl["T1WI"] != inpl["T2WI"]).sum()),
                                "patients": int(len(shp))}
    conv = pd.read_csv(T / "nnunet_conversion.csv", dtype={"patient_id": str})
    conv = conv[conv.status == "converted"]
    zsp = conv.spacing.str.strip("()").str.split(",").str[2].astype(float)
    N["acquisition"]["median_slice_spacing_mm"] = r(zsp.median(), 2)
    N["acquisition"]["pct_series_spacing_5p5"] = r(100 * (zsp.round(2) == 5.5).mean(), 1)
    cache = T / "grid_mismatch_2d.json"
    if args.tiff or not cache.exists():
        cache.write_text(json.dumps(grid_mismatch_2d(cfg.images_root), indent=2))
    N["grid_mismatch_2d"] = json.loads(cache.read_text())

    # --------------------------------------------------- released volumes, 277
    s06 = load_module("s06", Path(__file__).resolve().parent / "06_morphology_variability.py")

    # ------------------------------------------------------ cohort characteristics
    dx = s06.read_diagnoses(cfg.primary_data_root / "info_diagnoses.csv")
    dx = dx[0] if isinstance(dx, tuple) else dx
    age = pd.to_numeric(dx["age"], errors="coerce")
    sex = dx["gender"].astype(str).str.strip().str.lower()
    hist = dx["histopathological type"].astype(str).str.strip().str.lower()
    tst = pd.to_numeric(dx["T"], errors="coerce")
    stage = dx["stage_norm"].astype(str).str.upper().str.strip()
    N["cohort"] = {
        "n": int(len(dx)),
        "age_mean": r(age.mean(), 1), "age_sd": r(age.std(), 1),
        "age_median": r(age.median(), 1), "age_range": [r(age.min(), 0), r(age.max(), 0)],
        "n_male": int((sex == "male").sum()), "pct_male": r(100 * (sex == "male").mean(), 1),
        "n_nonkeratinising": int(hist.str.contains("non-keratin").sum()),
        "pct_nonkeratinising": r(100 * hist.str.contains("non-keratin").mean(), 1),
        "t_stage_counts": {str(int(k)): int(v) for k, v in tst.value_counts().sort_index().items()},
        "overall_stage_counts": {k: int(v) for k, v in stage.value_counts().sort_index().items()},
    }
    mo = s06.read_morphology(cfg.primary_data_root / "morphological_parameters.csv")
    V = pd.DataFrame({s: mo[f"{MORPH[s]}_Vol"] / 1000.0 for s in SEQS})
    V.index = mo.patient_id
    V = V.dropna()
    fr = friedmanchisquare(*[V[s] for s in SEQS])
    pw = {f"{a} vs {b}": float(wilcoxon(V[a], V[b]).pvalue) for a, b in combinations(SEQS, 2)}
    ms = pd.read_csv(T / "morphology_variability_summary.csv")
    # Spreads from the unrounded released values; the summary CSV is rounded to two
    # decimals, and rounding it again would round twice.
    spreads = {p: s06.spreads(mo, p) for p in ["Vol", "SA", "MaxD", "SReg"]}
    desc = {p: {"median": r(s.median(), 1), "iqr": [r(s.quantile(.25), 1), r(s.quantile(.75), 1)],
                "max": r(s.max(), 1)} for p, s in spreads.items()}
    sv = spreads["Vol"]
    mcs = pd.read_csv(T / "morphology_cross_sequence.csv", dtype={"patient_id": str})
    N["volume_277"] = {
        "n": int(len(V)),
        "median_cm3": {s: r(V[s].median(), 2) for s in SEQS},
        "range_cm3": [r(V.min().min(), 2), r(V.max().max(), 2)],
        "spread_median_pct": desc["Vol"]["median"],
        "spread_iqr_pct": desc["Vol"]["iqr"],
        "spread_max_pct": desc["Vol"]["max"],
        "pct_over_10": r(100 * (sv > 10).mean(), 1),
        "pct_over_20": r(100 * (sv > 20).mean(), 1),
        "pct_over_30": r(100 * (sv > 30).mean(), 1),
        "descriptor_spreads": desc,
        "diameter_over_volume_factor": r(spreads["Vol"].median() / spreads["MaxD"].median(), 2),
        "friedman_chi2": r(fr.statistic, 1), "friedman_p": float(fr.pvalue),
        "pairwise_wilcoxon_holm": {k: float(v) for k, v in holm(pw).items()},
        "ce_largest": int((mcs.largest_volume_sequence == "CE-T1WI").sum()),
        "t2_smallest": int((mcs.smallest_volume_sequence == "T2WI").sum()),
        # Same definition as the reader comparison: |a - b| over the pair mean.
        "median_abs_pct_diff": {f"{a} vs {b}": r(np.median(
            np.abs(V[a] - V[b]) / ((V[a] + V[b]) / 2) * 100), 2)
            for a, b in combinations(SEQS, 2)},
    }
    ra = json.loads((T / "reference_agreement_clinical.json").read_text())
    N["volume_agreement"] = ra

    # ---------------------------------------------------------------- readers
    ir = pd.read_csv(T / "inter_rater_agreement.csv", dtype={"patient_id": str})
    icc = pd.read_csv(T / "reader_agreement_icc_ba.csv").set_index("sequence")
    N["readers"] = {s: {"dice": r(ir[ir.sequence == s].dice.median(), 3),
                        "jaccard": r(ir[ir.sequence == s].jaccard.median(), 3),
                        "icc": r(icc.loc[s, "icc21"], 3),
                        "loa": [r(icc.loc[s, "ba_loa_low"], 2), r(icc.loc[s, "ba_loa_high"], 2)],
                        "median_abs_dv_pct": r(icc.loc[s, "median_abs_pct_diff"], 2)}
                    for s in SEQS}
    N["readers"]["loa_extremes_pct"] = [r(100 * (icc.ba_loa_low.min() - 1), 0),
                                        r(100 * (icc.ba_loa_high.max() - 1), 0)]
    # Width of the limits of agreement on the log scale, readers against sequences.
    w_read = np.log(icc.ba_loa_high / icc.ba_loa_low)
    ra_pw = pd.DataFrame(json.loads((T / "reference_agreement_clinical.json").read_text())["pairwise"]).T
    w_seq = np.log(ra_pw.loa_upper.astype(float) / ra_pw.loa_lower.astype(float))
    N["readers"]["loa_logwidth_readers"] = [r(w_read.min(), 3), r(w_read.max(), 3)]
    N["readers"]["loa_logwidth_sequences"] = [r(w_seq.min(), 3), r(w_seq.max(), 3)]
    N["readers"]["loa_width_ratio_range"] = [r(w_seq.min() / w_read.max(), 2),
                                             r(w_seq.max() / w_read.min(), 2)]

    # ----------------------------------------------- sequence vs reader, n = 30
    sm = pd.read_csv(T / "sequence_vs_reader_matched.csv", dtype={"patient_id": str})
    sj = json.loads((T / "sequence_vs_reader_matched.json").read_text())
    N["seq_vs_reader_volume"] = {
        "reader": med_iqr(sm.reader, 2), "sequence_matched": med_iqr(sm.sequence_matched, 2),
        "sequence_range": med_iqr(sm.sequence_range, 2),
        "ratio_matched": sj["ratio_matched"], "ci": [sj["ci_low"], sj["ci_high"]],
        "wilcoxon_p": sj["wilcoxon_p"], "n_sequence_higher": sj["n_sequence_higher"],
        "ratio_range": r(sm.sequence_range.median() / sm.reader.median(), 2)}
    import statsmodels.formula.api as smf
    rv = pd.read_csv(T / "reader_volumes_paired.csv", dtype={"patient_id": str})
    long = pd.concat([rv.assign(reader="R1", vol=rv.reader1_cm3),
                      rv.assign(reader="R2", vol=rv.reader2_cm3)])
    long = long[long.vol > 0].copy()
    long["log_vol"] = np.log(long.vol)
    fit = smf.mixedlm("log_vol ~ C(sequence) + C(reader)", long,
                      groups=long["patient_id"]).fit(reml=True)
    pe = json.loads((T / "primary_experiments.json").read_text())["E1_variance"]
    tot = pe["var_patient"] + pe["var_sequence"] + pe["var_reader"] + pe["var_residual"]
    N["mixed_model"] = {
        "n_obs": int(len(long)),
        "pct_patient": r(100 * pe["var_patient"] / tot, 1),
        "pct_sequence": r(pe["pct_sequence"], 2), "pct_reader": r(pe["pct_reader"], 3),
        "pct_residual": r(100 * pe["var_residual"] / tot, 1),
        "coef": {c.split("[T.")[-1].rstrip("]"): {"ratio": r(np.exp(fit.params[c]), 3),
                                                    "p": float(fit.pvalues[c])}
                 for c in fit.params.index if c.startswith("C(")}}

    # ------------------------------------------------------- spatial agreement
    N["spatial"] = {"all_pairs": json.loads((T / "reference_standard_confound.json").read_text()),
                    "paired_30": json.loads((T / "paired_sequence_vs_reader_dice.json").read_text())}
    rs = T / "resampling_registration_summary.json"
    rdf = T / "resampling_matched_reader.csv"
    if rdf.exists():
        rd = pd.read_csv(rdf, dtype={"patient_id": str})
        per = rd.groupby("patient_id").agg(cross_seq=("cross_seq", "median"),
                                            cross_seq_primary=("cross_seq_primary", "median"),
                                            reader_matched=("reader_matched", "median"))
        irr = pd.read_csv(T / "inter_rater_agreement.csv", dtype={"patient_id": str})
        irr["patient_id"] = irr.patient_id.str.zfill(3)
        per = per.join(irr.groupby("patient_id").dice.median().rename("reader_native")).dropna()
        gap = per.reader_matched - per.cross_seq
        N["reader_comparison"] = {
            "n": int(len(per)), "reader_native": r(per.reader_native.median()),
            "reader_matched": r(per.reader_matched.median()),
            "cross_seq_both": r(per.cross_seq.median()),
            "cross_seq_primary": r(per.cross_seq_primary.median()),
            "matched_gap": r(gap.median()), "matched_gap_ci": boot_median_ci(gap),
            "matched_gap_wilcoxon_p": float(wilcoxon(per.reader_matched, per.cross_seq).pvalue),
            "n_matched_reader_higher": int((gap > 0).sum())}
    if rs.exists():
        N["slice_geometry"] = slice_geometry(cfg, T)
        N["resampling_registration"] = json.loads(rs.read_text())
        pr = pd.read_csv(T / "resampling_registration_pairs.csv", dtype={"patient_id": str})
        pr = pr[pr.status == "ok"]
        N["resampling_registration"]["extra"] = {
            "floor": med_iqr(pr.floor), "floor_nn": med_iqr(pr.floor_nn),
            "observed": med_iqr(pr.observed), "affine": med_iqr(pr.affine),
            "header": med_iqr(pr.header),
            "pct_pairs_floor_below_090": r(100 * (pr.floor < 0.90).mean(), 1),
            "affine_minus_rigid": med_iqr(pr.affine - pr.observed),
            "loss_ratio_of_medians": r((1 - pr.floor).median() / (1 - pr.observed).median(), 3),
            "header_minus_rigid": med_iqr(pr.header - pr.observed),
            "correction_mm": med_iqr(pr.correction_mm, 2),
            "rigid_vs_affine_mm": med_iqr(pr.rigid_vs_affine_mm, 2),
            "pct_correction_over_2mm": r(100 * (pr.correction_mm > 2).mean(), 1)}
        # An affine optimisation that places the tumour more than 10 mm from the rigid
        # solution has diverged; the rigid correction itself is a few millimetres.
        cv = pr[pr.rigid_vs_affine_mm <= 10]
        dv = pr[pr.rigid_vs_affine_mm > 10]
        N["resampling_registration"]["affine_divergence"] = {
            "n_converged": int(len(cv)), "n_diverged": int(len(dv)),
            "pct_diverged": r(100 * len(dv) / len(pr), 1),
            "diverged_by_pair": {f"{a} vs {b}": int(len(g)) for (a, b), g in dv.groupby(["fixed", "moving"])},
            "diverged_patients": int(dv.patient_id.nunique()),
            "diverged_rigid_median": r(dv.observed.median()), "diverged_affine_median": r(dv.affine.median()),
            "diverged_header_median": r(dv.header.median()),
            "converged_rigid_median": r(cv.observed.median()), "converged_affine_median": r(cv.affine.median()),
            "converged_affine_minus_rigid": med_iqr(cv.affine - cv.observed),
            "converged_rigid_vs_affine_dice": med_iqr(cv.rigid_vs_affine_dice),
            "converged_rigid_vs_affine_mm": med_iqr(cv.rigid_vs_affine_mm, 2),
            "converged_affine_by_pair": {f"{a} vs {b}": r(g.affine.median())
                                         for (a, b), g in cv.groupby(["fixed", "moving"])},
            "max_rigid_correction_mm": r(pr.correction_mm.max(), 1),
            "n_rigid_correction_over_10mm": int((pr.correction_mm > 10).sum())}
        # Registration quality control without Dice: a rigid registration that moved the
        # tumour more than 20 mm from its scanner-coordinate position converged to a wrong
        # alignment; genuine within-session corrections here stay below that.
        bad = pr[pr.correction_mm > QC_MM]
        good = pr[pr.correction_mm <= QC_MM]
        N["registration_qc"] = {
            "threshold_mm": QC_MM,
            "annotation_level": {
                "n_failed": int(len(bad)), "failed_patients": sorted(bad.patient_id.unique().tolist()),
                "failed_correction_mm": [r(bad.correction_mm.min(), 1), r(bad.correction_mm.max(), 1)],
                "failed_rigid_dice_max": r(bad.observed.max(), 3),
                "failed_scanner_dice": [r(bad.header.min(), 3), r(bad.header.max(), 3)],
                "max_accepted_correction_mm": r(good.correction_mm.max(), 1),
                "median_all": r(pr.observed.median()), "median_excluding": r(good.observed.median()),
                "by_pair_excluding": {f"{a} vs {b}": r(g.observed.median())
                                      for (a, b), g in good.groupby(["fixed", "moving"])},
                "n_by_pair_excluding": {f"{a} vs {b}": int(len(g))
                                        for (a, b), g in good.groupby(["fixed", "moving"])},
                "floor_excluding": r(good.floor.median()),
                "failed_pairs": bad[["patient_id", "fixed", "moving"]].values.tolist(),
                # The largest accepted correction, where registration was essential.
                "largest_accepted": good.sort_values("correction_mm").tail(2)[
                    ["patient_id", "fixed", "moving", "correction_mm", "observed", "header"]
                ].round(3).to_dict(orient="records")}}

    # ------------------------------------------------------------ model level
    me = pd.read_csv(T / "multireference_evaluation.csv", dtype={"patient_id": str})
    me["patient_id"] = me.patient_id.str.zfill(3)
    me["model_seq"] = me.model.str.replace("-3d_fullres", "", regex=False)
    cols = [f"dice_vs_{s}" for s in SEQS]
    complete = None
    for s in SEQS:
        ok = set(me[me.model_seq == s].dropna(subset=cols).patient_id)
        complete = ok if complete is None else complete & ok
    t1 = me[(me.model_seq == "T1WI") & me.patient_id.isin(complete)].set_index("patient_id")
    sens = t1[cols].max(axis=1) - t1[cols].min(axis=1)
    frm = friedmanchisquare(*[t1[c] for c in cols])
    pwm = {f"{a} vs {b}": float(wilcoxon(t1[f"dice_vs_{a}"], t1[f"dice_vs_{b}"]).pvalue)
           for a, b in combinations(SEQS, 2)}
    N["model_T1WI"] = {
        "n": int(len(t1)),
        "dice": {s: med_iqr(t1[f"dice_vs_{s}"], 3) for s in SEQS},
        "friedman_chi2": r(frm.statistic, 1), "friedman_p": float(frm.pvalue),
        "wilcoxon_holm": holm(pwm),
        "sensitivity": med_iqr(sens), "sensitivity_ci": boot_median_ci(sens),
        "pct_sens_over_005": r(100 * (sens > 0.05).mean(), 0),
        "pct_sens_over_010": r(100 * (sens > 0.10).mean(), 0),
        "median_shift_t1_to_ce": r(t1["dice_vs_T1WI"].median() - t1["dice_vs_CE-T1WI"].median(), 3)}
    qcf = T / "model_registration_qc.csv"
    if qcf.exists() and "registration_qc" in N:
        q = pd.read_csv(qcf, dtype={"patient_id": str})
        q["patient_id"] = q.patient_id.str.zfill(3)
        qb = q[q.correction_mm > QC_MM]
        # Same registrations as the published scores: check they reproduce.
        rep = q.merge(me[["patient_id", "model_seq"] + cols].rename(columns={"model_seq": "model"}),
                      on=["patient_id", "model"])
        rep_diff = max((rep[rep.reference == s].dice_rigid - rep[rep.reference == s][f"dice_vs_{s}"])
                       .abs().max() for s in SEQS)
        bad_t1 = set(qb[qb.model == "T1WI"].patient_id)
        t1x = t1[~t1.index.isin(bad_t1)]
        sx = t1x[cols].max(axis=1) - t1x[cols].min(axis=1)
        bad_any = set(qb.patient_id)
        mx = {}
        for s in SEQS:
            g = me[(me.model_seq == s) & me.patient_id.isin(complete) & ~me.patient_id.isin(bad_any)]
            mx[s] = {c: r(g[f"dice_vs_{c}"].median(), 3) for c in SEQS}
            mx[s]["sensitivity"] = r((g[cols].max(axis=1) - g[cols].min(axis=1)).median())
            mx[s]["n"] = int(len(g))
        N["registration_qc"]["model_level"] = {
            "n_registrations": int(len(q)), "n_failed": int(len(qb)),
            "failed": qb[["patient_id", "model", "reference"]].values.tolist(),
            "failed_correction_mm": [r(qb.correction_mm.min(), 1), r(qb.correction_mm.max(), 1)],
            "failed_scanner_dice": [r(qb.dice_scanner.min(), 3), r(qb.dice_scanner.max(), 3)],
            "max_accepted_correction_mm": r(q[q.correction_mm <= QC_MM].correction_mm.max(), 1),
            "rigid_reproduction_max_abs_difference": r(rep_diff),
            "t1wi_n_excluding": int(len(t1x)),
            "t1wi_dice_excluding": {s: r(t1x[f"dice_vs_{s}"].median(), 3) for s in SEQS},
            "t1wi_sensitivity_excluding": r(sx.median()),
            "t1wi_sensitivity_excluding_ci": boot_median_ci(sx),
            "matrix_excluding": mx,
            "matrix_sensitivity_excluding_range": [min(v["sensitivity"] for v in mx.values()),
                                                   max(v["sensitivity"] for v in mx.values())]}
    N["matrix"] = json.loads((T / "model_reference_matrix.json").read_text())
    # The same matrix at full precision from the per-patient scores, so that a value
    # printed at three decimals is never rounded twice.
    mraw = {}
    for s in SEQS:
        g = me[(me.model_seq == s) & me.patient_id.isin(complete)]
        mraw[s] = {"per_reference": {c: r(g[f"dice_vs_{c}"].median()) for c in SEQS},
                   "sensitivity": r((g[cols].max(axis=1) - g[cols].min(axis=1)).median()),
                   "n": int(len(g))}
    N["matrix_raw"] = mraw
    N["surface"] = json.loads((T / "surface_metrics_summary.json").read_text())
    ml = T / "model_level_summary.json"
    if ml.exists():
        N["model_level"] = json.loads(ml.read_text())

    # ------------------------------------------------ 3D against 2D, magnitude
    d2 = pd.read_csv(T / "multireference_evaluation_2d.csv", dtype={"patient_id": str})
    d2["patient_id"] = d2.patient_id.str.zfill(3)
    d2 = d2[d2.patient_id.isin(complete)].set_index("patient_id").loc[t1.index]
    gap = pd.DataFrame({s: t1[f"dice_vs_{s}"] - d2[f"dice_vs_{s}"] for s in SEQS}).dropna()
    frg = friedmanchisquare(*[gap[s] for s in SEQS])
    N["config_3d_2d"] = {
        "n": int(len(gap)),
        "median_3d": {s: r(t1[f"dice_vs_{s}"].median()) for s in SEQS},
        "median_2d": {s: r(d2[f"dice_vs_{s}"].median()) for s in SEQS},
        "difference_of_medians": {s: r(t1[f"dice_vs_{s}"].median() - d2[f"dice_vs_{s}"].median())
                                  for s in SEQS},
        "median_paired_difference": {s: r(gap[s].median()) for s in SEQS},
        "wilcoxon_p": {s: float(wilcoxon(t1.loc[gap.index, f"dice_vs_{s}"],
                                         d2.loc[gap.index, f"dice_vs_{s}"]).pvalue) for s in SEQS},
        "friedman_gap_chi2": r(frg.statistic, 2), "friedman_gap_p": float(frg.pvalue),
        "median_3d_3dp": {s: r(t1[f"dice_vs_{s}"].median(), 3) for s in SEQS},
        "median_2d_3dp": {s: r(d2[f"dice_vs_{s}"].median(), 3) for s in SEQS},
        "shift_over_largest_config_difference": r(
            (t1["dice_vs_T1WI"].median() - t1["dice_vs_CE-T1WI"].median())
            / max(abs(t1[f"dice_vs_{s}"].median() - d2[f"dice_vs_{s}"].median()) for s in SEQS), 1)}
    s2 = T / "multireference_evaluation_seed2.csv"
    if s2.exists():
        e2 = pd.read_csv(s2, dtype={"patient_id": str})
        e2["patient_id"] = e2.patient_id.str.zfill(3)
        e2 = e2.set_index("patient_id").loc[t1.index]
        gs = pd.DataFrame({s: t1[f"dice_vs_{s}"] - e2[f"dice_vs_{s}"] for s in SEQS}).dropna()
        N["seed_3d"] = {
            "n": int(len(gs)),
            "median_seed2": {s: r(e2[f"dice_vs_{s}"].median()) for s in SEQS},
            "difference_of_medians": {s: r(t1[f"dice_vs_{s}"].median() - e2[f"dice_vs_{s}"].median())
                                      for s in SEQS},
            "wilcoxon_p": {s: float(wilcoxon(t1.loc[gs.index, f"dice_vs_{s}"],
                                             e2.loc[gs.index, f"dice_vs_{s}"]).pvalue) for s in SEQS},
            "median_abs_patient_difference": {s: r(gs[s].abs().median()) for s in SEQS},
            "shift_over_largest_seed_difference": r(
                (t1["dice_vs_T1WI"].median() - t1["dice_vs_CE-T1WI"].median())
                / max(abs(t1[f"dice_vs_{s}"].median() - e2[f"dice_vs_{s}"].median()) for s in SEQS), 1)}

    # ------------------------------------------------------------ stratification
    pe_all = json.loads((T / "primary_experiments.json").read_text())
    N["stratification"] = {**pe_all["E4_clinical_impact"], "kappa": ra["kappa_tertile"],
                           "kappa_mean": ra["kappa_mean"]}

    # ------------------------------------------------------ supplementary items
    spp = pd.read_csv(T / "scanner_per_patient.csv", dtype={"patient_id": str})
    spp = spp.merge(mcs[["patient_id", "spread_pct_Vol"]], on="patient_id")
    grp = {f: g.spread_pct_Vol for f, g in spp.groupby("field_t")}
    N["supp_scanner"] = {"median_spread_by_field": {str(k): r(v.median(), 1) for k, v in grp.items()},
                         "kw_field": pe_all["E3_scanner"]["field_t"],
                         "kw_vendor": pe_all["E3_scanner"]["manufacturer"],
                         "kw_field_recomputed_H": r(kruskal(*grp.values()).statistic, 2)}
    N["supp_instability"] = json.loads((T / "instability_prediction.json").read_text())
    N["supp_descriptors"] = {"summary": ms.to_dict(orient="records"),
                             "tstage": pe_all["E6_descriptors"]}

    out = T / "manuscript_numbers.json"
    out.write_text(json.dumps(N, indent=2, default=float))
    print(json.dumps(N, indent=2, default=float)[:20000])
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
