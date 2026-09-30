#!/usr/bin/env python3
"""Primary experiments: variance decomposition, agreement, scanner effects, clinical impact.

Everything here is model-free. It uses the dataset's own released measurements and
the two released reader annotations, so no result depends on a segmentation model
we trained, and none can be dismissed as an artefact of our pipeline.

    E1  Variance decomposition. A linear mixed model on log-volume separating
        patient, sequence, reader and residual variance. This is the rigorous
        form of the paper's central claim.
    E2  Reader agreement on volume: ICC(2,1), Bland-Altman bias and limits.
    E3  Scanner effects: does cross-sequence spread depend on field strength or
        vendor? Now answerable because field strength was recovered from DICOM.
    E4  Clinical impact: how often does the choice of sequence move a patient
        across a volume-based risk threshold? This is the "so what" of the paper.
    E5  Prognostic sensitivity: does the association between tumour volume and
        progression-free survival depend on which sequence was measured?
        Clearly labelled as exploratory - the dataset ships NO event indicator.
    E6  Measurement comparison: volume vs maximum diameter as the reported
        descriptor, including agreement with T stage.

Usage:
    python3 scripts/08_primary_experiments.py
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import load_config  # noqa: E402

warnings.filterwarnings("ignore")

SEQS = ["T1", "T2", "CET1"]
LABEL = {"T1": "T1WI", "T2": "T2WI", "CET1": "CE-T1WI"}
ROI_STEM = {"T1": "ROI-T1", "T2": "ROI-T2", "CET1": "ROI-CE-T1"}
PARAMS = ["SA", "Vol", "MaxD", "SReg"]

OUT: dict = {}


def hdr(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# ---------------------------------------------------------------- loading
def read_morphology(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", skiprows=1)
    df = df.iloc[:, : 1 + len(SEQS) * len(PARAMS)]
    df.columns = ["patient_id"] + [f"{s}_{p}" for s in SEQS for p in PARAMS]
    df["patient_id"] = pd.to_numeric(df["patient_id"], errors="coerce")
    df = df.dropna(subset=["patient_id"])
    df["patient_id"] = df["patient_id"].astype(int).map(lambda v: f"{v:03d}")
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # mm^3 -> cm^3, mm^2 -> cm^2, mm -> cm so units match the descriptor's Table 2
    for s in SEQS:
        df[f"{s}_Vol"] /= 1000.0
        df[f"{s}_SA"] /= 100.0
        df[f"{s}_MaxD"] /= 10.0
    return df


def read_diagnoses(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", skiprows=1)
    df.columns = [str(c).strip() for c in df.columns]
    df["patient_id"] = pd.to_numeric(df["ID"], errors="coerce")
    df = df.dropna(subset=["patient_id"])
    df["patient_id"] = df["patient_id"].astype(int).map(lambda v: f"{v:03d}")
    df["stage"] = (df["Stage"].astype(str).str.strip().str.upper()
                   .str.replace("IVA", "IVa", regex=False)
                   .str.replace("IVB", "IVb", regex=False))
    for c in ("T", "N", "M"):
        df[c.lower() + "_stage"] = pd.to_numeric(df[c], errors="coerce")
    pfs_col = next(c for c in df.columns if c.lower().startswith("pfs"))
    df["pfs_months"] = pd.to_numeric(df[pfs_col], errors="coerce")
    # No event column exists anywhere in the release. Events can only be inferred
    # from an observation time below the 60-month administrative censoring point,
    # which cannot separate progression from loss to follow-up. Flagged, never hidden.
    df["event_inferred"] = np.where(df["pfs_months"].isna(), np.nan,
                                    (df["pfs_months"] < 60).astype(float))
    df["age_num"] = pd.to_numeric(df.get("age"), errors="coerce")
    return df


def load_reader_volumes(seg_root: Path, rel_root: Path) -> pd.DataFrame:
    """Volumes in cm^3 for both readers on the 30-patient reliability subset."""
    import nibabel as nib

    rows = []
    for pdir in sorted(p for p in rel_root.iterdir() if p.is_dir()):
        pid = pdir.name
        for key, stem in ROI_STEM.items():
            p2 = pdir / f"{stem}.nii"
            p1 = seg_root / pid / f"{stem}.nii"
            if not (p1.exists() and p2.exists()):
                continue
            try:
                v = []
                for p in (p1, p2):
                    im = nib.load(str(p))
                    a = np.asanyarray(im.dataobj) > 0
                    z = im.header.get_zooms()[:3]
                    v.append(float(a.sum()) * float(np.prod(z)) / 1000.0)
            except Exception as exc:  # noqa: BLE001
                print(f"  WARN {pid}/{stem}: {exc}")
                continue
            rows.append({"patient_id": pid, "sequence": LABEL[key],
                         "reader1_cm3": v[0], "reader2_cm3": v[1]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ E1
def e1_variance_decomposition(rv: pd.DataFrame) -> dict:
    hdr("E1  Variance decomposition of tumour volume")
    if rv.empty:
        print("  no paired reader data")
        return {}

    long = pd.concat([
        rv.assign(reader="R1", vol=rv.reader1_cm3),
        rv.assign(reader="R2", vol=rv.reader2_cm3),
    ])[["patient_id", "sequence", "reader", "vol"]]
    long = long[long.vol > 0].copy()
    long["log_vol"] = np.log(long.vol)
    print(f"  observations: {len(long)}  "
          f"({long.patient_id.nunique()} patients x {long.sequence.nunique()} sequences x 2 readers)")

    import statsmodels.formula.api as smf

    # Random intercept for patient (tumours differ hugely in size, and that
    # between-patient variance is not what we are asking about); sequence and
    # reader enter as fixed effects so their contributions are separable.
    md = smf.mixedlm("log_vol ~ C(sequence) + C(reader)", long, groups=long["patient_id"])
    fit = md.fit(reml=True)
    var_patient = float(fit.cov_re.iloc[0, 0])
    var_resid = float(fit.scale)

    # Variance attributable to each fixed factor = variance of its fitted
    # contribution across the observed design.
    def factor_var(term_prefix: str) -> float:
        cols = [c for c in fit.params.index if c.startswith(term_prefix)]
        if not cols:
            return 0.0
        contrib = np.zeros(len(long))
        for c in cols:
            level = c.split("[T.")[-1].rstrip("]")
            col = "sequence" if "sequence" in c else "reader"
            contrib += (long[col].values == level) * fit.params[c]
        return float(np.var(contrib))

    var_seq = factor_var("C(sequence)")
    var_reader = factor_var("C(reader)")
    total = var_patient + var_seq + var_reader + var_resid

    print(f"\n  {'component':<22}{'variance':>12}{'% of total':>12}")
    comps = [("between patients", var_patient), ("sequence", var_seq),
             ("reader", var_reader), ("residual", var_resid)]
    for name, v in comps:
        print(f"  {name:<22}{v:>12.5f}{100 * v / total:>11.1f}%")

    ratio = var_seq / var_reader if var_reader > 1e-12 else float("inf")
    print(f"\n  sequence variance / reader variance = {ratio:.1f}x")
    print("  (log scale, so these are proportional effects on volume)")

    # Coefficients: how much each sequence shifts volume relative to CE-T1WI.
    print("\n  fixed effects (log-volume; exp() = multiplicative effect on volume)")
    for c in fit.params.index:
        if c.startswith("C("):
            lvl = c.split("[T.")[-1].rstrip("]")
            print(f"    {lvl:<10} beta {fit.params[c]:+.4f}  "
                  f"x{np.exp(fit.params[c]):.3f}  p={fit.pvalues[c]:.3g}")

    return {
        "n_obs": int(len(long)),
        "var_patient": var_patient, "var_sequence": var_seq,
        "var_reader": var_reader, "var_residual": var_resid,
        "pct_sequence": 100 * var_seq / total, "pct_reader": 100 * var_reader / total,
        "sequence_over_reader": ratio,
    }


# ------------------------------------------------------------------ E2
def e2_reader_agreement(rv: pd.DataFrame) -> pd.DataFrame:
    hdr("E2  Reader agreement on tumour volume (ICC and Bland-Altman)")
    if rv.empty:
        return pd.DataFrame()

    def icc21(a: np.ndarray, b: np.ndarray) -> float:
        """ICC(2,1), two-way random effects, absolute agreement, single measure."""
        m = np.vstack([a, b]).T
        n, k = m.shape
        gm = m.mean()
        ms_r = k * ((m.mean(axis=1) - gm) ** 2).sum() / (n - 1)
        ms_c = n * ((m.mean(axis=0) - gm) ** 2).sum() / (k - 1)
        ms_e = ((m - m.mean(axis=1, keepdims=True) - m.mean(axis=0, keepdims=True) + gm) ** 2).sum() / ((n - 1) * (k - 1))
        denom = ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n
        return float((ms_r - ms_e) / denom) if denom else float("nan")

    rows = []
    for seq, g in rv.groupby("sequence"):
        a, b = g.reader1_cm3.values, g.reader2_cm3.values
        # Bland-Altman on the log scale: volume error is proportional, not additive,
        # so limits of agreement are reported as ratios.
        ok = (a > 0) & (b > 0)
        d = np.log(a[ok]) - np.log(b[ok])
        bias, sd = float(d.mean()), float(d.std(ddof=1))
        rows.append({
            "sequence": seq, "n": int(len(g)),
            "icc21": round(icc21(a, b), 4),
            "mean_vol_r1": round(float(a.mean()), 2),
            "mean_vol_r2": round(float(b.mean()), 2),
            "ba_bias_ratio": round(float(np.exp(bias)), 4),
            "ba_loa_low": round(float(np.exp(bias - 1.96 * sd)), 4),
            "ba_loa_high": round(float(np.exp(bias + 1.96 * sd)), 4),
            "median_abs_pct_diff": round(float(np.median(
                np.abs(a - b) / ((a + b) / 2) * 100)), 2),
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("\n  ICC(2,1) above 0.90 is conventionally 'excellent'.")
    print("  Bland-Altman ratios: 1.00 = no systematic bias between readers;")
    print("  the limits say how far two readers can differ on the same tumour.")
    return df


# ------------------------------------------------------------------ E3
def e3_scanner_effects(m: pd.DataFrame, scan: pd.DataFrame) -> dict:
    hdr("E3  Does cross-sequence disagreement depend on the scanner?")
    d = m.merge(scan, on="patient_id", how="left")
    V = d[[f"{s}_Vol" for s in SEQS]]
    d["spread_pct"] = (V.max(axis=1) - V.min(axis=1)) / V.mean(axis=1) * 100

    from scipy import stats

    out = {}
    for key, label in (("field_t", "Field strength (T)"), ("manufacturer", "Vendor")):
        if key not in d.columns:
            continue
        print(f"\n  --- by {label} ---")
        groups, names = [], []
        for name, g in d.groupby(key):
            if len(g) >= 5:
                groups.append(g.spread_pct.dropna().values)
                names.append(name)
            print(f"    {str(name):<26} n={len(g):>3}  median spread {g.spread_pct.median():>5.1f}%  "
                  f"IQR {g.spread_pct.quantile(.25):.1f}-{g.spread_pct.quantile(.75):.1f}")
        if len(groups) >= 2:
            h = stats.kruskal(*groups)
            print(f"    Kruskal-Wallis across {len(groups)} groups: H={h.statistic:.2f}, p={h.pvalue:.3g}")
            out[key] = {"groups": names, "H": float(h.statistic), "p": float(h.pvalue),
                        "n": [int(len(g)) for g in groups]}

    # Absolute volume by field strength: a weaker magnet could bias size upward.
    print("\n  --- median tumour volume (cm3) by field strength ---")
    piv = d.groupby("field_t")[[f"{s}_Vol" for s in SEQS]].median().round(2)
    piv.columns = [LABEL[s] for s in SEQS]
    print(piv.to_string())
    return out


# ------------------------------------------------------------------ E4
def e4_clinical_impact(m: pd.DataFrame) -> dict:
    hdr("E4  Clinical impact: does the sequence change the patient's risk group?")
    V = m[[f"{s}_Vol" for s in SEQS]].copy()
    V.columns = [LABEL[s] for s in SEQS]

    res = {}
    # Thresholds: the cohort median (a data-driven split) plus round values in the
    # range used for NPC tumour-volume risk stratification in the literature.
    thresholds = {
        "cohort median": float(V.median().median()),
        "15 cm3": 15.0, "20 cm3": 20.0, "30 cm3": 30.0,
    }
    print(f"  {'threshold':<16}{'discordant patients':>22}{'% of cohort':>14}")
    for name, thr in thresholds.items():
        above = V.gt(thr)
        disc = above.nunique(axis=1) > 1  # not all sequences agree
        n = int(disc.sum())
        print(f"  {name:<16}{n:>16} / {len(V):<4}{100 * n / len(V):>13.1f}%")
        res[name] = {"threshold_cm3": round(thr, 2), "n_discordant": n,
                     "pct": round(100 * n / len(V), 1)}

    # Tertile-based staging surrogate: three risk bands rather than a binary cut.
    print("\n  Tertile risk bands (low/mid/high volume), assigned per sequence:")
    bands = V.apply(lambda c: pd.qcut(c, 3, labels=["low", "mid", "high"]))
    disc3 = bands.nunique(axis=1) > 1
    n3 = int(disc3.sum())
    print(f"    patients whose band changes with sequence: {n3}/{len(V)} ({100 * n3 / len(V):.1f}%)")
    worst = (bands == "low").any(axis=1) & (bands == "high").any(axis=1)
    print(f"    patients spanning LOW and HIGH depending on sequence: {int(worst.sum())}")
    res["tertile_bands"] = {"n_discordant": n3, "pct": round(100 * n3 / len(V), 1),
                            "n_low_to_high": int(worst.sum())}

    # The same question for maximum diameter, the more reproducible descriptor.
    D = m[[f"{s}_MaxD" for s in SEQS]].copy()
    D.columns = [LABEL[s] for s in SEQS]
    dband = D.apply(lambda c: pd.qcut(c, 3, labels=["low", "mid", "high"]))
    nd = int((dband.nunique(axis=1) > 1).sum())
    print(f"\n  Same tertile analysis using MAXIMUM DIAMETER instead of volume:")
    print(f"    patients whose band changes with sequence: {nd}/{len(D)} ({100 * nd / len(D):.1f}%)")
    delta = nd - n3
    verdict = (f"diameter reclassifies {delta} MORE patients than volume"
               if delta > 0 else
               f"diameter reclassifies {-delta} fewer patients than volume"
               if delta < 0 else "identical reclassification")
    print(f"    -> {verdict}")
    if delta > 0:
        print("       Note: lower relative spread does NOT imply fewer reclassifications.")
        print("       Diameter varies less within a patient but also spans a narrower")
        print("       range across patients, so its tertile boundaries sit closer")
        print("       together. Reproducibility and discriminative range trade off.")
    res["tertile_bands_diameter"] = {"n_discordant": nd, "pct": round(100 * nd / len(D), 1)}
    return res


# ------------------------------------------------------------------ E5
def e5_prognostic_sensitivity(m: pd.DataFrame, diag: pd.DataFrame) -> dict:
    hdr("E5  Does the prognostic signal depend on which sequence was measured?")
    print("  EXPLORATORY. The release contains no event indicator; events are")
    print("  inferred as PFS < 60 months, which cannot separate progression from")
    print("  loss to follow-up. Treat every number here as a sensitivity analysis.")

    d = m.merge(diag[["patient_id", "pfs_months", "event_inferred", "stage",
                      "t_stage", "n_stage", "age_num"]], on="patient_id", how="left")
    d = d.dropna(subset=["pfs_months", "event_inferred"])
    n_ev = int(d.event_inferred.sum())
    print(f"\n  analysable: {len(d)} patients, {n_ev} inferred events "
          f"({100 * n_ev / len(d):.1f}%)")
    if n_ev < 20:
        print("  too few events; skipping")
        return {}

    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index

    rows = []
    for s in SEQS:
        sub = d[["pfs_months", "event_inferred", f"{s}_Vol"]].dropna().copy()
        sub["log_vol"] = np.log(sub[f"{s}_Vol"])
        cph = CoxPHFitter()
        cph.fit(sub[["pfs_months", "event_inferred", "log_vol"]],
                duration_col="pfs_months", event_col="event_inferred")
        hr = float(np.exp(cph.params_["log_vol"]))
        lo, hi = np.exp(cph.confidence_intervals_.loc["log_vol"].values)
        p = float(cph.summary.loc["log_vol", "p"])
        ci = concordance_index(sub.pfs_months, -cph.predict_partial_hazard(sub), sub.event_inferred)
        rows.append({"sequence": LABEL[s], "n": len(sub), "events": int(sub.event_inferred.sum()),
                     "HR_per_log_cm3": round(hr, 3), "ci_low": round(float(lo), 3),
                     "ci_high": round(float(hi), 3), "p": p, "c_index": round(float(ci), 4)})
    df = pd.DataFrame(rows)
    print("\n  Univariable Cox on log tumour volume, fitted separately per sequence:")
    print(df.to_string(index=False))

    sig = df[df.p < 0.05].sequence.tolist()
    ns = df[df.p >= 0.05].sequence.tolist()
    print(f"\n  significant at 0.05: {sig or 'none'}")
    print(f"  not significant:     {ns or 'none'}")
    if sig and ns:
        print("  -> the qualitative conclusion CHANGES with the sequence measured.")
        print("     This is the clinical consequence of the measurement problem.")
    spread = (df.HR_per_log_cm3.max() - df.HR_per_log_cm3.min()) / df.HR_per_log_cm3.mean() * 100
    print(f"  hazard-ratio spread across sequences: {spread:.1f}% of the mean")

    # Same model on maximum diameter, to see whether the more reproducible
    # descriptor also gives a more stable prognostic estimate.
    rows_d = []
    for s in SEQS:
        sub = d[["pfs_months", "event_inferred", f"{s}_MaxD"]].dropna().copy()
        sub["log_d"] = np.log(sub[f"{s}_MaxD"])
        cph = CoxPHFitter()
        cph.fit(sub[["pfs_months", "event_inferred", "log_d"]],
                duration_col="pfs_months", event_col="event_inferred")
        rows_d.append({"sequence": LABEL[s],
                       "HR_per_log_cm": round(float(np.exp(cph.params_["log_d"])), 3),
                       "p": float(cph.summary.loc["log_d", "p"])})
    dd = pd.DataFrame(rows_d)
    print("\n  Same, using maximum diameter:")
    print(dd.to_string(index=False))
    spread_d = (dd.HR_per_log_cm.max() - dd.HR_per_log_cm.min()) / dd.HR_per_log_cm.mean() * 100
    print(f"  hazard-ratio spread across sequences: {spread_d:.1f}% of the mean")

    return {"volume": df.to_dict("records"), "diameter": dd.to_dict("records"),
            "hr_spread_pct_volume": round(float(spread), 1),
            "hr_spread_pct_diameter": round(float(spread_d), 1),
            "n_events_inferred": n_ev}


# ------------------------------------------------------------------ E6
def e6_descriptor_comparison(m: pd.DataFrame, diag: pd.DataFrame) -> dict:
    hdr("E6  Volume versus maximum diameter as the reported descriptor")
    from scipy import stats

    out = {}
    d = m.merge(diag[["patient_id", "t_stage"]], on="patient_id", how="left")
    print(f"  {'descriptor':<20}{'median spread':>15}{'Spearman rho vs T stage':>26}")
    for p, name in (("Vol", "Volume"), ("MaxD", "Max diameter"),
                    ("SA", "Surface area"), ("SReg", "Surface regularity")):
        X = d[[f"{s}_{p}" for s in SEQS]]
        spread = (X.max(axis=1) - X.min(axis=1)) / X.mean(axis=1).abs() * 100
        # Correlate the per-sequence mean with T stage; a descriptor that tracks
        # stage better AND varies less is the better thing to report.
        mean_val = X.mean(axis=1)
        ok = d.t_stage.notna() & mean_val.notna()
        rho, pv = stats.spearmanr(mean_val[ok], d.t_stage[ok])
        print(f"  {name:<20}{spread.median():>14.1f}%{rho:>18.3f} (p={pv:.2g})")
        out[name] = {"median_spread_pct": round(float(spread.median()), 2),
                     "spearman_rho_tstage": round(float(rho), 4), "p": float(pv)}
    print("\n  The descriptor to prefer is the one that is both reproducible across")
    print("  sequences and still tracks stage.")
    return out


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--allow-missing-roots", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, verify=True, allow_missing=True)
    root = cfg.primary_data_root
    tables = cfg.out("tables_dir")

    m = read_morphology(root / "morphological_parameters.csv")
    diag = read_diagnoses(root / "info_diagnoses.csv")
    print(f"morphology {len(m)} patients | diagnoses {len(diag)} patients")

    scan_path = tables / "scanner_per_patient.csv"
    scan = pd.read_csv(scan_path, dtype={"patient_id": str}) if scan_path.exists() else pd.DataFrame()
    if not scan.empty:
        print(f"scanner metadata {len(scan)} patients")

    rv = load_reader_volumes(root / "MRI-Segments", cfg.reliability_root)
    print(f"paired reader volumes {len(rv)} masks over {rv.patient_id.nunique() if len(rv) else 0} patients")

    OUT["E1_variance"] = e1_variance_decomposition(rv)
    e2 = e2_reader_agreement(rv)
    OUT["E2_agreement"] = e2.to_dict("records") if len(e2) else []
    if not scan.empty:
        OUT["E3_scanner"] = e3_scanner_effects(m, scan)
    OUT["E4_clinical_impact"] = e4_clinical_impact(m)
    OUT["E5_prognostic"] = e5_prognostic_sensitivity(m, diag)
    OUT["E6_descriptors"] = e6_descriptor_comparison(m, diag)

    if len(rv):
        rv.to_csv(tables / "reader_volumes_paired.csv", index=False)
    if len(e2):
        e2.to_csv(tables / "reader_agreement_icc_ba.csv", index=False)
    with open(tables / "primary_experiments.json", "w", encoding="utf-8") as fh:
        json.dump(OUT, fh, indent=2, default=float)
    print(f"\n\nWrote {tables}/primary_experiments.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
