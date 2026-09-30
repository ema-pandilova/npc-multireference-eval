#!/usr/bin/env python3
"""Agreement between per-sequence references, expressed in clinician-facing statistics.

The overlap metrics used in the segmentation literature (Dice, Jaccard) are not the
vocabulary of the clinical journals where this work has to be read. A survey of
1384 abstracts in the target journal found no occurrence of "Dice"; agreement
there is reported as the intraclass correlation coefficient, Bland-Altman limits,
and Cohen's kappa. This script therefore restates the reference-standard
disagreement in those terms, using tumour volume as the measurand.

Two things make this analysis stronger than the Dice comparison rather than merely
a translation of it. It uses the volumes released with the dataset, so it does not
depend on our registration, and it covers all 277 patients rather than the subset
where registration converged.

    ICC(2,1)        two-way random effects, absolute agreement, single measure.
                    Computed pairwise between sequences and across all three.
    Bland-Altman    on the log scale, because volume error is proportional rather
                    than additive, so limits are reported as ratios.
    Kappa           on the tertile risk band each sequence assigns, which is the
                    clinically consequential version of the disagreement.

Outputs:
    results/tables/reference_agreement_clinical.csv
    results/tables/reference_agreement_clinical.json

Usage:
    python3 scripts/17_reference_agreement_clinical.py
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

SEQS = ["T1", "T2", "CET1"]
LABEL = {"T1": "T1WI", "T2": "T2WI", "CET1": "CE-T1WI"}
PARAMS = ["SA", "Vol", "MaxD", "SReg"]


def read_morphology(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", skiprows=1)
    df = df.iloc[:, : 1 + len(SEQS) * len(PARAMS)]
    df.columns = ["patient_id"] + [f"{s}_{p}" for s in SEQS for p in PARAMS]
    df["patient_id"] = pd.to_numeric(df["patient_id"], errors="coerce")
    df = df.dropna(subset=["patient_id"])
    df["patient_id"] = df["patient_id"].astype(int).map(lambda v: f"{v:03d}")
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for s in SEQS:
        df[f"{s}_Vol"] /= 1000.0          # mm^3 -> cm^3
    return df


def icc21(m: np.ndarray) -> tuple[float, float, float]:
    """ICC(2,1) with an approximate 95% confidence interval.

    m is subjects x raters. Two-way random effects, absolute agreement, single
    measure, which is the form appropriate when the "raters" (here, sequences)
    are themselves a sample and systematic differences between them count as
    disagreement rather than being calibrated away.
    """
    n, k = m.shape
    gm = m.mean()
    ms_r = k * ((m.mean(axis=1) - gm) ** 2).sum() / (n - 1)
    ms_c = n * ((m.mean(axis=0) - gm) ** 2).sum() / (k - 1)
    ms_e = ((m - m.mean(axis=1, keepdims=True) - m.mean(axis=0, keepdims=True) + gm) ** 2).sum() \
        / ((n - 1) * (k - 1))
    denom = ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n
    icc = (ms_r - ms_e) / denom if denom else float("nan")

    # Bootstrap CI over subjects; analytic CIs for ICC(2,1) are awkward and this
    # is transparent about where the uncertainty comes from.
    rng = np.random.default_rng(20260727)
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        b = m[idx]
        try:
            g = b.mean()
            r = k * ((b.mean(axis=1) - g) ** 2).sum() / (n - 1)
            c = n * ((b.mean(axis=0) - g) ** 2).sum() / (k - 1)
            e = ((b - b.mean(axis=1, keepdims=True) - b.mean(axis=0, keepdims=True) + g) ** 2).sum() \
                / ((n - 1) * (k - 1))
            d = r + (k - 1) * e + k * (c - e) / n
            if d:
                boots.append((r - e) / d)
        except Exception:  # noqa: BLE001
            continue
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return float(icc), float(lo), float(hi)


def bland_altman_log(a: np.ndarray, b: np.ndarray) -> dict:
    """Bland-Altman on log volume, reported as ratios."""
    ok = (a > 0) & (b > 0)
    d = np.log(a[ok]) - np.log(b[ok])
    bias, sd = float(d.mean()), float(d.std(ddof=1))
    return {
        "n": int(ok.sum()),
        "bias_ratio": round(float(np.exp(bias)), 4),
        "loa_lower": round(float(np.exp(bias - 1.96 * sd)), 4),
        "loa_upper": round(float(np.exp(bias + 1.96 * sd)), 4),
        "mean_abs_pct_diff": round(float(np.mean(np.abs(a[ok] - b[ok]) / ((a[ok] + b[ok]) / 2) * 100)), 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--allow-missing-roots", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, verify=True, allow_missing=True)
    m = read_morphology(cfg.primary_data_root / "morphological_parameters.csv")
    V = m[[f"{s}_Vol" for s in SEQS]].dropna()
    arr = V.to_numpy()

    print("=" * 78)
    print("REFERENCE AGREEMENT IN CLINICIAN-FACING STATISTICS")
    print("=" * 78)
    print(f"tumour volume, {len(V)} patients, three per-sequence reference standards\n")

    out: dict = {"n_patients": int(len(V))}

    icc, lo, hi = icc21(arr)
    print(f"ICC(2,1) across all three references: {icc:.4f} (95% CI {lo:.4f} to {hi:.4f})")
    print("  Read this alongside the Bland-Altman limits below, not on its own. ICC is a")
    print("  ratio of between-subject to total variance, so it is inflated when subjects")
    print("  differ widely, and tumour volume here spans roughly 0.4 to 84 cm^3. A high")
    print("  ICC therefore does not establish that the references are interchangeable;")
    print("  the limits of agreement are the quantity that answers that question.")
    out["icc_all_three"] = {"icc": round(icc, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4)}

    print("\nPairwise agreement:")
    print(f"  {'pair':<20}{'ICC(2,1)':>10}{'95% CI':>20}{'BA bias':>10}{'BA limits':>20}{'mean |diff|':>13}")
    rows, pair_out = [], {}
    from itertools import combinations
    for a, b in combinations(SEQS, 2):
        x, y = V[f"{a}_Vol"].to_numpy(), V[f"{b}_Vol"].to_numpy()
        i2, l2, h2 = icc21(np.column_stack([x, y]))
        ba = bland_altman_log(x, y)
        name = f"{LABEL[a]} vs {LABEL[b]}"
        ci = "%.3f-%.3f" % (l2, h2)
        loa = "%.2f-%.2f" % (ba["loa_lower"], ba["loa_upper"])
        print("  %-20s%10.4f%20s%10.3f%20s%12.1f%%" % (
            name, i2, ci, ba["bias_ratio"], loa, ba["mean_abs_pct_diff"]))
        rows.append({"pair": name, "icc": round(i2, 4), "icc_lo": round(l2, 4), "icc_hi": round(h2, 4), **ba})
        pair_out[name] = {"icc": round(i2, 4), **ba}
    out["pairwise"] = pair_out

    # Clinically consequential form: does the tertile risk band agree?
    print("\nAgreement on tertile volume risk band (Cohen's kappa, unweighted):")
    bands = V.apply(lambda c: pd.qcut(c, 3, labels=[0, 1, 2]).astype(int))
    try:
        from sklearn.metrics import cohen_kappa_score

        kap = {}
        for a, b in combinations(SEQS, 2):
            k = cohen_kappa_score(bands[f"{a}_Vol"], bands[f"{b}_Vol"])
            kap[f"{LABEL[a]} vs {LABEL[b]}"] = round(float(k), 4)
            print(f"  {LABEL[a]} vs {LABEL[b]:<10} kappa = {k:.4f}")
        out["kappa_tertile"] = kap
        mean_k = float(np.mean(list(kap.values())))
        print(f"  mean kappa {mean_k:.4f}"
              f"  ({'substantial' if mean_k >= 0.61 else 'moderate' if mean_k >= 0.41 else 'fair'}"
              " agreement by the Landis and Koch bands)")
        out["kappa_mean"] = round(mean_k, 4)
    except ImportError:
        print("  scikit-learn unavailable; skipped")

    tables = cfg.out("tables_dir")
    pd.DataFrame(rows).to_csv(tables / "reference_agreement_clinical.csv", index=False)
    with open(tables / "reference_agreement_clinical.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {tables}/reference_agreement_clinical.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
