#!/usr/bin/env python3
"""Cross-sequence variability of tumour morphology, from the dataset's own tables.

This is the study's headline analysis and it needs no model, no re-segmentation
and no training. `morphological_parameters.csv` ships volume, surface area,
maximum diameter and surface regularity computed **separately for each of the
three sequences** for all 277 patients. Those per-sequence values disagree, and
the dataset descriptor never reports by how much.

Reading the authors' own numbers rather than recomputing them is deliberate: it
removes any dispute about our segmentation or our meshing. The medians this
script reproduces match the descriptor's Table 2 exactly, which is the check that
the parsing is faithful.

Both CSVs are semicolon-delimited with a decorative first row, and
`morphological_parameters.csv` has a two-level header (sequence group, then
parameter), so they need explicit handling rather than a bare read_csv.

Outputs:
    results/tables/morphology_cross_sequence.csv    per-patient spreads
    results/tables/morphology_variability_summary.csv  per-feature statistics
    results/tables/clinical_quality_issues.csv     data-quality findings
    results/figures/morphology_variability.png     figure

Usage:
    python3 scripts/06_morphology_variability.py
    python3 scripts/06_morphology_variability.py --primary-data ../dataset_v2/primary_data/data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import load_config  # noqa: E402

SEQS = ["T1", "T2", "CET1"]
SEQ_LABEL = {"T1": "T1WI", "T2": "T2WI", "CET1": "CE-T1WI"}
PARAMS = ["SA", "Vol", "MaxD", "SReg"]
PARAM_LABEL = {
    "Vol": "Volume (mm^3)",
    "SA": "Surface area (mm^2)",
    "MaxD": "Max diameter (mm)",
    "SReg": "Surface regularity",
}


def read_morphology(path: Path) -> pd.DataFrame:
    """Read morphological_parameters.csv, flattening its two-level header."""
    df = pd.read_csv(path, sep=";", skiprows=1)
    expected = 1 + len(SEQS) * len(PARAMS)
    if df.shape[1] < expected:
        raise ValueError(
            f"{path.name}: expected at least {expected} columns, found {df.shape[1]}. "
            "The file layout has changed; inspect the header before trusting any result."
        )
    df = df.iloc[:, :expected]
    df.columns = ["patient_id"] + [f"{s}_{p}" for s in SEQS for p in PARAMS]
    df["patient_id"] = pd.to_numeric(df["patient_id"], errors="coerce")
    df = df.dropna(subset=["patient_id"])
    df["patient_id"] = df["patient_id"].astype(int).map(lambda v: f"{v:03d}")
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def read_diagnoses(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read info_diagnoses.csv and collect its data-quality issues.

    Returns (clean table, issues table). Values are normalised for analysis, but
    every normalisation is recorded as a finding rather than applied silently.
    """
    df = pd.read_csv(path, sep=";", skiprows=1)
    df.columns = [str(c).strip() for c in df.columns]
    issues: list[dict] = []

    def note(col, kind, detail, n):
        issues.append({"column": col, "issue": kind, "detail": detail, "n_rows": n})

    df["patient_id"] = pd.to_numeric(df["ID"], errors="coerce")
    df = df.dropna(subset=["patient_id"])
    df["patient_id"] = df["patient_id"].astype(int).map(lambda v: f"{v:03d}")

    # Stage: case inconsistency ('iii' alongside 'III').
    if "Stage" in df.columns:
        raw = df["Stage"].astype(str).str.strip()
        bad = raw[raw != raw.str.upper().str.replace("IVA", "IVa").str.replace("IVB", "IVb")]
        lower = raw[raw.str.islower() & raw.str.len().le(4)]
        if len(lower):
            note("Stage", "inconsistent case", f"values {sorted(set(lower))} differ only in case", len(lower))
        df["stage_norm"] = (
            raw.str.upper().str.replace("IVA", "IVa", regex=False).str.replace("IVB", "IVb", regex=False)
        )

    # EBV-DNA / VCA-IgA: full-width Unicode parentheses mixed with ASCII.
    for col in ("EBV-DNA", "VCA-IgA"):
        if col not in df.columns:
            continue
        raw = df[col].astype(str).str.strip()
        fw = raw[raw.str.contains("（|）", regex=True)]
        if len(fw):
            note(col, "full-width Unicode parentheses",
                 f"{sorted(set(fw))} will not compare equal to the ASCII form", len(fw))
        df[f"{col}_norm"] = (
            raw.str.replace("（", "(", regex=False).str.replace("）", ")", regex=False)
        )

    # T stage: T0 is unusual for a primary NPC cohort.
    if "T" in df.columns:
        t = df["T"].astype(str).str.strip()
        if (t == "0").any():
            note("T", "T0 in a primary NPC cohort",
                 f"{int((t == '0').sum())} case(s) coded T0", int((t == "0").sum()))

    # PFS: months only, with no event indicator anywhere in the file.
    pfs_col = next((c for c in df.columns if c.lower().startswith("pfs")), None)
    if pfs_col:
        p = pd.to_numeric(df[pfs_col], errors="coerce")
        df["pfs_months"] = p
        n_missing = int(p.isna().sum())
        if n_missing:
            note(pfs_col, "missing values", f"{n_missing} patients have no PFS value", n_missing)
        note(pfs_col, "no event/censoring indicator",
             "the file contains time only; events must be INFERRED from PFS < 60, "
             "which cannot distinguish progression from loss to follow-up",
             int(p.notna().sum()))
        # Explicit, clearly-labelled inference. Never presented as ground truth.
        df["pfs_event_inferred"] = np.where(p.isna(), np.nan, (p < 60).astype(float))

    return df, pd.DataFrame(issues)


def spreads(df: pd.DataFrame, param: str) -> pd.Series:
    """Within-patient relative spread across the three sequences, in percent."""
    X = df[[f"{s}_{param}" for s in SEQS]]
    mu = X.mean(axis=1).abs()
    return (X.max(axis=1) - X.min(axis=1)) / mu.replace(0, np.nan) * 100


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--allow-missing-roots", action="store_true")
    ap.add_argument("--primary-data", default=None,
                    help="override primary_data/data directory holding the CSVs")
    args = ap.parse_args()

    cfg = load_config(args.config, verify=True, allow_missing=True)
    root = (
        Path(args.primary_data) if args.primary_data and Path(args.primary_data).is_absolute()
        else (cfg.repo_root / args.primary_data).resolve() if args.primary_data
        else cfg.primary_data_root
    )
    morph_path, diag_path = root / "morphological_parameters.csv", root / "info_diagnoses.csv"
    for p in (morph_path, diag_path):
        if not p.exists() or p.stat().st_size == 0:
            print(f"ERROR: {p} missing or empty. Stage primary_data.zip first.", file=sys.stderr)
            return 2

    print("=" * 78)
    print("CROSS-SEQUENCE MORPHOLOGY VARIABILITY")
    print("=" * 78)
    m = read_morphology(morph_path)
    diag, issues = read_diagnoses(diag_path)
    print(f"morphology : {len(m)} patients")
    print(f"diagnoses  : {len(diag)} patients")
    print()

    # -- fidelity check against the published Table 2 --------------------
    print("Parsing fidelity check (medians in cm^3 vs descriptor Table 2: 8.60 / 8.67 / 9.42)")
    med = {s: m[f"{s}_Vol"].median() / 1000.0 for s in SEQS}
    print("  computed: " + " / ".join(f"{med[s]:.2f}" for s in SEQS))
    ref = [8.60, 8.67, 9.42]
    ok = all(abs(med[s] - r) < 0.02 for s, r in zip(SEQS, ref))
    print(f"  {'MATCH — parsing is faithful' if ok else 'MISMATCH — do not trust downstream numbers'}")
    print()

    # -- per-feature variability -----------------------------------------
    from scipy import stats

    rows, per_patient = [], pd.DataFrame({"patient_id": m["patient_id"]})
    print(f"{'feature':<22}{'median%':>9}{'mean%':>8}{'IQR%':>14}{'max%':>8}{'>20%':>8}{'Friedman p':>13}")
    for p in PARAMS:
        sp = spreads(m, p)
        per_patient[f"spread_pct_{p}"] = sp.round(2)
        X = m[[f"{s}_{p}" for s in SEQS]].dropna()
        fr = stats.friedmanchisquare(*[X[c] for c in X.columns])
        rows.append(
            {
                "feature": PARAM_LABEL[p],
                "n": int(sp.notna().sum()),
                "median_spread_pct": round(float(sp.median()), 2),
                "mean_spread_pct": round(float(sp.mean()), 2),
                "iqr_low_pct": round(float(sp.quantile(0.25)), 2),
                "iqr_high_pct": round(float(sp.quantile(0.75)), 2),
                "max_spread_pct": round(float(sp.max()), 2),
                "n_over_10pct": int((sp > 10).sum()),
                "n_over_20pct": int((sp > 20).sum()),
                "n_over_30pct": int((sp > 30).sum()),
                "friedman_chi2": round(float(fr.statistic), 2),
                "friedman_p": float(fr.pvalue),
                **{f"median_{SEQ_LABEL[s]}": round(float(m[f'{s}_{p}'].median()), 4) for s in SEQS},
            }
        )
        print(f"{PARAM_LABEL[p]:<22}{sp.median():>9.1f}{sp.mean():>8.1f}"
              f"{f'{sp.quantile(.25):.1f}-{sp.quantile(.75):.1f}':>14}{sp.max():>8.1f}"
              f"{int((sp > 20).sum()):>8}{fr.pvalue:>13.2g}")
    summary = pd.DataFrame(rows)

    # -- pairwise tests on volume ----------------------------------------
    print("\nPairwise Wilcoxon on volume (Holm-corrected):")
    pairs = [("CET1", "T1"), ("CET1", "T2"), ("T2", "T1")]
    raw_p = [stats.wilcoxon(m[f"{a}_Vol"], m[f"{b}_Vol"]).pvalue for a, b in pairs]
    order = np.argsort(raw_p)
    holm = np.empty(len(raw_p))
    for rank, i in enumerate(order):
        holm[i] = min(1.0, raw_p[i] * (len(raw_p) - rank))
    holm = np.maximum.accumulate(holm[order])[np.argsort(order)]
    for (a, b), rp, hp in zip(pairs, raw_p, holm):
        print(f"  {SEQ_LABEL[a]:>8} vs {SEQL(b):<8} p={rp:.3g}  Holm p={hp:.3g}")

    # -- which sequence is extreme ---------------------------------------
    V = m[[f"{s}_Vol" for s in SEQS]]
    largest = V.idxmax(axis=1).str.replace("_Vol", "", regex=False).map(SEQ_LABEL)
    smallest = V.idxmin(axis=1).str.replace("_Vol", "", regex=False).map(SEQ_LABEL)
    print("\nSequence giving the LARGEST volume: "
          + ", ".join(f"{k} {v}" for k, v in largest.value_counts().items()))
    print("Sequence giving the SMALLEST volume: "
          + ", ".join(f"{k} {v}" for k, v in smallest.value_counts().items()))
    per_patient["largest_volume_sequence"] = largest.values
    per_patient["smallest_volume_sequence"] = smallest.values

    worst = per_patient.loc[per_patient["spread_pct_Vol"].idxmax()]
    wid = worst["patient_id"]
    wr = m[m.patient_id == wid].iloc[0]
    print(f"\nLargest-spread patient: {wid}  "
          + ", ".join(f"{SEQ_LABEL[s]} {wr[f'{s}_Vol']/1000:.2f}" for s in SEQS)
          + f" cm^3  spread {worst['spread_pct_Vol']:.0f}%")

    # -- clinical fields --------------------------------------------------
    print("\nClinical table")
    if "pfs_months" in diag.columns:
        p = diag["pfs_months"]
        n_ev = int((diag["pfs_event_inferred"] == 1).sum())
        print(f"  PFS present for {int(p.notna().sum())}/{len(diag)} patients "
              f"({int(p.isna().sum())} missing)")
        print(f"  censored at 60 months: {int((p == 60).sum())}  "
              f"inferred events (<60): {n_ev}")
        print(f"  -> at most {n_ev} events, so a Cox model supports roughly "
              f"{max(1, n_ev // 10)} covariates. No event column exists; this is an inference.")
    if len(issues):
        print(f"\n  {len(issues)} data-quality issues recorded:")
        for _, r in issues.iterrows():
            print(f"    [{r['column']}] {r['issue']}: {r['detail']} (n={r['n_rows']})")

    # -- outputs -----------------------------------------------------------
    tables, figs = cfg.out("tables_dir"), cfg.out("figures_dir")
    per_patient = per_patient.merge(
        diag[[c for c in ("patient_id", "stage_norm", "pfs_months", "pfs_event_inferred") if c in diag.columns]],
        on="patient_id", how="left",
    )
    per_patient.to_csv(tables / "morphology_cross_sequence.csv", index=False)
    summary.to_csv(tables / "morphology_variability_summary.csv", index=False)
    issues.to_csv(tables / "clinical_quality_issues.csv", index=False)
    make_figure(m, per_patient, summary, figs / "morphology_variability.png")
    print(f"\nWrote tables to {tables}\n      figure to {figs / 'morphology_variability.png'}")
    return 0


def SEQL(k: str) -> str:
    return SEQ_LABEL.get(k, k)


def make_figure(m: pd.DataFrame, per_patient: pd.DataFrame, summary: pd.DataFrame, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    # (a) paired volumes per sequence
    V = m[[f"{s}_Vol" for s in SEQS]] / 1000.0
    for i, row in V.iterrows():
        ax[0].plot([0, 1, 2], row.values, color="#888", alpha=0.18, lw=0.6)
    bp = ax[0].boxplot([V[c].dropna() for c in V.columns], positions=[0, 1, 2],
                       widths=0.5, showfliers=False, patch_artist=True)
    for patch, c in zip(bp["boxes"], ["#4c8ec9", "#5fb87a", "#d4694a"]):
        patch.set_facecolor(c)
        patch.set_alpha(0.65)
    ax[0].set_xticks([0, 1, 2])
    ax[0].set_xticklabels([SEQ_LABEL[s] for s in SEQS])
    ax[0].set_ylabel("Tumour volume (cm$^3$)")
    ax[0].set_yscale("log")
    ax[0].set_title(f"(a) Same patient, three sequences (n={len(V)})", fontsize=10)

    # (b) spread distribution per feature
    data = [per_patient[f"spread_pct_{p}"].dropna() for p in PARAMS]
    bp2 = ax[1].boxplot(data, labels=[PARAM_LABEL[p].split(" (")[0] for p in PARAMS],
                        showfliers=False, patch_artist=True, widths=0.55)
    for patch in bp2["boxes"]:
        patch.set_facecolor("#7a8ca8")
        patch.set_alpha(0.7)
    ax[1].axhline(20, color="#d4694a", ls="--", lw=1, label="20% spread")
    ax[1].set_ylabel("Within-patient spread across sequences (%)")
    ax[1].set_title("(b) Which measurement is most reproducible", fontsize=10)
    ax[1].legend(fontsize=8)
    ax[1].tick_params(axis="x", rotation=18)

    # (c) cumulative distribution of volume spread
    sp = per_patient["spread_pct_Vol"].dropna().sort_values()
    ax[2].plot(sp.values, np.linspace(0, 100, len(sp)), lw=2, color="#4c8ec9")
    # Stagger the annotation heights; at 10/20/30% the labels would otherwise
    # overlap into an illegible block.
    for (t, c), y in zip(((10, "#3f9e5f"), (20, "#c98a2e"), (30, "#c4553a")), (72, 46, 20)):
        frac = (sp > t).mean() * 100
        ax[2].axvline(t, color=c, ls="--", lw=1)
        ax[2].annotate(
            f"{frac:.0f}% exceed {t}%",
            xy=(t, y), xytext=(t + 6, y),
            fontsize=8, color=c, va="center",
            arrowprops=dict(arrowstyle="-", color=c, lw=0.8, shrinkA=0, shrinkB=2),
        )
    ax[2].set_xlabel("Volume spread across sequences (%)")
    ax[2].set_ylabel("Cumulative % of patients")
    ax[2].set_xlim(0, 80)
    ax[2].set_title("(c) How often the disagreement is large", fontsize=10)

    fig.suptitle(
        "Tumour morphology depends on which MRI sequence it was delineated on "
        f"(NPC, n={len(V)}, dataset's own measurements)", fontsize=11.5,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
