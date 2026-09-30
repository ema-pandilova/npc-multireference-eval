#!/usr/bin/env python3
"""Predict, from a single sequence, whether a tumour measurement is sequence-unstable.

The rest of the study shows that tumour measurements disagree across sequences,
by a median of 20% in volume. That is only actionable if the unstable cases can
be identified in advance, because in practice a clinician or a pipeline often has
one sequence and has to decide whether to trust the number it gives.

This is a predictive-modelling task with a concrete decision attached: flag the
patients whose measurement would change materially had a different sequence been
used, using only what is available from the sequence in hand.

    Outcome    cross-sequence volume spread above a threshold (default 20%,
               which splits the cohort roughly in half and is the point beyond
               which a volume-based risk band starts to flip).
    Predictors morphology from ONE sequence only (volume, surface area, maximum
               diameter, surface regularity), plus acquisition metadata (field
               strength, vendor) and stage. Nothing that requires having already
               measured the other sequences, which would leak the outcome.
    Validation stratified 5-fold cross-validation at patient level, repeated,
               with the model refit inside every fold.

Reported as ROC AUC, average precision, and a decision-curve view: at a chosen
operating point, how many unstable cases are caught and how many stable ones are
needlessly flagged.

Usage:
    python3 scripts/14_predict_instability.py
    python3 scripts/14_predict_instability.py --threshold 30
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
    for c in ("T", "N", "M"):
        df[f"{c.lower()}_stage"] = pd.to_numeric(df[c], errors="coerce")
    df["age_num"] = pd.to_numeric(df.get("age"), errors="coerce")
    df["sex_m"] = (df.get("gender", pd.Series(dtype=str)).astype(str).str.strip().str.lower() == "male").astype(int)
    return df[["patient_id", "t_stage", "n_stage", "m_stage", "age_num", "sex_m"]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--allow-missing-roots", action="store_true")
    ap.add_argument("--threshold", type=float, default=20.0,
                    help="cross-sequence volume spread (%%) defining an unstable case")
    ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args()

    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    cfg = load_config(args.config, verify=True, allow_missing=True)
    root = cfg.primary_data_root
    tables = cfg.out("tables_dir")

    m = read_morphology(root / "morphological_parameters.csv")
    diag = read_diagnoses(root / "info_diagnoses.csv")
    scan_p = tables / "scanner_per_patient.csv"
    scan = pd.read_csv(scan_p, dtype={"patient_id": str}) if scan_p.exists() else pd.DataFrame()

    d = m.merge(diag, on="patient_id", how="left")
    if not scan.empty:
        d = d.merge(scan[["patient_id", "manufacturer", "field_t"]], on="patient_id", how="left")

    V = d[[f"{s}_Vol" for s in SEQS]]
    d["spread_pct"] = (V.max(axis=1) - V.min(axis=1)) / V.mean(axis=1) * 100
    d["unstable"] = (d.spread_pct > args.threshold).astype(int)

    print("=" * 78)
    print("PREDICTING SEQUENCE-INSTABILITY OF A TUMOUR MEASUREMENT")
    print("=" * 78)
    print(f"outcome: cross-sequence volume spread > {args.threshold:.0f}%")
    print(f"patients: {len(d)}   unstable: {int(d.unstable.sum())} "
          f"({100 * d.unstable.mean():.1f}%)\n")

    results = {}
    for seq in SEQS:
        # Only features obtainable from THIS sequence, plus acquisition and stage.
        num = [f"{seq}_{p}" for p in PARAMS] + ["t_stage", "n_stage", "age_num", "sex_m"]
        if "field_t" in d.columns:
            num.append("field_t")
        cat = ["manufacturer"] if "manufacturer" in d.columns else []
        X = d[num + cat].copy()
        y = d["unstable"].values

        pre = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), num),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
        ]) if cat else ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), num)])

        models = {
            "logistic": Pipeline([("pre", pre),
                                  ("clf", LogisticRegression(max_iter=2000, C=1.0))]),
            "gradient boosting": Pipeline([("pre", pre),
                                           ("clf", GradientBoostingClassifier(random_state=cfg.seed))]),
        }
        print(f"--- reference sequence available: {LABEL[seq]} ---")
        for name, pipe in models.items():
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=args.repeats,
                                         random_state=cfg.seed)
            # cross_val_predict with a repeated splitter averages over repeats by
            # refitting inside each fold, so no information crosses the fold wall.
            aucs, aps = [], []
            from sklearn.model_selection import StratifiedKFold
            for r in range(args.repeats):
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.seed + r)
                p = cross_val_predict(pipe, X, y, cv=skf, method="predict_proba")[:, 1]
                aucs.append(roc_auc_score(y, p))
                aps.append(average_precision_score(y, p))
            results.setdefault(LABEL[seq], {})[name] = {
                "auc_mean": round(float(np.mean(aucs)), 4),
                "auc_sd": round(float(np.std(aucs)), 4),
                "ap_mean": round(float(np.mean(aps)), 4),
                "prevalence": round(float(y.mean()), 4),
            }
            print(f"    {name:<20} AUC {np.mean(aucs):.3f} (sd {np.std(aucs):.3f})   "
                  f"AP {np.mean(aps):.3f}  (baseline AP {y.mean():.3f})")

        # Operating point from the best fold-averaged model, for the decision view.
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.seed)
        best = max(models, key=lambda k: results[LABEL[seq]][k]["auc_mean"])
        p = cross_val_predict(models[best], X, y, cv=skf, method="predict_proba")[:, 1]
        fpr, tpr, thr = roc_curve(y, p)
        # Choose the threshold catching 80% of unstable cases; report the cost.
        i = int(np.argmin(np.abs(tpr - 0.80)))
        flagged = float((p >= thr[i]).mean())
        print(f"    operating point ({best}): catch {tpr[i]:.0%} of unstable cases "
              f"by flagging {flagged:.0%} of patients "
              f"(false-positive rate {fpr[i]:.0%})")
        results[LABEL[seq]]["operating_point"] = {
            "model": best, "sensitivity": round(float(tpr[i]), 3),
            "fpr": round(float(fpr[i]), 3), "fraction_flagged": round(flagged, 3),
        }
        print()

    # Which features drive it?
    print("--- what predicts instability (logistic coefficients, T1WI model) ---")
    num = [f"T1_{p}" for p in PARAMS] + ["t_stage", "n_stage", "age_num", "sex_m"]
    if "field_t" in d.columns:
        num.append("field_t")
    Xn = d[num]
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=2000))])
    pipe.fit(Xn, d.unstable)
    coefs = pd.Series(pipe.named_steps["clf"].coef_[0], index=num).sort_values(key=abs, ascending=False)
    for k, v in coefs.items():
        print(f"    {k:<16} {v:+.3f}")

    with open(tables / "instability_prediction.json", "w", encoding="utf-8") as fh:
        json.dump({"threshold_pct": args.threshold, "results": results,
                   "logistic_coefficients_T1WI": {k: round(float(v), 4) for k, v in coefs.items()}},
                  fh, indent=2)
    d[["patient_id", "spread_pct", "unstable"]].to_csv(tables / "instability_labels.csv", index=False)
    print(f"\nWrote {tables}/instability_prediction.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
