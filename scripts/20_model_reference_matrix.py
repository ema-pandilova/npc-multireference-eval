#!/usr/bin/env python3
"""The full model-by-reference matrix: three models, three references.

The single-model result answers only part of the question. A model trained on
T1WI is aligned with the T1WI reference by construction and is the only one
evaluated in native space, since the other two references must be registered onto
it. Both properties inflate its measured reference sensitivity, so that figure is
an upper estimate rather than a neutral one.

Evaluating all three trained models against all three references separates the
two. Each model is anchored in its own native space, so registration burden and
training alignment move together around the diagonal. If the effect were an
artefact of choosing T1WI, it would shrink when T2WI or CE-T1WI supplies the
anchor.

Reports:
    the 3 x 3 median Dice matrix, models in rows and references in columns
    reference sensitivity per row, and its median across the three anchors
    whether every model scores highest against its own training reference
    diagonal against off-diagonal performance

Outputs:
    results/tables/model_reference_matrix.csv
    results/tables/model_reference_matrix.json

Usage:
    python3 scripts/20_model_reference_matrix.py
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

SEQS = ["T1WI", "T2WI", "CE-T1WI"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--allow-missing-roots", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, verify=True, allow_missing=True)
    tables = cfg.out("tables_dir")
    df = pd.read_csv(tables / "multireference_evaluation.csv", dtype={"patient_id": str})
    df["patient_id"] = df.patient_id.str.zfill(3)
    df["model_seq"] = df.model.str.replace("-3d_fullres", "", regex=False)

    have = sorted(set(df.model_seq))
    missing = [s for s in SEQS if s not in have]
    if missing:
        print(f"models not yet evaluated: {', '.join(missing)}")
        print("run scripts/16_multireference_evaluation.py for those first")
        return 1

    cols = [f"dice_vs_{s}" for s in SEQS]

    # One analysis population: patients scored against all three references by
    # every model. Anything less makes the rows incomparable.
    complete = None
    for s in SEQS:
        ok = set(df[(df.model_seq == s)].dropna(subset=cols).patient_id)
        complete = ok if complete is None else (complete & ok)
    d = df[df.patient_id.isin(complete)]
    print("=" * 78)
    print("MODEL BY REFERENCE MATRIX")
    print("=" * 78)
    print(f"\n{len(complete)} patients scored by all three models against all three references\n")

    print(f"{'trained on':<12}" + "".join(f"{'vs ' + s:>13}" for s in SEQS)
          + f"{'ref sens':>11}{'own ref best':>14}")
    matrix, out = {}, {}
    for s in SEQS:
        g = d[d.model_seq == s]
        meds = [g[c].median() for c in cols]
        sens = (g[cols].max(axis=1) - g[cols].min(axis=1)).median()
        best_is_own = SEQS[int(np.argmax(meds))] == s
        print(f"{s:<12}" + "".join(f"{m:>13.4f}" for m in meds)
              + f"{sens:>11.4f}{('yes' if best_is_own else 'NO'):>14}")
        matrix[s] = {r: round(float(m), 4) for r, m in zip(SEQS, meds)}
        out[s] = {"per_reference": matrix[s],
                  "reference_sensitivity_median": round(float(sens), 4),
                  "best_is_own_reference": bool(best_is_own),
                  "n": int(len(g))}

    # Off-diagonal-only sensitivity. Each model's own reference is the single
    # cell compared in native space; the other two are registered onto it. The
    # diagonal therefore carries a registration advantage that has nothing to do
    # with training alignment, and any spread involving it mixes the two. Taking
    # the difference between the two registered cells removes that advantage,
    # because both carry a comparable registration penalty. If this quantity
    # stays large in every anchor space, the effect is not a registration
    # artefact.
    print()
    print(f"{'anchor space':<14}{'off-diagonal pair':>24}{'|difference|':>14}")
    for s in SEQS:
        g = d[d.model_seq == s]
        others = [r for r in SEQS if r != s]
        oc = [f"dice_vs_{r}" for r in others]
        off = (g[oc[0]] - g[oc[1]]).abs().median()
        print(f"{s:<14}{others[0] + ' vs ' + others[1]:>24}{off:>14.4f}")
        out[s]["offdiagonal_difference_median"] = round(float(off), 4)
        out[s]["offdiagonal_pair"] = others
    offs = [out[s]["offdiagonal_difference_median"] for s in SEQS]
    print(f"\n  median across anchors: {np.median(offs):.4f} "
          f"(range {min(offs):.4f} to {max(offs):.4f})")
    print("  Both cells in each pair are registered, so this difference is free of")
    print("  the native-space advantage the diagonal enjoys.")

    sens_all = [out[s]["reference_sensitivity_median"] for s in SEQS]
    diag = [matrix[s][s] for s in SEQS]
    offd = [matrix[s][r] for s in SEQS for r in SEQS if r != s]
    print(f"\n  median reference sensitivity across the three anchors: {np.median(sens_all):.4f}"
          f"  (range {min(sens_all):.4f} to {max(sens_all):.4f})")
    print(f"  diagonal median {np.median(diag):.4f}, off-diagonal median {np.median(offd):.4f}, "
          f"difference {np.median(diag) - np.median(offd):.4f}")
    n_own = sum(out[s]["best_is_own_reference"] for s in SEQS)
    print(f"  each model scores highest against its own training reference: {n_own} of 3")
    print("\n  Each row is one fixed set of predictions scored three ways, so every")
    print("  within-row difference is attributable to the reference alone.")

    rows = [{"model": s, **{f"vs_{r}": matrix[s][r] for r in SEQS},
             "reference_sensitivity": out[s]["reference_sensitivity_median"],
             "best_is_own_reference": out[s]["best_is_own_reference"]} for s in SEQS]
    pd.DataFrame(rows).to_csv(tables / "model_reference_matrix.csv", index=False)
    with open(tables / "model_reference_matrix.json", "w", encoding="utf-8") as fh:
        json.dump({"n_patients": int(len(complete)), "models": out,
                   "median_reference_sensitivity": round(float(np.median(sens_all)), 4),
                   "sensitivity_range": [round(min(sens_all), 4), round(max(sens_all), 4)],
                   "diagonal_median": round(float(np.median(diag)), 4),
                   "offdiagonal_median": round(float(np.median(offd)), 4),
                   "n_models_best_on_own_reference": int(n_own),
                   "offdiagonal_difference_median": round(float(np.median(offs)), 4),
                   "offdiagonal_range": [round(min(offs), 4), round(max(offs), 4)]}, fh, indent=2)
    print(f"\nWrote {tables}/model_reference_matrix.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
