#!/usr/bin/env python3
"""Paired comparison of cross-sequence against inter-reader disagreement, in Dice.

The headline contrast in this paper had been the difference between two medians
computed on different groups: cross-sequence agreement over all 277 patients and
inter-reader agreement over the 30-patient two-reader subset. Subtracting them is
descriptively informative but is not a paired estimate, because no patient
contributes to both terms in a matched way.

This script restricts both quantities to the same 30 patients and pairs them
within patient, which is the comparison the paper should be making. For each
patient it takes the median cross-sequence Dice over the three sequence pairs and
the median inter-reader Dice over the three sequences, then tests the paired
difference.

Outputs:
    results/tables/paired_sequence_vs_reader_dice.csv
    results/tables/paired_sequence_vs_reader_dice.json
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--allow-missing-roots", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, verify=True, allow_missing=True)
    tables = cfg.out("tables_dir")

    sw = pd.read_csv(tables / "reference_swap_pairs.csv", dtype={"patient_id": str})
    ir = pd.read_csv(tables / "inter_rater_agreement.csv", dtype={"patient_id": str})
    sw["patient_id"] = sw.patient_id.str.zfill(3)
    ir["patient_id"] = ir.patient_id.str.zfill(3)
    sw = sw[sw.status == "ok"]

    seq = sw.groupby("patient_id").dice.agg(["median", "count"]).rename(
        columns={"median": "cross_sequence_dice", "count": "n_pairs"})
    rdr = ir.groupby("patient_id").dice.agg(["median", "count"]).rename(
        columns={"median": "inter_reader_dice", "count": "n_sequences"})
    d = seq.join(rdr, how="inner").dropna()
    d["difference"] = d.inter_reader_dice - d.cross_sequence_dice

    print("=" * 78)
    print("PAIRED CROSS-SEQUENCE VERSUS INTER-READER AGREEMENT (Dice)")
    print("=" * 78)
    print(f"\n{len(d)} patients contribute to both quantities\n")
    print(f"  median cross-sequence Dice   {d.cross_sequence_dice.median():.4f}")
    print(f"  median inter-reader Dice     {d.inter_reader_dice.median():.4f}")
    print(f"  median paired difference     {d.difference.median():.4f}")
    print(f"  inter-reader higher in       {int((d.difference > 0).sum())} of {len(d)} patients")

    rng = np.random.default_rng(20260803)
    x = d.difference.to_numpy()
    boot = [np.median(rng.choice(x, len(x), replace=True)) for _ in range(10000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  bootstrap 95% CI             {lo:.4f} to {hi:.4f}")

    out = {"n_patients": int(len(d)),
           "cross_sequence_median": round(float(d.cross_sequence_dice.median()), 4),
           "inter_reader_median": round(float(d.inter_reader_dice.median()), 4),
           "paired_difference_median": round(float(d.difference.median()), 4),
           "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4),
           "n_inter_reader_higher": int((d.difference > 0).sum())}

    try:
        from scipy.stats import wilcoxon
        st, pv = wilcoxon(d.inter_reader_dice, d.cross_sequence_dice)
        print(f"  Wilcoxon signed-rank         W = {st:.1f}, p = {pv:.3g}")
        out["wilcoxon_W"] = float(st); out["wilcoxon_p"] = float(pv)
    except ImportError:
        print("  scipy unavailable; test skipped")

    # The unpaired contrast the paper previously quoted, for comparison.
    print(f"\n  For reference, the unpaired contrast used previously was "
          f"{ir.dice.median():.4f} minus {sw.dice.median():.4f} = "
          f"{ir.dice.median() - sw.dice.median():.4f},")
    print("  computed on different groups of patients and therefore descriptive only.")
    out["unpaired_contrast"] = round(float(ir.dice.median() - sw.dice.median()), 4)

    d.round(4).to_csv(tables / "paired_sequence_vs_reader_dice.csv")
    with open(tables / "paired_sequence_vs_reader_dice.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {tables}/paired_sequence_vs_reader_dice.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
