#!/usr/bin/env python3
"""Generate and verify leakage-free train / calibration / test splits.

Three partitions are produced because conformal calibration must be disjoint
from both training and testing. Every guard in src/splits.py is executed and
its result printed, so the absence of leakage is demonstrated rather than
assumed. Any violation aborts with a non-zero exit code.

Usage:
    python3 scripts/02_make_splits.py
    python3 scripts/02_make_splits.py --eligible eligible_modelling
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.discovery import discover_patients  # noqa: E402
from src.paths import load_config  # noqa: E402
from src.splits import (  # noqa: E402
    LeakageError,
    SplitSpec,
    expand_to_slices,
    make_splits,
    save_splits,
    split_fingerprint,
    verify_slice_splits,
    verify_splits,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--allow-missing-roots", action="store_true",
                    help="downgrade absent dataset roots to a warning (partially-staged dataset)")
    ap.add_argument(
        "--eligible",
        default="eligible_imaging",
        choices=["eligible_imaging", "eligible_modelling"],
        help="eligible_imaging = has slices+masks on disk; "
             "eligible_modelling = additionally has readable pixels",
    )
    args = ap.parse_args()

    cfg = load_config(args.config, verify=True, allow_missing=args.allow_missing_roots)
    np.random.seed(cfg.seed)

    manifest_path = cfg.out("manifests_dir") / "patient_manifest.csv"
    if not manifest_path.exists():
        print(f"ERROR: manifest not found at {manifest_path}\n"
              "Run scripts/01_audit_dataset.py first.", file=sys.stderr)
        return 2
    manifest = pd.read_csv(manifest_path, dtype={"patient_id": str})

    sp = cfg.raw.get("splits", {})
    spec = SplitSpec(
        train_frac=float(sp.get("train_frac", 0.55)),
        calib_frac=float(sp.get("calib_frac", 0.15)),
        test_frac=float(sp.get("test_frac", 0.30)),
        seed=cfg.seed,
        reliability_partition=sp.get("reliability_partition", "test"),
        stratify_on=tuple(sp.get("stratify_on", [])),
    )

    print("=" * 78)
    print("SPLIT GENERATION")
    print("=" * 78)
    n_elig = int(manifest[args.eligible].fillna(False).sum())
    print(f"Eligibility column : {args.eligible}  ->  {n_elig}/{len(manifest)} patients")
    print(f"Target fractions   : train {spec.train_frac} / calib {spec.calib_frac} / test {spec.test_frac}")
    print(f"Seed               : {spec.seed}")
    print()

    if n_elig == 0:
        print(f"ERROR: no patients satisfy '{args.eligible}'.", file=sys.stderr)
        if args.eligible == "eligible_modelling":
            print("The dataset currently has no readable pixel data. Use --eligible "
                  "eligible_imaging to exercise the split logic on the structure alone.",
                  file=sys.stderr)
        return 3

    try:
        splits = make_splits(manifest, spec, eligible_column=args.eligible)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    counts = splits["split"].value_counts()
    print("Patient-level assignment:")
    for name in ("train", "calib", "test", "excluded"):
        n = int(counts.get(name, 0))
        pct = n / max(1, n_elig) * 100 if name != "excluded" else n / len(splits) * 100
        print(f"  {name:9s} {n:4d}  ({pct:5.1f}%)")
    print(f"  stratified on: {splits['stratified_on'].iloc[0]}")
    dropped = splits["stratify_columns_dropped"].iloc[0]
    if dropped:
        print(f"  stratification columns unavailable and skipped: {dropped}")
    print()

    # -------------------------------------------------------------- guards
    print("Running leakage guards ...")
    try:
        passed = verify_splits(splits, manifest)
    except LeakageError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 4
    for p in passed:
        print(f"  PASS  {p}")

    # ---------------------------------------------------- slice-level check
    patients = discover_patients(cfg, cfg.images_root)
    slice_df = expand_to_slices(splits, patients)
    print(f"\nSlice-level expansion: {len(slice_df)} image slices")
    if len(slice_df):
        try:
            passed_s = verify_slice_splits(slice_df)
        except LeakageError as exc:
            print(f"\n{exc}", file=sys.stderr)
            return 4
        for p in passed_s:
            print(f"  PASS  {p}")
        print()
        print(slice_df.groupby(["split", "modality"]).size().unstack(fill_value=0).to_string())

    # ------------------------------------------------------------- outputs
    out_dir = cfg.out("splits_dir")
    path = save_splits(splits, out_dir)
    slice_path = out_dir / "slice_splits.csv"
    slice_df.to_csv(slice_path, index=False)

    print(f"\nFingerprint: {split_fingerprint(splits)}")
    print(f"Wrote:\n  {path}\n  {slice_path}")
    print("\nAll leakage guards passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
