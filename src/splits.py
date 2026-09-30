"""Leakage-free patient-level split generation and verification.

The study needs THREE disjoint partitions, not two:

    train  — fits the segmentation model
    calib  — fits the conformal / uncertainty calibration ONLY
    test   — reports coverage and all downstream results

Conformal guarantees are void if calibration data touches either training or
testing, so `verify_splits` treats that as a hard error rather than a warning.

Splitting is done at PATIENT level and then expanded to slices. Doing it the
other way round would put neighbouring slices of the same tumour in both train
and test — the classic and near-invisible leak in 2D slice-based medical
imaging studies, which inflates Dice substantially.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


class LeakageError(RuntimeError):
    """Raised when a split violates a partition-disjointness guarantee."""


@dataclass
class SplitSpec:
    train_frac: float
    calib_frac: float
    test_frac: float
    seed: int
    reliability_partition: str = "test"
    stratify_on: tuple[str, ...] = ()

    def validate(self) -> None:
        total = self.train_frac + self.calib_frac + self.test_frac
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"split fractions must sum to 1.0, got {total}")
        if self.reliability_partition not in ("train", "calib", "test"):
            raise ValueError(f"invalid reliability_partition: {self.reliability_partition}")


def _stratum_key(row: pd.Series, columns: list[str]) -> str:
    parts = []
    for c in columns:
        v = row.get(c)
        parts.append("NA" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
    return "|".join(parts) if parts else "ALL"


def make_splits(
    manifest: pd.DataFrame,
    spec: SplitSpec,
    eligible_column: str = "eligible_imaging",
) -> pd.DataFrame:
    """Assign every eligible patient to exactly one of train / calib / test.

    Reliability-subset patients are placed first, into a single fixed partition,
    so that genuine inter-rater disagreement is measured only on held-out cases
    and never influences model fitting or calibration.
    """
    spec.validate()
    rng = np.random.default_rng(spec.seed)

    df = manifest.copy()
    if eligible_column in df.columns:
        pool = df[df[eligible_column].fillna(False)].copy()
    else:
        pool = df.copy()

    if pool.empty:
        raise ValueError(
            f"No patients satisfy '{eligible_column}'. Refusing to emit an empty split. "
            "Check the manifest and the dataset contents."
        )

    # Only stratify on columns that exist and carry real signal.
    strat_cols = [
        c for c in spec.stratify_on
        if c in pool.columns and pool[c].notna().any() and pool[c].nunique(dropna=True) > 1
    ]
    dropped = [c for c in spec.stratify_on if c not in strat_cols]

    assignment: dict[str, str] = {}

    rel_mask = pool.get("in_reliability_subset", pd.Series(False, index=pool.index)).fillna(False)
    for pid in pool.loc[rel_mask, "patient_id"]:
        assignment[str(pid)] = spec.reliability_partition

    remaining = pool.loc[~rel_mask].copy()
    if strat_cols:
        remaining["_stratum"] = remaining.apply(lambda r: _stratum_key(r, strat_cols), axis=1)
    else:
        remaining["_stratum"] = "ALL"

    # Within each stratum, shuffle deterministically and cut by fraction. The
    # reliability patients already consumed part of their target partition, so
    # fractions apply to the remaining pool; exact global proportions are
    # reported afterwards rather than forced.
    for stratum, grp in remaining.groupby("_stratum", sort=True):
        ids = sorted(str(p) for p in grp["patient_id"])
        order = rng.permutation(len(ids))
        ids = [ids[i] for i in order]
        n = len(ids)
        n_train = int(round(spec.train_frac * n))
        n_calib = int(round(spec.calib_frac * n))
        # Guarantee calibration is non-empty whenever the stratum can afford it.
        if n >= 3 and n_calib == 0:
            n_calib = 1
        n_train = min(n_train, n - n_calib)
        for i, pid in enumerate(ids):
            if i < n_train:
                assignment[pid] = "train"
            elif i < n_train + n_calib:
                assignment[pid] = "calib"
            else:
                assignment[pid] = "test"

    out = df[["patient_id"]].copy()
    out["patient_id"] = out["patient_id"].astype(str)
    out["split"] = out["patient_id"].map(assignment).fillna("excluded")
    out["in_reliability_subset"] = (
        df.get("in_reliability_subset", pd.Series(False, index=df.index)).fillna(False).values
    )
    out["stratified_on"] = "|".join(strat_cols) if strat_cols else "NONE"
    out["stratify_columns_dropped"] = "|".join(dropped) if dropped else ""
    out["seed"] = spec.seed
    return out


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def verify_splits(splits: pd.DataFrame, manifest: pd.DataFrame | None = None) -> list[str]:
    """Assert every partition-disjointness guarantee. Raises on violation.

    Returns the list of checks that passed, so callers can print positive
    evidence that the guards actually ran rather than assuming they did.
    """
    passed: list[str] = []
    errors: list[str] = []

    # 1. A patient appears in exactly one partition.
    counts = splits.groupby("patient_id")["split"].nunique()
    multi = counts[counts > 1]
    if len(multi):
        errors.append(f"patients in multiple partitions: {sorted(multi.index)[:10]}")
    else:
        passed.append("no patient appears in more than one partition")

    dupes = splits["patient_id"][splits["patient_id"].duplicated()]
    if len(dupes):
        errors.append(f"duplicate patient rows in splits: {sorted(set(dupes))[:10]}")
    else:
        passed.append("no duplicate patient rows")

    # 2. Calibration is disjoint from train and test (conformal validity).
    part = {k: set(v) for k, v in splits.groupby("split")["patient_id"].apply(set).items()}
    train, calib, test = part.get("train", set()), part.get("calib", set()), part.get("test", set())
    for a_name, a, b_name, b in (
        ("calib", calib, "train", train),
        ("calib", calib, "test", test),
        ("train", train, "test", test),
    ):
        overlap = a & b
        if overlap:
            errors.append(f"{a_name} overlaps {b_name}: {sorted(overlap)[:10]}")
        else:
            passed.append(f"{a_name} is disjoint from {b_name}")

    # 3. Reliability patients are confined to a single partition, so repeated
    #    annotations of one case cannot straddle partitions. 'excluded' is not a
    #    partition — it means the patient is outside the study entirely — so it
    #    is reported but does not count as straddling.
    rel = splits[splits["in_reliability_subset"].fillna(False)]
    if len(rel):
        assigned = rel[rel["split"] != "excluded"]
        n_excluded = len(rel) - len(assigned)
        rel_parts = sorted(assigned["split"].unique())
        if len(rel_parts) > 1:
            errors.append(f"reliability-subset patients span multiple partitions: {rel_parts}")
        elif len(rel_parts) == 1:
            msg = f"all {len(assigned)} assigned reliability patients confined to '{rel_parts[0]}'"
            if n_excluded:
                msg += f" ({n_excluded} ineligible and excluded from the study)"
            passed.append(msg)
        else:
            passed.append(f"no eligible reliability patients ({n_excluded} all excluded)")
    else:
        passed.append("no reliability patients in the eligible pool (nothing to confine)")

    # 4. Partitions are non-empty; an empty calib set silently disables conformal.
    for name, s in (("train", train), ("calib", calib), ("test", test)):
        if not s:
            errors.append(f"partition '{name}' is empty")
    if train and calib and test:
        passed.append("train / calib / test are all non-empty")

    # 5. Every eligible manifest patient was assigned.
    if manifest is not None and "eligible_imaging" in manifest.columns:
        eligible = set(manifest.loc[manifest["eligible_imaging"].fillna(False), "patient_id"].astype(str))
        assigned = set(splits.loc[splits["split"] != "excluded", "patient_id"].astype(str))
        missed = eligible - assigned
        if missed:
            errors.append(f"eligible patients left unassigned: {sorted(missed)[:10]}")
        else:
            passed.append(f"all {len(eligible)} eligible patients assigned")

    if errors:
        raise LeakageError("SPLIT VERIFICATION FAILED:\n  - " + "\n  - ".join(errors))
    return passed


def expand_to_slices(splits: pd.DataFrame, patients: dict) -> pd.DataFrame:
    """Expand a patient-level split to a slice-level table.

    Every slice of every modality inherits its patient's partition, which is
    what keeps all modalities of one patient together and prevents adjacent
    slices from straddling partitions.
    """
    assign = dict(zip(splits["patient_id"].astype(str), splits["split"]))
    rows = []
    for pid, rec in sorted(patients.items()):
        split = assign.get(str(pid), "excluded")
        for s in rec.slices:
            if s.is_mask:
                continue
            rows.append(
                {
                    "patient_id": str(pid),
                    "split": split,
                    "modality": s.modality,
                    "slice_index": s.index,
                    "image_path": str(s.path),
                    "mask_path": str(s.path.with_name(f"{s.path.stem}_mask{s.path.suffix}")),
                    "image_bytes": s.size_bytes,
                }
            )
    return pd.DataFrame(rows)


def verify_slice_splits(slice_df: pd.DataFrame) -> list[str]:
    """Verify the slice-level expansion preserved patient-level disjointness."""
    passed, errors = [], []
    if slice_df.empty:
        return ["slice table is empty (no readable slices to verify)"]

    per_patient = slice_df.groupby("patient_id")["split"].nunique()
    bad = per_patient[per_patient > 1]
    if len(bad):
        errors.append(f"patients whose slices straddle partitions: {sorted(bad.index)[:10]}")
    else:
        passed.append("every patient's slices sit in exactly one partition")

    # All modalities of a patient must share a partition.
    mod_check = slice_df.groupby(["patient_id", "modality"])["split"].nunique()
    bad_mod = mod_check[mod_check > 1]
    if len(bad_mod):
        errors.append(f"modality/partition split for: {sorted({i[0] for i in bad_mod.index})[:10]}")
    else:
        passed.append("all modalities of each patient share one partition")

    if errors:
        raise LeakageError("SLICE-LEVEL VERIFICATION FAILED:\n  - " + "\n  - ".join(errors))
    return passed


def split_fingerprint(splits: pd.DataFrame) -> str:
    """Stable hash of the assignment, so a split can be proven identical later."""
    payload = json.dumps(
        sorted(zip(splits["patient_id"].astype(str), splits["split"])), sort_keys=True
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def save_splits(splits: pd.DataFrame, out_dir: Path, name: str = "patient_splits") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.csv"
    splits.to_csv(path, index=False)
    meta = {
        "fingerprint": split_fingerprint(splits),
        "counts": splits["split"].value_counts().to_dict(),
        "n_reliability": int(splits["in_reliability_subset"].fillna(False).sum()),
        "stratified_on": splits["stratified_on"].iloc[0] if len(splits) else "",
        "seed": int(splits["seed"].iloc[0]) if len(splits) else None,
    }
    with open(out_dir / f"{name}_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return path
