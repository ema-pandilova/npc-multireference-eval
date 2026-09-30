#!/usr/bin/env python3
"""Recover per-patient scanner metadata from the DICOM headers.

The data descriptor names `meta_data.csv` as the source of MRI machine
information, but that file is **absent from the released archive**. The DICOM
headers still carry it, and in more detail than the descriptor's Table 3, which
lists model names only:

    Manufacturer, ManufacturerModelName, MagneticFieldStrength,
    PixelSpacing, SliceThickness, Rows/Columns, TR/TE, StationName

Field strength in particular is not reported anywhere in the publication, and it
matters: a third of this cohort turns out to sit near 1.0 T rather than the
1.5 T / 3.0 T that secondary summaries assume.

One header is read per patient per sequence (pixels skipped), so this is fast.

Outputs:
    results/tables/scanner_metadata.csv          per patient per sequence
    results/tables/scanner_distribution.csv      cohort-level distribution

Usage:
    python3 scripts/07_scanner_metadata.py
    python3 scripts/07_scanner_metadata.py --segments-root ../dataset_v2/primary_data/data/MRI-Segments
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import load_config  # noqa: E402

TAGS = [
    "Manufacturer", "ManufacturerModelName", "MagneticFieldStrength",
    "SeriesDescription", "PixelSpacing", "SliceThickness", "SpacingBetweenSlices",
    "Rows", "Columns", "RepetitionTime", "EchoTime", "FlipAngle",
    "StationName", "SoftwareVersions", "StudyDate",
]

KNOWN_SEQUENCES = {"T1WI", "T2WI", "CE-T1WI"}


def classify_sequence(series_dir: str, desc: str = "") -> str:
    """Identify the sequence from the directory the series sits in.

    The dataset organises each patient as <id>/{T1WI,T2WI,CE-T1WI}/*.dcm, so the
    folder name is authoritative. An earlier version parsed the free-text
    SeriesDescription instead and split T1WI from CE-T1WI unreliably (it assigned
    454/277/100 across the three sequences where the truth is 277 each), because
    console descriptions do not consistently mark contrast. SeriesDescription is
    kept only as a fallback for data laid out some other way.
    """
    folder = os.path.basename(str(series_dir).rstrip("/"))
    if folder in KNOWN_SEQUENCES:
        return folder

    d = (desc or "").lower()
    if "t2" in d:
        return "T2WI"
    if "t1" in d and any(k in d for k in ("+c", "post", "gd", "ce", "c+")):
        return "CE-T1WI"
    if "t1" in d:
        return "T1WI"
    return "UNKNOWN"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--allow-missing-roots", action="store_true")
    ap.add_argument("--segments-root", default=None,
                    help="MRI-Segments directory holding per-patient DICOM series")
    args = ap.parse_args()

    cfg = load_config(args.config, verify=True, allow_missing=True)
    root = (
        Path(args.segments_root) if args.segments_root and Path(args.segments_root).is_absolute()
        else (cfg.repo_root / args.segments_root).resolve() if args.segments_root
        else cfg.primary_data_root / "MRI-Segments"
    )
    if not root.is_dir():
        print(f"ERROR: {root} not found. Stage primary_data.zip first.", file=sys.stderr)
        return 2

    try:
        import pydicom
    except ImportError:
        print("ERROR: pydicom is required (pip install --user pydicom).", file=sys.stderr)
        return 3

    print("=" * 78)
    print("SCANNER METADATA FROM DICOM HEADERS")
    print("=" * 78)
    print(f"Root: {root}\n")

    rows: list[dict] = []
    pids = sorted(p for p in os.listdir(root) if (root / p).is_dir())
    for pid in pids:
        files = sorted(glob.glob(str(root / pid / "**" / "*.dcm"), recursive=True))
        if not files:
            rows.append({"patient_id": pid, "status": "NO_DICOM"})
            continue
        # Group by directory so one header is read per series rather than per slice.
        by_dir: dict[str, str] = {}
        for f in files:
            by_dir.setdefault(os.path.dirname(f), f)
        for d, f in sorted(by_dir.items()):
            try:
                ds = pydicom.dcmread(f, stop_before_pixels=True)
            except Exception as exc:  # noqa: BLE001
                rows.append({"patient_id": pid, "status": f"UNREADABLE: {type(exc).__name__}"})
                continue
            rec: dict = {
                "patient_id": pid,
                "status": "OK",
                "series_dir": os.path.relpath(d, root),
                "n_slices": sum(1 for x in files if os.path.dirname(x) == d),
            }
            for t in TAGS:
                v = getattr(ds, t, None)
                if t == "PixelSpacing" and v is not None:
                    try:
                        rec["pixel_spacing_x_mm"] = round(float(v[0]), 4)
                        rec["pixel_spacing_y_mm"] = round(float(v[1]), 4)
                    except Exception:  # noqa: BLE001
                        pass
                    continue
                rec[t] = (
                    round(float(v), 4)
                    if t in ("MagneticFieldStrength", "SliceThickness", "SpacingBetweenSlices",
                             "RepetitionTime", "EchoTime", "FlipAngle") and v is not None
                    else (str(v) if v is not None else None)
                )
            rec["sequence"] = classify_sequence(rec["series_dir"], rec.get("SeriesDescription"))
            rows.append(rec)

    df = pd.DataFrame(rows)
    ok = df[df.status == "OK"].copy()
    print(f"patients: {len(pids)}   series read: {len(ok)}   "
          f"patients with no DICOM: {int((df.status == 'NO_DICOM').sum())}")

    if ok.empty:
        print("No readable headers.", file=sys.stderr)
        return 4

    # -- per-patient scanner identity -------------------------------------
    # A patient is assigned the model of their first series; disagreement within a
    # patient is reported rather than silently collapsed.
    per_pat = ok.groupby("patient_id").agg(
        manufacturer=("Manufacturer", lambda s: s.dropna().iloc[0] if s.notna().any() else None),
        model=("ManufacturerModelName", lambda s: s.dropna().iloc[0] if s.notna().any() else None),
        field_t=("MagneticFieldStrength", lambda s: s.dropna().iloc[0] if s.notna().any() else None),
        n_models=("ManufacturerModelName", lambda s: s.dropna().nunique()),
        n_series=("series_dir", "size"),
    ).reset_index()

    mixed = per_pat[per_pat.n_models > 1]
    print(f"patients imaged on more than one model: {len(mixed)}"
          + (f"  {sorted(mixed.patient_id)[:8]}" if len(mixed) else ""))
    print()

    print("=== Scanner distribution (per patient) ===")
    dist = per_pat.groupby(["manufacturer", "model", "field_t"], dropna=False).size()
    dist = dist.reset_index(name="n_patients").sort_values("n_patients", ascending=False)
    print(dist.to_string(index=False))
    print(f"\ntotal: {int(dist.n_patients.sum())} patients")

    print("\n=== Field strength (per patient) ===")
    fs = per_pat.field_t.round(2).value_counts(dropna=False).sort_index()
    for v, n in fs.items():
        print(f"  {v} T : {n} patients ({100 * n / len(per_pat):.1f}%)")
    print("\nNOTE: field strength appears NOWHERE in the data descriptor. Any claim")
    print("about 1.5 T / 3.0 T in secondary summaries of this dataset is unsupported.")

    print("\n=== Vendor (per patient) ===")
    for v, n in per_pat.manufacturer.value_counts(dropna=False).items():
        print(f"  {v}: {n} ({100 * n / len(per_pat):.1f}%)")

    print("\n=== Acquisition geometry (per series) ===")
    geo = ok.groupby("sequence").agg(
        n=("series_dir", "size"),
        px_median=("pixel_spacing_x_mm", "median"),
        px_min=("pixel_spacing_x_mm", "min"),
        px_max=("pixel_spacing_x_mm", "max"),
        thick_median=("SliceThickness", "median"),
        rows_median=("Rows", lambda s: pd.to_numeric(s, errors="coerce").median()),
    )
    print(geo.round(3).to_string())

    # -- feasibility of a scanner-stratified analysis -----------------------
    print("\n=== Feasibility of scanner stratification ===")
    strata = dist[dist.n_patients >= 20]
    print(f"strata with n>=20 patients: {len(strata)} "
          f"covering {int(strata.n_patients.sum())}/{len(per_pat)} patients")
    if len(strata) >= 2:
        print("  A scanner-stratified or leave-one-scanner-out analysis is viable on")
        print("  these strata; smaller strata must be pooled or reported descriptively.")
    else:
        print("  Too few adequately-sized strata; keep scanner as a descriptive covariate only.")

    tables = cfg.out("tables_dir")
    df.to_csv(tables / "scanner_metadata.csv", index=False)
    per_pat.to_csv(tables / "scanner_per_patient.csv", index=False)
    dist.to_csv(tables / "scanner_distribution.csv", index=False)
    print(f"\nWrote {tables}/scanner_metadata.csv, scanner_per_patient.csv, scanner_distribution.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
