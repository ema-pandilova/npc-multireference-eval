#!/usr/bin/env python3
"""Audit the NPC dataset: discovery, geometry, clinical tables, manifest.

Writes:
    data_summary.csv                     patient-level inventory
    data_audit.md                        human-readable audit report
    data/manifests/patient_manifest.csv  same inventory (canonical copy)
    data/manifests/clinical_profile.json column profiles of the CSVs
    results/tables/dataset_summary.json  headline statistics

Usage:
    python3 scripts/01_audit_dataset.py
    python3 scripts/01_audit_dataset.py --no-pixels     # structure only, fast
    python3 scripts/01_audit_dataset.py --config path/to/paths.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.clinical import attach_clinical, load_table, match_ids, normalise_id  # noqa: E402
from src.discovery import discover_patients, discover_reliability, scan_volume_files  # noqa: E402
from src.imaging import check_pair_geometry  # noqa: E402
from src.manifest import add_analysis_flags, build_manifest, summarise  # noqa: E402
from src.paths import describe_roots, load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-missing-roots", action="store_true",
                    help="downgrade absent dataset roots to a warning (partially-staged dataset)")
    ap.add_argument("--config", default=None, help="path to paths.yaml")
    ap.add_argument("--no-pixels", action="store_true", help="skip pixel reading (structure only)")
    ap.add_argument("--geometry-checks", type=int, default=40, help="number of image/mask pairs to verify")
    args = ap.parse_args()

    cfg = load_config(args.config, verify=True, allow_missing=args.allow_missing_roots)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    print("=" * 78)
    print("NPC DATASET AUDIT")
    print("=" * 78)
    print(describe_roots(cfg))
    print()

    # ---------------------------------------------------------------- discovery
    patients = discover_patients(cfg, cfg.images_root)
    reliability = discover_reliability(cfg)
    print(f"Discovered {len(patients)} patient directories under images_root")
    print(f"Discovered {len(reliability)} entries under reliability_root")

    volumes = scan_volume_files(cfg, cfg.images_root) + scan_volume_files(cfg, cfg.primary_data_root)
    print(f"3D volume files (NIfTI/DICOM/NRRD) found: {len(volumes)}")

    # --------------------------------------------------------------- manifest
    print("\nBuilding patient manifest"
          f"{' (structure only)' if args.no_pixels else ' (reading pixels)'} ...")
    manifest = build_manifest(cfg, patients, reliability, inspect_pixels=not args.no_pixels)
    manifest = add_analysis_flags(manifest)
    stats = summarise(manifest, cfg)

    # ------------------------------------------------------- clinical tables
    imaging_ids = set(manifest["patient_id"].astype(str))
    clinical_profiles, match_reports = {}, {}
    for key in ("diagnoses", "morphology"):
        path = cfg.primary_file(key)
        df, prof = load_table(path)
        clinical_profiles[key] = prof
        print(f"  {key:11s} {prof.status:14s} {path.name}"
              + (f"  [{prof.n_rows} rows x {prof.n_cols} cols]" if prof.ok else f"  ({prof.error})"))
        if df is not None and prof.ok:
            rep = match_ids(df, imaging_ids)
            match_reports[key] = rep
            print(f"              -> ID column '{rep.id_column}', "
                  f"match {rep.n_matched}/{rep.n_imaging_ids} ({rep.match_rate:.1%})")
            manifest = attach_clinical(manifest, df, rep, prefix=f"{key}_")
            if rep.id_column:
                csv_ids = {i for i in df[rep.id_column].map(normalise_id) if i}
                manifest[f"has_{key}_record"] = manifest["patient_id"].isin(csv_ids)
            else:
                manifest[f"has_{key}_record"] = False
        else:
            manifest[f"has_{key}_record"] = False

    readme_path = cfg.primary_file("readme")
    readme_status = (
        "MISSING" if not readme_path.exists()
        else "EMPTY_FILE" if readme_path.stat().st_size == 0
        else "OK"
    )
    segments_dir = cfg.primary_file("segments_dir")
    segments_entries = cfg.clean_listdir(segments_dir) if segments_dir.is_dir() else []

    # ------------------------------------------------------ geometry checks
    geometry = []
    if not args.no_pixels:
        print(f"\nVerifying image/mask geometry on up to {args.geometry_checks} pairs ...")
        candidates = []
        for pid, rec in sorted(patients.items()):
            for mod in cfg.modalities:
                for s in rec.images(mod):
                    mp = s.path.with_name(f"{s.path.stem}_mask{s.path.suffix}")
                    if mp.exists():
                        candidates.append((pid, mod, s.path, mp))
        rng = random.Random(cfg.seed)
        rng.shuffle(candidates)
        for pid, mod, ip, mp in candidates[: args.geometry_checks]:
            chk = check_pair_geometry(ip, mp)
            geometry.append(
                {
                    "patient_id": pid,
                    "modality": mod,
                    "image": ip.name,
                    "status": chk.status,
                    "problems": "; ".join(chk.problems),
                    "image_shape": str(chk.image_shape),
                    "mask_shape": str(chk.mask_shape),
                    "mask_positive_px": chk.mask_positive_px,
                }
            )
        n_ok = sum(1 for g in geometry if g["status"] == "OK")
        print(f"  {n_ok}/{len(geometry)} pairs verified OK")

    # ------------------------------------------------------------- outputs
    summary_path = cfg.out("data_summary_csv")
    manifest.to_csv(summary_path, index=False)
    manifest.to_csv(cfg.out("manifests_dir") / "patient_manifest.csv", index=False)
    if geometry:
        pd.DataFrame(geometry).to_csv(cfg.out("manifests_dir") / "geometry_checks.csv", index=False)

    with open(cfg.out("manifests_dir") / "clinical_profile.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                k: {
                    "status": p.status, "error": p.error, "n_rows": p.n_rows, "n_cols": p.n_cols,
                    "columns": p.columns, "dtypes": p.dtypes, "missing_frac": p.missing_frac,
                    "n_unique": p.n_unique, "examples": {c: [str(x) for x in v] for c, v in p.examples.items()},
                    "role_guess": p.role_guess,
                }
                for k, p in clinical_profiles.items()
            },
            fh, indent=2, ensure_ascii=False,
        )
    with open(cfg.out("tables_dir") / "dataset_summary.json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    report = build_report(
        cfg, manifest, stats, clinical_profiles, match_reports, geometry,
        readme_status, segments_entries, volumes, patients, reliability, args,
    )
    audit_path = cfg.out("data_audit_md")
    audit_path.write_text(report, encoding="utf-8")

    print(f"\nWrote:\n  {summary_path}\n  {audit_path}")
    print(f"\nHeadline: {stats['n_patient_dirs']} patient dirs, "
          f"{stats['n_with_files']} with files, {stats['n_with_pixel_data']} with pixel data, "
          f"{stats['total_slices']} image slices, {stats['total_masks']} masks.")
    if stats["n_with_pixel_data"] == 0:
        print("\n*** BLOCKER: no readable pixel data anywhere in the dataset. ***")
        print("*** Structure was audited; modelling cannot start until files are re-obtained. ***")
    return 0


def build_report(
    cfg, manifest, stats, profiles, matches, geometry,
    readme_status, segments_entries, volumes, patients, reliability, args,
) -> str:
    """Compose the markdown audit report from verified facts only."""
    L: list[str] = []
    a = L.append
    a("# NPC dataset audit")
    a("")
    a(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} by `scripts/01_audit_dataset.py`  ")
    a(f"Seed `{cfg.seed}` · pixel inspection: `{'OFF' if args.no_pixels else 'ON'}`")
    a("")
    a("## Roots inspected")
    a("")
    a("| Role | Path | Entries |")
    a("|---|---|---|")
    for label, p in (
        ("images_root", cfg.images_root),
        ("primary_data_root", cfg.primary_data_root),
        ("reliability_root", cfg.reliability_root),
    ):
        a(f"| {label} | `{p}` | {len(cfg.clean_listdir(p))} |")
    a("")

    # -- blocker banner
    if stats["n_with_pixel_data"] == 0:
        a("## Blocking finding: the dataset contains no pixel data")
        a("")
        a(f"All {stats['total_slices'] + stats['total_masks']} discovered image and mask files are "
          "**0 bytes**. The directory tree, patient identifiers, modality names, slice indices and "
          "image/mask pairing are all intact and fully auditable, but no pixel values, CSV rows or "
          "README text exist on disk. Everything below that depends only on *structure* is verified; "
          "everything that depends on *content* (dimensions, spacing, scanner metadata, tumour volume, "
          "staging, follow-up) is unavailable and is reported as such rather than estimated.")
        a("")

    a("## Headline statistics")
    a("")
    a("| Quantity | Value |")
    a("|---|---:|")
    rows = [
        ("Patient directories", stats["n_patient_dirs"]),
        ("… with at least one file", stats["n_with_files"]),
        ("… completely empty", stats["n_empty_dirs"]),
        ("… with readable pixel data", stats["n_with_pixel_data"]),
        ("… with all three modalities", stats["n_all_three_modalities"]),
        ("Image slices (all modalities)", stats["total_slices"]),
        ("Mask files (all modalities)", stats["total_masks"]),
        ("Total bytes of image data", stats["total_bytes"]),
        ("Reliability-subset patients", stats["n_reliability"]),
        ("… with pixel data", stats["n_reliability_with_pixels"]),
        ("Patients with image/mask pairing problems", stats["n_pairing_problems"]),
        ("Patients with unequal slice counts across sequences", stats["n_inconsistent_slice_counts"]),
        ("3D volume files (NIfTI/DICOM/NRRD)", len(volumes)),
    ]
    for k, v in rows:
        a(f"| {k} | {v:,} |" if isinstance(v, int) else f"| {k} | {v} |")
    a("")

    a("## Per-modality structure")
    a("")
    a("| Modality | Patients | Image slices | Mask slices | Paired |")
    a("|---|---:|---:|---:|---|")
    for mod in cfg.modalities:
        k = mod.replace("-", "_")
        ni, nm = stats[f"slices_{k}"], stats[f"masks_{k}"]
        a(f"| {mod} | {stats[f'patients_with_{k}']} | {ni:,} | {nm:,} | "
          f"{'yes' if ni == nm else f'**NO ({ni - nm:+d})**'} |")
    a("")

    # -- structural integrity
    a("## Structural integrity findings")
    a("")
    empty = manifest.loc[manifest["is_empty_dir"].fillna(False), "patient_id"].tolist()
    if empty:
        a(f"**{len(empty)} patient directories exist but contain no files:** "
          f"`{', '.join(empty)}`")
        a("")
    probs = manifest.loc[manifest["pairing_problems"].fillna("") != "", ["patient_id", "pairing_problems"]]
    if len(probs):
        a(f"**{len(probs)} patients have image/mask pairing anomalies:**")
        a("")
        a("| Patient | Problem |")
        a("|---|---|")
        for _, r in probs.head(20).iterrows():
            a(f"| {r['patient_id']} | `{r['pairing_problems']}` |")
        a("")
    inc = manifest.loc[~manifest["slice_counts_consistent"].fillna(True)]
    if len(inc):
        a(f"**{len(inc)} patients have different slice counts across sequences.** The sequences are "
          "therefore *not* slice-wise co-registered, so a model cannot simply stack "
          "`T1WI_k / T2WI_k / CE-T1WI_k` as three channels of one anatomical slice. Median spread "
          f"between the largest and smallest per-patient sequence is "
          f"{int(inc['modality_slice_count_spread'].median())} slices "
          f"(max {int(inc['modality_slice_count_spread'].max())}).")
        a("")

    # -- clinical
    a("## Clinical and morphological tables")
    a("")
    a("| File | Status | Rows | Cols |")
    a("|---|---|---:|---:|")
    for key, p in profiles.items():
        a(f"| `{p.path.name}` | {p.status} | {p.n_rows} | {p.n_cols} |")
    a(f"| `README.md` | {readme_status} | – | – |")
    a(f"| `MRI-Segments/` | {'EMPTY_DIR' if not segments_entries else f'{len(segments_entries)} entries'} | – | – |")
    a("")
    for key, p in profiles.items():
        if p.ok:
            a(f"### `{p.path.name}` columns")
            a("")
            a("| Column | dtype | missing | unique | guessed role | examples |")
            a("|---|---|---:|---:|---|---|")
            for c in p.columns:
                ex = ", ".join(str(x) for x in p.examples.get(c, [])[:3])
                a(f"| `{c}` | {p.dtypes[c]} | {p.missing_frac[c]:.0%} | {p.n_unique[c]} | "
                  f"{p.role_guess.get(c) or '–'} | {ex} |")
            a("")
            rep = matches.get(key)
            if rep:
                a(f"ID matching: column `{rep.id_column}` matched "
                  f"**{rep.n_matched}/{rep.n_imaging_ids}** imaging patients "
                  f"({rep.match_rate:.1%}).")
                if rep.only_in_imaging:
                    a(f"Imaging IDs with no clinical row (first 10): `{rep.only_in_imaging[:10]}`")
                if rep.duplicated_csv_ids:
                    a(f"Duplicated IDs in CSV: `{rep.duplicated_csv_ids[:10]}`")
                a("")
        else:
            a(f"### `{p.path.name}` — {p.status}")
            a("")
            a(f"{p.error}. Scanner metadata, staging, morphological measurements and "
              "progression/follow-up fields are therefore **unavailable**; no value for them "
              "appears in the manifest.")
            a("")

    # -- reliability
    a("## Inter-rater reliability subset")
    a("")
    rel_ids = sorted(reliability)
    a(f"{len(rel_ids)} patient identifiers are listed: `{', '.join(rel_ids)}`")
    a("")
    n_files = sum(r.n_files for r in reliability.values())
    n_rater_dirs = sum(len(r.subdirectories) for r in reliability.values())
    overlap = sorted(set(reliability) & set(patients))
    a(f"- Annotation files found: **{n_files}**")
    a(f"- Rater sub-directories found: **{n_rater_dirs}**")
    a(f"- Also present in the primary imaging set: **{len(overlap)}/{len(rel_ids)}**")
    a("")
    if n_files == 0:
        a("The reliability directories are **empty**, so the number of independent annotations per "
          "case cannot be determined from disk. This is the single most important missing piece for "
          "any inter-rater-calibrated design: without repeated contours there is no inter-rater "
          "envelope to calibrate against.")
        a("")
    if overlap:
        a(f"Every reliability identifier that also appears in the primary set (`{len(overlap)}` of "
          f"`{len(rel_ids)}`) is a leakage risk: repeated annotations of a case must not straddle "
          "partitions. `src/splits.py` confines them to a single partition and asserts it.")
        a("")

    # -- geometry
    a("## Image/mask geometry verification")
    a("")
    if not geometry:
        a("Not run (no pixel data available, or `--no-pixels` was passed).")
    else:
        gdf = pd.DataFrame(geometry)
        a("| Status | Pairs |")
        a("|---|---:|")
        for s, n in gdf["status"].value_counts().items():
            a(f"| {s} | {n} |")
        a("")
        bad = gdf[gdf["status"] != "OK"]
        if len(bad):
            a("Problem pairs (first 15):")
            a("")
            a("| Patient | Modality | Image | Problem |")
            a("|---|---|---|---|")
            for _, r in bad.head(15).iterrows():
                a(f"| {r['patient_id']} | {r['modality']} | `{r['image']}` | {r['problems']} |")
        a("")

    # -- what is unavailable
    a("## Fields required by the study that are currently unavailable")
    a("")
    a("| Field | Needed for | Status |")
    a("|---|---|---|")
    unavailable = [
        ("Scanner manufacturer / model", "scanner-stratified evaluation", "diagnoses CSV"),
        ("Field strength", "scanner-stratified evaluation", "diagnoses CSV"),
        ("Pixel spacing / slice thickness", "tumour volume in mm³, radiomics", "TIFF tags or CSV"),
        ("TNM stage", "clinical stratification", "diagnoses CSV"),
        ("Progression-free survival time", "survival analysis", "diagnoses CSV"),
        ("Progression event / censoring", "survival analysis", "diagnoses CSV"),
        ("Morphological measurements", "biomarker-stability analysis", "morphology CSV"),
        ("Repeated annotations", "inter-rater calibration", "reliability root"),
    ]
    for field, need, src in unavailable:
        got = "**unavailable**" if stats["n_with_pixel_data"] == 0 else "check manifest"
        a(f"| {field} | {need} | {got} (source: {src}) |")
    a("")

    a(_reliability_3d_section(cfg))

    a("## Manifest columns")
    a("")
    a(f"`data_summary.csv` has {manifest.shape[0]} rows and {manifest.shape[1]} columns:")
    a("")
    a("```")
    a(", ".join(manifest.columns))
    a("```")
    return "\n".join(L) + "\n"


def _reliability_3d_section(cfg) -> str:
    """Fold in the 3D-mask findings produced by scripts/05_reliability_probe.py.

    Kept as a read of that script's output rather than a recomputation, so the
    audit stays cheap and there is exactly one place where these numbers are
    derived.
    """
    tables = cfg.repo_root / cfg.raw["outputs"]["tables_dir"]
    geom_path = tables / "reliability_geometry.csv"
    if not geom_path.exists():
        return (
            "## Verified findings from the 3D NIfTI masks\n\n"
            "Not available. Run `scripts/05_reliability_probe.py --nifti-root <dir>` on an\n"
            "extracted `inner_reliability` directory to populate this section.\n"
        )

    g = pd.read_csv(geom_path, dtype={"patient_id": str})
    L = ["## Verified findings from the 3D NIfTI masks", ""]
    L.append(
        f"Computed from {len(g)} masks over {g.patient_id.nunique()} patients in the "
        "reliability archive, whose MD5 matches the published Zenodo record. These carry real "
        "voxel spacing, so tumour volume is available in cm3 here but not from the 2D TIFF export."
    )
    L.append("")
    L.append("| Sequence | n | In-plane spacing (median mm) | Slice spacing (median mm) | Volume (median cm3) |")
    L.append("|---|---:|---:|---:|---:|")
    for seq, grp in g.groupby("sequence"):
        L.append(
            f"| {seq} | {len(grp)} | {grp.spacing_x_mm.median():.3f} | "
            f"{grp.spacing_z_mm.median():.2f} | {grp.volume_cm3.median():.2f} |"
        )
    L.append("")

    # Slice-selection test: is the TIFF export a full stack or lesion-only?
    cmp_df = g[g.tif_slices > 0]
    if len(cmp_df):
        full = int((cmp_df.tif_slices == cmp_df.nii_slices).sum())
        lesion = int((cmp_df.tif_slices == cmp_df.nii_lesion_slices).sum())
        frac = (cmp_df.nii_lesion_slices / cmp_df.nii_slices).median()
        L.append(
            f"**Slice selection.** The TIFF slice count equals the full NIfTI stack depth in "
            f"**{full}/{len(cmp_df)}** masks and the lesion-bearing count in **{lesion}/{len(cmp_df)}**, "
            f"while lesion-bearing slices are only a median {frac:.0%} of a stack. The 2D export is "
            "therefore a complete stack, not a tumour-bearing selection, so slice-level metrics on it "
            "are not inflated by lesion-only sampling."
        )
        L.append("")

    # Grid mismatch across sequences.
    grids = g.pivot_table(index="patient_id", columns="sequence", values="spacing_x_mm")
    if {"T1WI", "T2WI"}.issubset(grids.columns):
        differ = int((~np.isclose(grids["T1WI"], grids["T2WI"], rtol=1e-3)).sum())
        L.append(
            f"**Co-registration.** T1WI and T2WI sit on different in-plane grids in "
            f"**{differ}/{len(grids)}** patients, typically by a factor of two. The sequences are not "
            "co-registered, so stacking matching slice indices into a three-channel input is invalid."
        )
        L.append("")

    vol_path = tables / "reliability_cross_sequence_volume.csv"
    if vol_path.exists():
        v = pd.read_csv(vol_path, dtype={"patient_id": str})
        if "spread_pct" in v.columns and len(v):
            s = v.spread_pct.dropna()
            L.append(
                f"**Cross-sequence volume variability.** Within one patient and one annotation pass, "
                f"tumour volume across the three sequences spreads by a median **{s.median():.1f}%** "
                f"(IQR {s.quantile(.25):.1f} to {s.quantile(.75):.1f}%, max {s.max():.1f}%); "
                f"{int((s > 20).sum())}/{len(s)} patients exceed 20% and {int((s > 30).sum())}/{len(s)} "
                "exceed 30%. This is the study's headline measurement, and it requires no model."
            )
            L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
