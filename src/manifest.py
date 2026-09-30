"""Patient-level manifest construction.

One row per patient, combining filesystem structure, image geometry, clinical
fields and reliability-subset membership. This manifest is the single source of
truth for every downstream step (splitting, training, analysis), so it records
data-quality flags alongside the data itself rather than dropping problem cases
silently.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .discovery import PatientRecord, count_raters
from .imaging import load_pixels, read_any
from .paths import Config


def _fmt_list(values) -> str:
    return "|".join(str(v) for v in values)


def build_patient_row(
    cfg: Config,
    rec: PatientRecord,
    reliability: dict[str, PatientRecord],
    inspect_pixels: bool = True,
    max_slices_to_inspect: int = 3,
) -> dict:
    """Assemble one manifest row for a single patient."""
    row: dict = {"patient_id": rec.patient_id}

    modalities = [m for m in cfg.modalities if m in rec.modalities_present]
    row["n_modalities"] = len(modalities)
    row["modalities_present"] = _fmt_list(modalities)
    row["has_all_modalities"] = len(modalities) == len(cfg.modalities)
    row["is_empty_dir"] = rec.is_empty_dir
    row["has_pixel_data"] = rec.has_pixel_data
    row["total_bytes"] = rec.total_bytes
    row["n_files"] = rec.n_files
    row["unparsed_files"] = _fmt_list(rec.unparsed_files[:5])

    exts = {p.path.suffix.lower() for p in rec.slices}
    row["image_format"] = _fmt_list(sorted(exts)) or "NONE"

    # -- per-modality slice / mask structure -----------------------------
    total_img = total_msk = 0
    pairing_problems: list[str] = []
    for mod in cfg.modalities:
        pair = rec.pairing(mod)
        key = mod.replace("-", "_")
        row[f"n_slices_{key}"] = pair["n_images"]
        row[f"n_masks_{key}"] = pair["n_masks"]
        total_img += pair["n_images"]
        total_msk += pair["n_masks"]
        if pair["images_without_mask"]:
            pairing_problems.append(f"{mod}:img_no_mask={pair['images_without_mask']}")
        if pair["masks_without_image"]:
            pairing_problems.append(f"{mod}:mask_no_img={pair['masks_without_image']}")
        if pair["index_gaps"]:
            pairing_problems.append(f"{mod}:index_gaps={pair['index_gaps'][:5]}")

    row["n_slices_total"] = total_img
    row["n_masks_total"] = total_msk
    row["mask_availability"] = (
        "COMPLETE" if total_img and total_img == total_msk
        else "PARTIAL" if total_msk
        else "NONE"
    )
    row["pairing_problems"] = _fmt_list(pairing_problems[:5])

    # Slice counts differing across sequences means the sequences are NOT
    # slice-wise co-registered; flag it, because it constrains any 3-channel
    # multimodal model that naively stacks corresponding slice indices.
    counts = {row[f"n_slices_{m.replace('-', '_')}"] for m in cfg.modalities}
    counts.discard(0)
    row["slice_counts_consistent"] = len(counts) <= 1
    row["modality_slice_count_spread"] = (max(counts) - min(counts)) if counts else 0

    # -- reliability-subset membership -----------------------------------
    rel = reliability.get(rec.patient_id)
    row["in_reliability_subset"] = rel is not None
    row["n_rater_annotation_sets"] = count_raters(rel) if rel is not None else 0
    row["reliability_files"] = rel.n_files if rel is not None else 0
    row["reliability_has_pixel_data"] = bool(rel.has_pixel_data) if rel is not None else False

    # -- geometry / spacing / tumour burden ------------------------------
    row.update(_inspect_geometry(cfg, rec, inspect_pixels, max_slices_to_inspect))
    return row


def _inspect_geometry(
    cfg: Config, rec: PatientRecord, inspect_pixels: bool, max_slices: int
) -> dict:
    """Read a few slices to recover dimensions, spacing and tumour burden.

    Returns explicit nulls (not zeros) when pixels are unavailable, so that a
    dataset with no readable pixels never masquerades as a dataset of empty
    tumours.
    """
    out: dict = {
        "image_height": np.nan,
        "image_width": np.nan,
        "image_dtype": None,
        "pixel_spacing_mm": None,
        "spacing_source": "UNAVAILABLE",
        "tumour_pixels_total": np.nan,
        "tumour_slices_with_lesion": np.nan,
        "tumour_area_px_per_slice_mean": np.nan,
        "geometry_status": "NOT_INSPECTED",
    }
    if not inspect_pixels:
        return out

    readable = [s for s in rec.slices if not s.is_empty]
    if not readable:
        out["geometry_status"] = "NO_PIXEL_DATA" if rec.slices else "NO_FILES"
        return out

    # Dimensions from the first readable image slice of the primary sequence.
    ref = None
    for mod in cfg.modalities:
        imgs = [s for s in rec.images(mod) if not s.is_empty]
        if imgs:
            ref = imgs[len(imgs) // 2]
            break
    if ref is not None:
        meta = read_any(ref.path, load_pixels=False)
        if meta.ok and meta.shape:
            out["image_height"] = int(meta.shape[0])
            out["image_width"] = int(meta.shape[1]) if len(meta.shape) > 1 else np.nan
            out["image_dtype"] = meta.dtype
            if meta.spacing:
                out["pixel_spacing_mm"] = "x".join(f"{v:.4f}" for v in meta.spacing)
                out["spacing_source"] = "TIFF_RESOLUTION_TAG"
        out["geometry_status"] = meta.status

    # Tumour burden in PIXELS. Deliberately not converted to mm^3: without
    # verified pixel spacing and slice thickness, a millimetre volume would be
    # a fabricated number. Conversion happens only if spacing is recovered.
    total_px, lesion_slices, per_slice = 0, 0, []
    for mod in cfg.modalities:
        for m in rec.masks(mod):
            if m.is_empty:
                continue
            arr = load_pixels(m.path)
            if arr is None:
                continue
            pos = int((arr > 0).sum())
            total_px += pos
            per_slice.append(pos)
            if pos > 0:
                lesion_slices += 1
    if per_slice:
        out["tumour_pixels_total"] = total_px
        out["tumour_slices_with_lesion"] = lesion_slices
        out["tumour_area_px_per_slice_mean"] = float(np.mean(per_slice))

    return out


def manifest_columns(cfg: Config) -> list[str]:
    """Canonical manifest schema.

    Declared explicitly so the manifest has the same columns whether it describes
    277 patients or none. Without this, a dataset root that is still being staged
    produces a column-less frame and every downstream step fails with an opaque
    KeyError instead of an empty-but-valid manifest.
    """
    cols = [
        "patient_id", "n_modalities", "modalities_present", "has_all_modalities",
        "is_empty_dir", "has_pixel_data", "total_bytes", "n_files", "unparsed_files",
        "image_format",
    ]
    for mod in cfg.modalities:
        key = mod.replace("-", "_")
        cols += [f"n_slices_{key}", f"n_masks_{key}"]
    cols += [
        "n_slices_total", "n_masks_total", "mask_availability", "pairing_problems",
        "slice_counts_consistent", "modality_slice_count_spread",
        "in_reliability_subset", "n_rater_annotation_sets", "reliability_files",
        "reliability_has_pixel_data",
        "image_height", "image_width", "image_dtype", "pixel_spacing_mm",
        "spacing_source", "tumour_pixels_total", "tumour_slices_with_lesion",
        "tumour_area_px_per_slice_mean", "geometry_status",
    ]
    return cols


def build_manifest(
    cfg: Config,
    patients: dict[str, PatientRecord],
    reliability: dict[str, PatientRecord],
    inspect_pixels: bool = True,
) -> pd.DataFrame:
    """Build the full patient-level manifest, sorted by patient ID."""
    rows = [
        build_patient_row(cfg, rec, reliability, inspect_pixels=inspect_pixels)
        for _, rec in sorted(patients.items())
    ]

    # Reliability patients with no matching primary-imaging directory are a real
    # integrity question, so they become explicit rows rather than being dropped.
    for pid in sorted(set(reliability) - set(patients)):
        rows.append(
            {
                "patient_id": pid,
                "in_reliability_subset": True,
                "n_rater_annotation_sets": count_raters(reliability[pid]),
                "reliability_files": reliability[pid].n_files,
                "reliability_has_pixel_data": reliability[pid].has_pixel_data,
                "is_empty_dir": True,
                "has_pixel_data": False,
                "geometry_status": "NOT_IN_PRIMARY_IMAGING",
            }
        )

    df = pd.DataFrame(rows, columns=manifest_columns(cfg))
    if df.empty:
        return df
    return df.sort_values("patient_id").reset_index(drop=True)


def add_analysis_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add the eligibility flags that downstream steps filter on."""
    df = df.copy()
    if df.empty:
        for c in ("eligible_imaging", "eligible_modelling"):
            df[c] = pd.Series(dtype=bool)
        df["tumour_burden_stratum"] = pd.Series(dtype=object)
        return df

    df["eligible_imaging"] = (
        (~df["is_empty_dir"].fillna(True))
        & (df["n_slices_total"].fillna(0) > 0)
        & (df["mask_availability"].isin(["COMPLETE", "PARTIAL"]))
    )
    df["eligible_modelling"] = df["eligible_imaging"] & df["has_pixel_data"].fillna(False)

    # Tumour-burden stratum for balanced splitting, computed only where the
    # measurement genuinely exists.
    if df["tumour_pixels_total"].notna().any():
        valid = df["tumour_pixels_total"].dropna()
        if valid.nunique() >= 3:
            df["tumour_burden_stratum"] = pd.qcut(
                df["tumour_pixels_total"], q=3, labels=["low", "mid", "high"], duplicates="drop"
            ).astype(object)
        else:
            df["tumour_burden_stratum"] = None
    else:
        df["tumour_burden_stratum"] = None

    return df


def summarise(df: pd.DataFrame, cfg: Config) -> dict:
    """Headline dataset statistics for the audit report."""
    s: dict = {}
    s["n_patient_dirs"] = int(len(df))
    if df.empty:
        # Report explicit zeros so the audit renders instead of failing; an empty
        # manifest is a legitimate state while the dataset is being staged.
        for k in ("n_empty_dirs", "n_with_files", "n_with_pixel_data",
                  "n_all_three_modalities", "n_reliability", "n_reliability_with_pixels",
                  "total_slices", "total_masks", "total_bytes",
                  "n_inconsistent_slice_counts", "n_pairing_problems"):
            s[k] = 0
        for mod in cfg.modalities:
            key = mod.replace("-", "_")
            s[f"slices_{key}"] = s[f"masks_{key}"] = s[f"patients_with_{key}"] = 0
        return s
    s["n_empty_dirs"] = int(df["is_empty_dir"].fillna(False).sum())
    s["n_with_files"] = int((~df["is_empty_dir"].fillna(True)).sum())
    s["n_with_pixel_data"] = int(df["has_pixel_data"].fillna(False).sum())
    s["n_all_three_modalities"] = int(df["has_all_modalities"].fillna(False).sum())
    s["n_reliability"] = int(df["in_reliability_subset"].fillna(False).sum())
    s["n_reliability_with_pixels"] = int(df["reliability_has_pixel_data"].fillna(False).sum())
    s["total_slices"] = int(df["n_slices_total"].fillna(0).sum())
    s["total_masks"] = int(df["n_masks_total"].fillna(0).sum())
    s["total_bytes"] = int(df["total_bytes"].fillna(0).sum())
    s["n_inconsistent_slice_counts"] = int((~df["slice_counts_consistent"].fillna(True)).sum())
    s["n_pairing_problems"] = int((df["pairing_problems"].fillna("") != "").sum())
    for mod in cfg.modalities:
        key = mod.replace("-", "_")
        s[f"slices_{key}"] = int(df[f"n_slices_{key}"].fillna(0).sum())
        s[f"masks_{key}"] = int(df[f"n_masks_{key}"].fillna(0).sum())
        s[f"patients_with_{key}"] = int((df[f"n_slices_{key}"].fillna(0) > 0).sum())
    return s
