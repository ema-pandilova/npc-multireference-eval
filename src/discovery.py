"""Directory, patient, modality and slice discovery.

The dataset stores one directory per patient, containing 2D slices as TIFFs with
the filename grammar

    <MODALITY>_<SLICE:03d>.tif          image slice
    <MODALITY>_<SLICE:03d>_mask.tif     matching binary tumour mask

This module turns that layout into structured records without reading pixel
data, so it stays fast and works even when files are unreadable or empty. File
*size* is recorded for every entry, because a 0-byte file is a distinct and
important state: the name promises data that is not there.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .paths import Config

# Modality tokens contain a hyphen (CE-T1WI), so the slice index anchors the parse.
SLICE_RE = re.compile(r"^(?P<modality>.+?)_(?P<idx>\d+)(?P<mask>_mask)?\.(?P<ext>tif|tiff|png|nii|nii\.gz)$", re.IGNORECASE)

VOLUME_EXTS = {".nii", ".gz", ".mha", ".mhd", ".nrrd", ".dcm"}


@dataclass
class SliceFile:
    path: Path
    modality: str
    index: int
    is_mask: bool
    size_bytes: int

    @property
    def is_empty(self) -> bool:
        return self.size_bytes == 0


@dataclass
class PatientRecord:
    """Everything discoverable about one patient from the filesystem alone."""

    patient_id: str
    directory: Path
    slices: list[SliceFile] = field(default_factory=list)
    unparsed_files: list[str] = field(default_factory=list)
    subdirectories: list[str] = field(default_factory=list)

    # -- derived views ---------------------------------------------------
    def images(self, modality: str) -> list[SliceFile]:
        return sorted(
            (s for s in self.slices if s.modality == modality and not s.is_mask),
            key=lambda s: s.index,
        )

    def masks(self, modality: str) -> list[SliceFile]:
        return sorted(
            (s for s in self.slices if s.modality == modality and s.is_mask),
            key=lambda s: s.index,
        )

    @property
    def modalities_present(self) -> list[str]:
        return sorted({s.modality for s in self.slices})

    @property
    def n_files(self) -> int:
        return len(self.slices)

    @property
    def total_bytes(self) -> int:
        return sum(s.size_bytes for s in self.slices)

    @property
    def is_empty_dir(self) -> bool:
        """No parseable slice files at all."""
        return not self.slices

    @property
    def has_pixel_data(self) -> bool:
        """At least one file that actually carries bytes."""
        return self.total_bytes > 0

    def pairing(self, modality: str) -> dict:
        """Image/mask correspondence for one modality, by slice index."""
        img_idx = {s.index for s in self.images(modality)}
        msk_idx = {s.index for s in self.masks(modality)}
        return {
            "n_images": len(img_idx),
            "n_masks": len(msk_idx),
            "images_without_mask": sorted(img_idx - msk_idx),
            "masks_without_image": sorted(msk_idx - img_idx),
            "index_gaps": _gaps(img_idx),
        }


def _gaps(indices: set[int]) -> list[int]:
    """Missing slice indices inside the observed range (non-contiguous stacks)."""
    if not indices:
        return []
    lo, hi = min(indices), max(indices)
    return sorted(set(range(lo, hi + 1)) - indices)


# Whole-volume ROI masks, as shipped in inner_reliability.zip and in
# primary_data's MRI-Segments: one 3D NIfTI per sequence, with no slice index.
ROI_RE = re.compile(r"^(?P<stem>ROI-(?:CE-T1|T1|T2))\.(?:nii|nii\.gz)$", re.IGNORECASE)

ROI_STEM_TO_MODALITY = {"ROI-T1": "T1WI", "ROI-T2": "T2WI", "ROI-CE-T1": "CE-T1WI"}


def parse_slice_filename(name: str) -> tuple[str, int, bool] | None:
    """Return (modality, slice_index, is_mask) or None if the name does not match.

    Handles both naming schemes in this dataset: the per-slice TIFF grammar
    `<MODALITY>_<idx>[_mask].tif`, and the whole-volume `ROI-<SEQ>.nii` masks.
    A ROI volume has no slice index, so it is reported with index -1 and
    is_mask=True; callers distinguish it by that sentinel.
    """
    roi = ROI_RE.match(name)
    if roi:
        stem = roi.group("stem")
        canonical = next(
            (k for k in ROI_STEM_TO_MODALITY if k.lower() == stem.lower()), stem
        )
        return ROI_STEM_TO_MODALITY.get(canonical, canonical), -1, True

    m = SLICE_RE.match(name)
    if not m:
        return None
    return m.group("modality"), int(m.group("idx")), m.group("mask") is not None


def discover_patients(cfg: Config, root: Path) -> dict[str, PatientRecord]:
    """Walk one dataset root and build a PatientRecord per patient directory."""
    records: dict[str, PatientRecord] = {}
    if not root.is_dir():
        return records

    for entry in cfg.clean_listdir(root):
        pdir = root / entry
        if not pdir.is_dir():
            continue

        rec = PatientRecord(patient_id=entry, directory=pdir)
        for fname in cfg.clean_listdir(pdir):
            fpath = pdir / fname
            if fpath.is_dir():
                rec.subdirectories.append(fname)
                continue
            parsed = parse_slice_filename(fname)
            if parsed is None:
                rec.unparsed_files.append(fname)
                continue
            modality, idx, is_mask = parsed
            try:
                size = fpath.stat().st_size
            except OSError:
                size = -1
            rec.slices.append(
                SliceFile(path=fpath, modality=modality, index=idx, is_mask=is_mask, size_bytes=size)
            )
        records[entry] = rec

    return records


def discover_reliability(cfg: Config) -> dict[str, PatientRecord]:
    """Discover the inter-rater subset.

    This root may nest one level deeper than the primary images root (one
    directory per rater inside each patient), so subdirectories are walked and
    the rater label is recorded as a prefix on the modality. If no nesting is
    present the layout matches the primary root exactly.
    """
    records = discover_patients(cfg, cfg.reliability_root)

    for pid, rec in records.items():
        for sub in list(rec.subdirectories):
            subdir = rec.directory / sub
            for fname in cfg.clean_listdir(subdir):
                fpath = subdir / fname
                if fpath.is_dir():
                    continue
                parsed = parse_slice_filename(fname)
                if parsed is None:
                    rec.unparsed_files.append(f"{sub}/{fname}")
                    continue
                modality, idx, is_mask = parsed
                try:
                    size = fpath.stat().st_size
                except OSError:
                    size = -1
                # Namespace the modality with the rater/sub-folder label so that
                # repeated annotations of the same slice never collide.
                rec.slices.append(
                    SliceFile(
                        path=fpath,
                        modality=f"{sub}::{modality}",
                        index=idx,
                        is_mask=is_mask,
                        size_bytes=size,
                    )
                )
    return records


def count_raters(rec: PatientRecord) -> int:
    """Number of independent annotation sets discoverable for one patient.

    Counts distinct rater sub-folders when present; otherwise falls back to 1 if
    any mask exists at the patient level, and 0 if nothing is there.

    Note for this dataset: `inner_reliability` holds ONE set of three ROI volumes
    per patient, so this returns 1. The second reader lives in primary_data's
    MRI-Segments, and pairing the two roots is what yields two readers. Counting
    within a single root will never report 2.
    """
    raters = {m.split("::")[0] for m in rec.modalities_present if "::" in m}
    if raters:
        return len(raters)
    return 1 if any(s.is_mask for s in rec.slices) else 0


def roi_volumes(rec: PatientRecord) -> dict[str, Path]:
    """Map modality -> ROI volume path for whole-volume masks (index -1)."""
    return {s.modality: s.path for s in rec.slices if s.is_mask and s.index == -1}


def scan_volume_files(cfg: Config, root: Path) -> list[Path]:
    """Find any 3D volume files (NIfTI/DICOM/NRRD) anywhere under a root.

    The published release is 2D TIFF, but a fuller volumetric release would
    change the study design substantially, so we check rather than assume.
    """
    found: list[Path] = []
    if not root.is_dir():
        return found
    for path in root.rglob("*"):
        if any(part in cfg.ignore_dir_names for part in path.parts):
            continue
        if not path.is_file() or cfg.is_noise(path.name):
            continue
        if path.suffix.lower() in VOLUME_EXTS or path.name.lower().endswith(".nii.gz"):
            found.append(path)
    return found
