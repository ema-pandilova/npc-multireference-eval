"""Image/mask overlay rendering for visual verification of alignment.

Overlays are the only reliable way to catch a mask that is geometrically valid
but semantically wrong (shifted, mirrored, or belonging to a different
sequence). When no pixel data is available the renderer says so explicitly on
the figure rather than emitting a blank panel that could be mistaken for a
correctly-rendered empty mask.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .discovery import PatientRecord  # noqa: E402
from .imaging import load_pair, load_pixels, resample_mask_to_image, to_2d  # noqa: E402
from .paths import Config  # noqa: E402


def _normalise(img: np.ndarray) -> np.ndarray:
    """Robust 1-99 percentile window; MRI intensities have no fixed range."""
    img = img.astype(np.float32)
    if img.ndim == 3:
        img = img.mean(axis=-1)
    lo, hi = np.percentile(img, [1, 99])
    if hi <= lo:
        lo, hi = float(img.min()), float(img.max())
    if hi <= lo:
        return np.zeros_like(img)
    return np.clip((img - lo) / (hi - lo), 0, 1)


def _pick_slices(rec: PatientRecord, modality: str, n: int = 3) -> list:
    """Choose the slices with the largest lesion; those show alignment best."""
    imgs = [s for s in rec.images(modality) if not s.is_empty]
    if not imgs:
        return []
    scored = []
    for s in imgs:
        mask_path = s.path.with_name(f"{s.path.stem}_mask{s.path.suffix}")
        arr = load_pixels(mask_path)
        # Lesion size is counted on the mask's own grid; only relative ordering
        # matters for choosing which slices to display.
        scored.append((int((to_2d(arr) > 0).sum()) if arr is not None else -1, s))
    scored.sort(key=lambda t: t[0], reverse=True)
    chosen = [s for score, s in scored if score > 0][:n]
    # Fall back to the middle slice when no lesion pixels were found anywhere.
    return chosen or [imgs[len(imgs) // 2]]


def render_patient_overlay(
    cfg: Config, rec: PatientRecord, out_path: Path, n_slices: int = 3
) -> tuple[bool, str]:
    """Render one figure per patient: rows = modalities, cols = selected slices.

    Returns (success, message). success=False means no pixels were renderable;
    a diagnostic figure is still written so the failure is visible in results/.
    """
    mods = [m for m in cfg.modalities if m in rec.modalities_present]
    if not mods:
        return _diagnostic_figure(out_path, rec.patient_id, "no slice files found for this patient")

    if not rec.has_pixel_data:
        return _diagnostic_figure(
            out_path,
            rec.patient_id,
            f"{rec.n_files} files present but ALL are 0 bytes\nno pixel data to overlay",
        )

    panels = {m: _pick_slices(rec, m, n_slices) for m in mods}
    ncols = max((len(v) for v in panels.values()), default=0)
    if ncols == 0:
        return _diagnostic_figure(out_path, rec.patient_id, "no readable slices")

    fig, axes = plt.subplots(
        len(mods), ncols, figsize=(3.2 * ncols, 3.4 * len(mods)), squeeze=False
    )
    rendered, n_resampled = 0, 0
    for r, mod in enumerate(mods):
        for c in range(ncols):
            ax = axes[r][c]
            ax.axis("off")
            slices = panels[mod]
            if c >= len(slices):
                continue
            s = slices[c]
            # load_pair puts the mask on the image grid. Without this the mask is
            # drawn in its own (larger, native) coordinate system and appears
            # outside the head entirely, which is exactly how this dataset's
            # image/mask grid defect first became visible.
            img, mask, status = load_pair(
                s.path, s.path.with_name(f"{s.path.stem}_mask{s.path.suffix}")
            )
            if img is None:
                ax.text(0.5, 0.5, "unreadable", ha="center", va="center", fontsize=8)
                continue
            ax.imshow(_normalise(img), cmap="gray", interpolation="nearest")
            npx = 0
            if mask is not None:
                npx = int(mask.sum())
                if status == "MASK_RESAMPLED":
                    n_resampled += 1
                if npx:
                    ax.contour(mask, levels=[0.5], colors="#ff2d55", linewidths=1.2)
                    ax.imshow(
                        np.ma.masked_where(~mask, mask),
                        cmap="autumn",
                        alpha=0.25,
                        interpolation="nearest",
                    )
            flag = "*" if status == "MASK_RESAMPLED" else ""
            ax.set_title(f"{mod} #{s.index}{flag}  ({npx} px)", fontsize=8)
            rendered += 1

    note = (
        f"  ({n_resampled}/{rendered} masks resampled to the image grid, marked *)"
        if n_resampled else ""
    )
    fig.suptitle(
        f"Patient {rec.patient_id} — image/mask alignment check{note}", fontsize=11
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return (rendered > 0), f"rendered {rendered} panels"


def _diagnostic_figure(out_path: Path, patient_id: str, message: str) -> tuple[bool, str]:
    """Write a figure that states why an overlay could not be produced."""
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.axis("off")
    ax.text(
        0.5, 0.62, f"Patient {patient_id}", ha="center", va="center", fontsize=14, weight="bold"
    )
    ax.text(0.5, 0.34, message, ha="center", va="center", fontsize=10, color="#b00020")
    ax.text(
        0.5, 0.08, "OVERLAY NOT POSSIBLE", ha="center", va="center", fontsize=9, color="#666"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return False, message


def render_rater_comparison(
    cfg: Config, rec: PatientRecord, rel: PatientRecord, out_path: Path
) -> tuple[bool, str]:
    """Overlay the independent annotation sets of one reliability case.

    This is the figure that shows the genuine inter-rater envelope the study
    calibrates against, so it is worth rendering as soon as data allows.
    """
    raters = sorted({m.split("::")[0] for m in rel.modalities_present if "::" in m})
    if not raters:
        return _diagnostic_figure(
            out_path,
            rec.patient_id,
            f"reliability entry has {rel.n_files} files and no\ndistinguishable rater sub-folders",
        )
    if not rel.has_pixel_data:
        return _diagnostic_figure(
            out_path, rec.patient_id, f"{len(raters)} rater folders but no pixel data"
        )

    colours = ["#ff2d55", "#00b4d8", "#ffd60a", "#06d6a0", "#b388ff"]
    base_mod = next((m for m in cfg.modalities if m in rec.modalities_present), None)
    imgs = [s for s in rec.images(base_mod)] if base_mod else []
    if not imgs:
        return _diagnostic_figure(out_path, rec.patient_id, "no base image to draw contours on")

    ref = imgs[len(imgs) // 2]
    img = load_pixels(ref.path)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.axis("off")
    if img is not None:
        ax.imshow(_normalise(img), cmap="gray")
    drawn = 0
    for i, rater in enumerate(raters):
        cand = [
            s for s in rel.slices
            if s.is_mask and s.modality.startswith(f"{rater}::") and s.index == ref.index
        ]
        for s in cand:
            arr = load_pixels(s.path)
            if arr is None:
                continue
            m = arr.mean(axis=-1) if arr.ndim == 3 else arr
            if (m > 0).any():
                ax.contour(m > 0, levels=[0.5], colors=[colours[i % len(colours)]], linewidths=1.4)
                drawn += 1
    ax.set_title(
        f"Patient {rec.patient_id} — {len(raters)} independent annotations (slice {ref.index})",
        fontsize=10,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return drawn > 0, f"drew {drawn} rater contours"
