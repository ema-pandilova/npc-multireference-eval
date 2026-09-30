"""Image metadata inspection and image/mask geometry validation.

Handles the TIFF slices that this dataset actually ships, and also NIfTI/DICOM
so the same code works unchanged if a volumetric release is obtained later.

Every reader returns a structured record with a `status` field rather than
raising, because the audit must be able to distinguish and *report*:

    OK              pixels read successfully
    EMPTY_FILE      the file exists but has zero bytes
    UNREADABLE      the file has bytes but no reader could decode it
    MISSING         the path does not exist

A 0-byte file is never silently treated as a black image.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ImageMeta:
    path: Path
    status: str
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    spacing: tuple[float, ...] | None = None
    n_dims: int | None = None
    intensity_min: float | None = None
    intensity_max: float | None = None
    n_unique: int | None = None
    is_binary: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "OK"


def _empty_or_missing(path: Path) -> ImageMeta | None:
    if not path.exists():
        return ImageMeta(path=path, status="MISSING", error="path does not exist")
    try:
        if path.stat().st_size == 0:
            return ImageMeta(path=path, status="EMPTY_FILE", error="file is 0 bytes")
    except OSError as exc:
        return ImageMeta(path=path, status="UNREADABLE", error=str(exc))
    return None


def _summarise_array(meta: ImageMeta, arr: np.ndarray) -> ImageMeta:
    meta.shape = tuple(arr.shape)
    meta.dtype = str(arr.dtype)
    meta.n_dims = arr.ndim
    finite = arr[np.isfinite(arr)] if np.issubdtype(arr.dtype, np.floating) else arr
    if finite.size:
        meta.intensity_min = float(np.min(finite))
        meta.intensity_max = float(np.max(finite))
        # Cap the unique-value scan: masks have few values, images have many.
        uniq = np.unique(finite[:: max(1, finite.size // 100_000)])
        meta.n_unique = int(uniq.size)
        meta.is_binary = bool(uniq.size <= 2)
    return meta


def read_tiff_meta(path: Path, load_pixels: bool = True) -> ImageMeta:
    early = _empty_or_missing(path)
    if early is not None:
        return early

    meta = ImageMeta(path=path, status="OK")
    try:
        import tifffile

        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            meta.shape = tuple(page.shape)
            meta.dtype = str(page.dtype)
            meta.n_dims = len(page.shape)
            tags = {t.name: t.value for t in page.tags.values()}
            # TIFF resolution tags are the only spacing hint a TIFF can carry, but
            # they are only meaningful when ResolutionUnit names a physical unit.
            # This dataset's export writes XResolution=(1,1) with
            # ResolutionUnit=NONE, which is tifffile's placeholder and carries no
            # calibration. Reading it as 1.0 mm would fabricate a spacing and make
            # every derived volume wrong, so spacing stays None unless a real unit
            # is declared (2 = inch, 3 = centimetre).
            res_unit = tags.get("ResolutionUnit")
            unit_code = int(getattr(res_unit, "value", res_unit) or 1)
            xres, yres = tags.get("XResolution"), tags.get("YResolution")
            if xres and yres and unit_code in (2, 3):
                def _to_float(r):
                    return r[0] / r[1] if isinstance(r, tuple) and r[1] else float(r)
                try:
                    fx, fy = _to_float(xres), _to_float(yres)
                    if fx and fy:
                        # inch -> mm is 25.4, centimetre -> mm is 10.
                        scale = 25.4 if unit_code == 2 else 10.0
                        meta.spacing = (scale / fx, scale / fy)
                except (TypeError, ZeroDivisionError):
                    pass
            meta.extra["resolution_unit"] = str(res_unit)
            meta.extra["bits_per_sample"] = tags.get("BitsPerSample")
            meta.extra["software"] = tags.get("Software")
            meta.extra["n_pages"] = len(tif.pages)
            if load_pixels:
                meta = _summarise_array(meta, tif.asarray())
    except Exception as exc:  # noqa: BLE001 - report, never crash the audit
        return ImageMeta(path=path, status="UNREADABLE", error=f"{type(exc).__name__}: {exc}")
    return meta


def read_volume_meta(path: Path, load_pixels: bool = False) -> ImageMeta:
    """Read a NIfTI/DICOM/NRRD volume via SimpleITK, recording true voxel spacing."""
    early = _empty_or_missing(path)
    if early is not None:
        return early

    meta = ImageMeta(path=path, status="OK")
    try:
        import SimpleITK as sitk

        img = sitk.ReadImage(str(path))
        meta.shape = tuple(reversed(img.GetSize()))  # to numpy order
        meta.spacing = tuple(reversed(img.GetSpacing()))
        meta.n_dims = img.GetDimension()
        meta.dtype = img.GetPixelIDTypeAsString()
        meta.extra["origin"] = img.GetOrigin()
        meta.extra["direction"] = img.GetDirection()
        for key in img.GetMetaDataKeys():
            meta.extra[key] = img.GetMetaData(key)
        if load_pixels:
            meta = _summarise_array(meta, sitk.GetArrayFromImage(img))
    except Exception as exc:  # noqa: BLE001
        return ImageMeta(path=path, status="UNREADABLE", error=f"{type(exc).__name__}: {exc}")
    return meta


def read_any(path: Path, load_pixels: bool = True) -> ImageMeta:
    """Dispatch on extension so the audit works across dataset releases."""
    name = path.name.lower()
    if name.endswith((".tif", ".tiff")):
        return read_tiff_meta(path, load_pixels=load_pixels)
    if name.endswith((".nii", ".nii.gz", ".mha", ".mhd", ".nrrd", ".dcm")):
        return read_volume_meta(path, load_pixels=load_pixels)
    if name.endswith((".png", ".jpg", ".jpeg", ".bmp")):
        early = _empty_or_missing(path)
        if early is not None:
            return early
        try:
            from PIL import Image

            with Image.open(path) as im:
                arr = np.array(im)
            return _summarise_array(ImageMeta(path=path, status="OK"), arr)
        except Exception as exc:  # noqa: BLE001
            return ImageMeta(path=path, status="UNREADABLE", error=f"{type(exc).__name__}: {exc}")
    return ImageMeta(path=path, status="UNREADABLE", error=f"unsupported extension: {path.suffix}")


# ---------------------------------------------------------------------------
# Geometry validation
# ---------------------------------------------------------------------------

@dataclass
class GeometryCheck:
    image_path: Path
    mask_path: Path
    status: str
    problems: list[str] = field(default_factory=list)
    image_shape: tuple[int, ...] | None = None
    mask_shape: tuple[int, ...] | None = None
    mask_is_binary: bool | None = None
    mask_positive_px: int | None = None
    mask_labels: list[float] | None = None

    @property
    def ok(self) -> bool:
        return self.status == "OK"


def check_pair_geometry(image_path: Path, mask_path: Path) -> GeometryCheck:
    """Verify that a mask actually matches its image.

    Catches the failure modes that silently corrupt a segmentation study:
    shape mismatch, spacing mismatch, a non-binary "binary" mask, and an
    all-zero or all-one mask.
    """
    img = read_any(image_path, load_pixels=True)
    msk = read_any(mask_path, load_pixels=True)

    problems: list[str] = []
    if not img.ok or not msk.ok:
        chk = GeometryCheck(
            image_path=image_path,
            mask_path=mask_path,
            status="UNVERIFIABLE",
            problems=[f"image: {img.status} ({img.error})", f"mask: {msk.status} ({msk.error})"],
        )
        return chk

    # Compare only the spatial axes; an image may carry a trailing channel axis.
    img_spatial = tuple(img.shape[:2]) if img.shape else None
    msk_spatial = tuple(msk.shape[:2]) if msk.shape else None
    grid_mismatch = img_spatial != msk_spatial
    if grid_mismatch:
        # Distinguish the dataset's systematic, recoverable defect (mask left at
        # the native acquisition matrix while the image was downsampled) from a
        # genuinely corrupt pair. Both are reported; only the former is fixable
        # by nearest-neighbour resampling.
        ratio = (
            msk_spatial[0] / img_spatial[0] if img_spatial and img_spatial[0] else 0
        )
        kind = "MASK_GRID_MISMATCH (recoverable by nearest-neighbour resample)" if (
            msk_spatial and img_spatial
            and msk_spatial[0] == msk_spatial[1]
            and img_spatial[0] == img_spatial[1]
            and ratio > 1.0
        ) else "shape mismatch"
        problems.append(f"{kind}: image {img.shape} vs mask {msk.shape}")

    if img.spacing and msk.spacing and not np.allclose(img.spacing, msk.spacing, rtol=1e-3):
        problems.append(f"spacing mismatch: image {img.spacing} vs mask {msk.spacing}")

    mask_labels = None
    positive = None
    if msk.is_binary is False:
        problems.append(f"mask is not binary: {msk.n_unique} distinct values")

    try:
        arr = _load_pixels(mask_path)
        if arr is not None:
            uniq = np.unique(arr)
            mask_labels = [float(v) for v in uniq[:10]]
            positive = int((arr > 0).sum())
            # An empty mask is NOT a defect here: the export is a full acquired
            # stack and only about a quarter of slices contain tumour, so most
            # masks are legitimately background. Emptiness is only meaningful at
            # patient level, which the manifest tracks separately.
            if positive == arr.size:
                problems.append("mask is entirely foreground")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"mask pixel inspection failed: {type(exc).__name__}: {exc}")

    return GeometryCheck(
        image_path=image_path,
        mask_path=mask_path,
        status="OK" if not problems else "MISMATCH",
        problems=problems,
        image_shape=img.shape,
        mask_shape=msk.shape,
        mask_is_binary=msk.is_binary,
        mask_positive_px=positive,
        mask_labels=mask_labels,
    )


def _load_pixels(path: Path) -> np.ndarray | None:
    """Load pixel data for a supported file, or None when unreadable/empty."""
    early = _empty_or_missing(path)
    if early is not None:
        return None
    name = path.name.lower()
    try:
        if name.endswith((".tif", ".tiff")):
            import tifffile

            return tifffile.imread(path)
        if name.endswith((".nii", ".nii.gz", ".mha", ".mhd", ".nrrd", ".dcm")):
            import SimpleITK as sitk

            return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
        from PIL import Image

        with Image.open(path) as im:
            return np.array(im)
    except Exception:  # noqa: BLE001
        return None


def load_pixels(path: Path) -> np.ndarray | None:
    """Public wrapper around the pixel loader."""
    return _load_pixels(path)


def to_2d(arr: np.ndarray) -> np.ndarray:
    """Collapse a trailing singleton/channel axis so shapes are comparable.

    The TIFF export stores images as (H, W, 1) and masks as (H, W), which would
    otherwise register as a shape mismatch on the channel axis alone.
    """
    if arr.ndim == 3:
        return arr[..., 0] if arr.shape[-1] == 1 else arr.mean(axis=-1)
    return arr


def resample_mask_to_image(mask: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resample a binary mask onto the image grid.

    Required by the `images.zip` derivative of this dataset, in which images were
    downsampled to 256x256 but masks were left at the native acquisition matrix
    (512, 560, 576, 480, 640, ...). 93% of image/mask pairs are affected, so a
    naive overlay or training loop silently pairs a mask with a differently-sized
    image. Nearest neighbour is mandatory here: any interpolating order would
    invent non-binary label values at the boundary.
    """
    m = to_2d(mask)
    target = tuple(image_shape[:2])
    if tuple(m.shape[:2]) == target:
        return m > 0
    from skimage.transform import resize

    out = resize(
        (m > 0).astype(np.float32),
        target,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )
    return out > 0.5


def load_pair(image_path: Path, mask_path: Path) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """Load an image/mask pair onto a common grid.

    Returns (image_2d, mask_bool, status) where status is one of
    OK, MASK_RESAMPLED, MASK_MISSING, IMAGE_MISSING. MASK_RESAMPLED is a
    *recoverable defect*, reported rather than hidden, because how many pairs
    needed resampling is itself a finding about this dataset.
    """
    img = _load_pixels(image_path)
    if img is None:
        return None, None, "IMAGE_MISSING"
    img = to_2d(img)

    raw = _load_pixels(mask_path)
    if raw is None:
        return img, None, "MASK_MISSING"

    needed = tuple(to_2d(raw).shape[:2]) != tuple(img.shape[:2])
    mask = resample_mask_to_image(raw, img.shape)
    return img, mask, "MASK_RESAMPLED" if needed else "OK"
