#!/usr/bin/env python3
"""Preprocessing and model-loading smoke test on a handful of patients.

Exercises the full path from disk to a batched tensor without launching any
expensive training:

    1. resolve paths and discover patients
    2. load image+mask slices for 2-3 patients
    3. verify geometry, normalise, resize to the model input size
    4. assemble a batch tensor and report its statistics
    5. instantiate a small U-Net and run one forward pass (CPU or GPU)

Every stage reports PASS / FAIL / SKIP so a partial environment is diagnosed
precisely rather than failing with a stack trace.

Usage:
    python3 scripts/04_preprocess_smoke_test.py
    python3 scripts/04_preprocess_smoke_test.py --n 3 --size 256
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.discovery import discover_patients  # noqa: E402
from src.imaging import check_pair_geometry, load_pixels  # noqa: E402
from src.paths import load_config  # noqa: E402

RESULTS: list[tuple[str, str, str]] = []


def report(stage: str, status: str, detail: str = "") -> None:
    RESULTS.append((stage, status, detail))
    print(f"  [{status:4s}] {stage}" + (f" — {detail}" if detail else ""))


def normalise_slice(img: np.ndarray) -> np.ndarray:
    """Z-score within the non-zero body region; MRI has no absolute intensity scale."""
    img = img.astype(np.float32)
    if img.ndim == 3:
        img = img.mean(axis=-1)
    body = img[img > 0]
    if body.size < 10:
        return np.zeros_like(img)
    mu, sd = float(body.mean()), float(body.std())
    return (img - mu) / (sd if sd > 1e-6 else 1.0)


def resize_to(arr: np.ndarray, size: int, is_mask: bool = False) -> np.ndarray:
    from skimage.transform import resize

    return resize(
        arr,
        (size, size),
        order=0 if is_mask else 1,
        preserve_range=True,
        anti_aliasing=not is_mask,
    ).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--allow-missing-roots", action="store_true",
                    help="downgrade absent dataset roots to a warning (partially-staged dataset)")
    ap.add_argument("--n", type=int, default=3, help="patients to process")
    ap.add_argument("--size", type=int, default=256, help="model input size")
    args = ap.parse_args()

    print("=" * 78)
    print("PREPROCESSING SMOKE TEST")
    print("=" * 78)

    # ---------------------------------------------------------- stage 1
    print("\n1. Configuration and discovery")
    try:
        cfg = load_config(args.config, verify=True, allow_missing=args.allow_missing_roots)
        report("load config + verify dataset roots", "PASS", f"seed={cfg.seed}")
    except Exception as exc:  # noqa: BLE001
        report("load config", "FAIL", str(exc))
        return summarize()

    patients = discover_patients(cfg, cfg.images_root)
    report("discover patients", "PASS", f"{len(patients)} directories")

    usable = [p for p, r in patients.items() if r.has_pixel_data]
    with_files = [p for p, r in patients.items() if not r.is_empty_dir]
    if usable:
        report("patients with pixel data", "PASS", f"{len(usable)}")
    else:
        report(
            "patients with pixel data", "FAIL",
            f"0 of {len(with_files)} non-empty directories contain bytes",
        )

    chosen = sorted(usable)[: args.n]
    if not chosen:
        report("select patients for preprocessing", "SKIP", "no patient has pixel data")
        print("\n2.-5.  Skipped: nothing to preprocess.")
        run_model_check(args.size, data_available=False)
        return summarize()

    # ---------------------------------------------------------- stage 2
    print(f"\n2. Loading slices for {len(chosen)} patients: {chosen}")
    batch_imgs: list[np.ndarray] = []
    batch_masks: list[np.ndarray] = []
    for pid in chosen:
        rec = patients[pid]
        mod = next((m for m in cfg.modalities if m in rec.modalities_present), None)
        if mod is None:
            report(f"patient {pid}: modality", "FAIL", "no known modality present")
            continue
        imgs = [s for s in rec.images(mod) if not s.is_empty]
        if not imgs:
            report(f"patient {pid}: slices", "FAIL", "no readable slices")
            continue
        s = imgs[len(imgs) // 2]
        mp = s.path.with_name(f"{s.path.stem}_mask{s.path.suffix}")

        chk = check_pair_geometry(s.path, mp)
        report(
            f"patient {pid}: geometry {mod} #{s.index}",
            "PASS" if chk.ok else "FAIL",
            "; ".join(chk.problems) or f"shape {chk.image_shape}, {chk.mask_positive_px} lesion px",
        )

        img, msk = load_pixels(s.path), load_pixels(mp)
        if img is None:
            report(f"patient {pid}: load pixels", "FAIL", "image unreadable")
            continue
        norm = normalise_slice(img)
        img_r = resize_to(norm, args.size)
        msk_r = resize_to((msk > 0).astype(np.float32), args.size, is_mask=True) if msk is not None else np.zeros_like(img_r)
        batch_imgs.append(img_r)
        batch_masks.append(msk_r)
        report(
            f"patient {pid}: preprocess",
            "PASS",
            f"{img.shape} -> {img_r.shape}, mean {img_r.mean():+.3f}, sd {img_r.std():.3f}, "
            f"lesion {int(msk_r.sum())} px",
        )

    # ---------------------------------------------------------- stage 3
    print("\n3. Batch assembly")
    if batch_imgs:
        x = np.stack(batch_imgs)[:, None]
        y = np.stack(batch_masks)[:, None]
        report("stack batch", "PASS", f"images {x.shape} {x.dtype}, masks {y.shape}")
        report(
            "batch statistics", "PASS",
            f"x range [{x.min():.2f}, {x.max():.2f}], positive mask fraction {y.mean():.4f}",
        )
    else:
        report("stack batch", "FAIL", "no slices survived preprocessing")
        x = None

    # ---------------------------------------------------------- stage 4-5
    run_model_check(args.size, data_available=x is not None, batch=x)
    return summarize()


def run_model_check(size: int, data_available: bool, batch: np.ndarray | None = None) -> None:
    print("\n4. Model instantiation and forward pass")
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        report("import torch", "FAIL", str(exc))
        return

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    report("torch available", "PASS", f"{torch.__version__}, device={dev}"
           + (f", {torch.cuda.device_count()} GPU(s)" if dev == "cuda" else ""))

    class TinyUNet(nn.Module):
        """Minimal encoder/decoder used only to prove the tensor path works.

        The study's actual baseline is nnU-Net v2; this is a shape check, not a
        model proposal.
        """

        def __init__(self, ch: int = 16):
            super().__init__()
            self.enc1 = self._block(1, ch)
            self.enc2 = self._block(ch, ch * 2)
            self.bottleneck = self._block(ch * 2, ch * 4)
            self.up2 = nn.ConvTranspose2d(ch * 4, ch * 2, 2, stride=2)
            self.dec2 = self._block(ch * 4, ch * 2)
            self.up1 = nn.ConvTranspose2d(ch * 2, ch, 2, stride=2)
            self.dec1 = self._block(ch * 2, ch)
            self.head = nn.Conv2d(ch, 1, 1)
            self.pool = nn.MaxPool2d(2)

        @staticmethod
        def _block(i: int, o: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            )

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            b = self.bottleneck(self.pool(e2))
            d2 = self.dec2(torch.cat([self.up2(b), e2], 1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
            return self.head(d1)

    try:
        model = TinyUNet().to(dev).eval()
        n_params = sum(p.numel() for p in model.parameters())
        report("instantiate model", "PASS", f"{n_params:,} parameters")
    except Exception as exc:  # noqa: BLE001
        report("instantiate model", "FAIL", str(exc))
        return

    # Use the real batch when it exists, otherwise a synthetic one so the tensor
    # path is still validated end to end.
    if data_available and batch is not None:
        t = torch.from_numpy(batch).float().to(dev)
        source = "real data"
    else:
        t = torch.randn(2, 1, size, size, device=dev)
        source = "synthetic tensor (no pixel data on disk)"

    try:
        with torch.no_grad():
            out = model(t)
        report("forward pass", "PASS", f"{tuple(t.shape)} -> {tuple(out.shape)} using {source}")
    except Exception as exc:  # noqa: BLE001
        report("forward pass", "FAIL", str(exc))
        return

    # Test-time augmentation is the uncertainty mechanism the study will use,
    # so its plumbing is smoke-tested here too.
    try:
        with torch.no_grad():
            preds = [torch.sigmoid(model(t))]
            preds.append(torch.flip(torch.sigmoid(model(torch.flip(t, [-1]))), [-1]))
            preds.append(torch.flip(torch.sigmoid(model(torch.flip(t, [-2]))), [-2]))
            stack = torch.stack(preds)
        report(
            "TTA uncertainty plumbing", "PASS",
            f"{stack.shape[0]} augmentations, mean sd {stack.std(0).mean().item():.5f}",
        )
    except Exception as exc:  # noqa: BLE001
        report("TTA uncertainty plumbing", "FAIL", str(exc))


def summarize() -> int:
    print("\n" + "=" * 78)
    n_pass = sum(1 for _, s, _ in RESULTS if s == "PASS")
    n_fail = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    n_skip = sum(1 for _, s, _ in RESULTS if s == "SKIP")
    print(f"SMOKE TEST: {n_pass} passed, {n_fail} failed, {n_skip} skipped")
    if n_fail:
        print("\nFailures:")
        for stage, status, detail in RESULTS:
            if status == "FAIL":
                print(f"  - {stage}: {detail}")
        print("\nThe code path is exercised as far as the data allows. Failures above that "
              "mention missing pixel data are dataset problems, not code problems.")
    print("=" * 78)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
