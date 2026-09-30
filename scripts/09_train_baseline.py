#!/usr/bin/env python3
"""Train a per-sequence 2D segmentation baseline with an uncertainty estimate.

The model is deliberately conventional. The paper's claims are about measurement
validity, not architecture, so the baseline exists to answer two questions that
need a model at all:

    H4  does a model's uncertainty land where the two readers actually disagree?
    H5  can a referral threshold be anchored to measured human disagreement
        rather than to an arbitrary Dice value?

Design notes that matter for correctness:

  * Splits are patient-level and read from data/splits/patient_splits.csv, which
    was produced and leakage-checked by scripts/02_make_splits.py. All 30
    two-reader patients sit in the test partition, so reader disagreement is only
    ever measured on held-out cases.
  * Masks are resampled onto the image grid with nearest neighbour. Without this
    93% of pairs are mismatched (see data_audit.md) and the model would train
    against masks that do not correspond to their images.
  * One model per sequence. The sequences are not co-registered (T1WI and T2WI
    sit on different in-plane grids in 27 of 30 patients), so stacking them as
    channels would be wrong, and keeping them separate is also what makes the
    per-sequence comparison meaningful.
  * Uncertainty comes from a deep ensemble of N members with different seeds,
    optionally combined with test-time augmentation. Both are standard; neither
    is claimed as a contribution.

Usage:
    python3 scripts/09_train_baseline.py --sequence T1WI --members 5 --epochs 60
    python3 scripts/09_train_baseline.py --sequence all --members 5 --epochs 60
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.imaging import load_pair  # noqa: E402
from src.paths import load_config  # noqa: E402


# --------------------------------------------------------------- data
class SliceDataset:
    """Slices for one sequence and one partition, cached in memory as float32.

    The whole cohort at 256x256 is a few GB at most, so caching avoids hammering
    the network filesystem for every epoch.
    """

    def __init__(self, cfg, split_ids: set[str], sequence: str, size: int = 256,
                 require_lesion: bool = False):
        self.items: list[tuple[np.ndarray, np.ndarray]] = []
        self.size = size
        root = cfg.images_root
        n_resampled = 0
        for pid in sorted(split_ids):
            pdir = root / pid
            if not pdir.is_dir():
                continue
            for img_path in sorted(pdir.glob(f"{sequence}_[0-9][0-9][0-9].tif")):
                mask_path = img_path.with_name(f"{img_path.stem}_mask{img_path.suffix}")
                if not mask_path.exists():
                    continue
                img, mask, status = load_pair(img_path, mask_path)
                if img is None or mask is None:
                    continue
                if status == "MASK_RESAMPLED":
                    n_resampled += 1
                if require_lesion and not mask.any():
                    continue
                self.items.append((self._prep(img), mask.astype(np.float32)))
        self.n_resampled = n_resampled

    def _prep(self, img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32)
        body = img[img > 0]
        if body.size < 10:
            return np.zeros_like(img)
        mu, sd = float(body.mean()), float(body.std())
        return (img - mu) / (sd if sd > 1e-6 else 1.0)

    def __len__(self) -> int:
        return len(self.items)

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        x = np.stack([a for a, _ in self.items])[:, None]
        y = np.stack([b for _, b in self.items])[:, None]
        return x, y


# --------------------------------------------------------------- model
def build_unet(torch, nn, ch: int = 32):
    class Block(nn.Module):
        def __init__(self, i, o):
            super().__init__()
            self.f = nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1, bias=False), nn.InstanceNorm2d(o, affine=True),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Conv2d(o, o, 3, padding=1, bias=False), nn.InstanceNorm2d(o, affine=True),
                nn.LeakyReLU(0.01, inplace=True))

        def forward(self, x):
            return self.f(x)

    class UNet(nn.Module):
        """A plain 4-level U-Net. nnU-Net's own 2D configuration for anisotropic
        data reduces to essentially this; we keep it explicit and small so the
        ensemble is affordable."""

        def __init__(self):
            super().__init__()
            self.e1, self.e2, self.e3, self.e4 = Block(1, ch), Block(ch, ch * 2), Block(ch * 2, ch * 4), Block(ch * 4, ch * 8)
            self.b = Block(ch * 8, ch * 16)
            self.u4 = nn.ConvTranspose2d(ch * 16, ch * 8, 2, 2); self.d4 = Block(ch * 16, ch * 8)
            self.u3 = nn.ConvTranspose2d(ch * 8, ch * 4, 2, 2); self.d3 = Block(ch * 8, ch * 4)
            self.u2 = nn.ConvTranspose2d(ch * 4, ch * 2, 2, 2); self.d2 = Block(ch * 4, ch * 2)
            self.u1 = nn.ConvTranspose2d(ch * 2, ch, 2, 2); self.d1 = Block(ch * 2, ch)
            self.head = nn.Conv2d(ch, 1, 1)
            self.pool = nn.MaxPool2d(2)

        def forward(self, x):
            e1 = self.e1(x); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2)); e4 = self.e4(self.pool(e3))
            b = self.b(self.pool(e4))
            d4 = self.d4(torch.cat([self.u4(b), e4], 1))
            d3 = self.d3(torch.cat([self.u3(d4), e3], 1))
            d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
            d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
            return self.head(d1)

    return UNet()


def dice_loss(torch, logits, target, eps: float = 1.0):
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return 1 - (num / den).mean()


def augment(torch, x, y):
    """Flips and 90-degree rotations only. Head-and-neck MRI is not
    left-right symmetric in general, but the tumour is midline and the
    augmentation is standard for this task; intensity jitter is added
    separately because MRI has no absolute scale."""
    if torch.rand(1).item() < 0.5:
        x, y = torch.flip(x, [-1]), torch.flip(y, [-1])
    if torch.rand(1).item() < 0.3:
        k = int(torch.randint(1, 4, (1,)).item())
        x, y = torch.rot90(x, k, [-2, -1]), torch.rot90(y, k, [-2, -1])
    if torch.rand(1).item() < 0.3:
        x = x * (0.9 + 0.2 * torch.rand(1, device=x.device)) + 0.1 * torch.randn(1, device=x.device)
    return x, y


# --------------------------------------------------------------- train
def make_batches(rng, n_fg: int, n_bg: int, batch: int, fg_frac: float,
                 fg_idx: np.ndarray, bg_idx: np.ndarray, steps: int):
    """Batches with a guaranteed foreground share.

    Only about a quarter of slices contain tumour and lesion pixels are ~0.16% of
    the image, so uniform sampling drives the model straight to an all-background
    solution (measured: Dice 0.000). nnU-Net solves this by forcing a foreground
    fraction into every batch; the same fix is applied here.
    """
    n_fg_b = max(1, int(round(batch * fg_frac)))
    n_bg_b = batch - n_fg_b
    for _ in range(steps):
        a = rng.choice(fg_idx, size=min(n_fg_b, n_fg), replace=n_fg < n_fg_b)
        b = rng.choice(bg_idx, size=min(n_bg_b, n_bg), replace=n_bg < n_bg_b) if n_bg else np.array([], int)
        yield np.concatenate([a, b])


def train_member(torch, nn, xtr, ytr, xva, yva, seed: int, epochs: int,
                 batch: int, dev: str, lr: float = 3e-4, fg_frac: float = 0.34,
                 eval_every: int = 2) -> tuple:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_unet(torch, nn).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bce = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=dev.startswith("cuda"))

    has_fg = ytr.reshape(len(ytr), -1).sum(1) > 0
    fg_idx, bg_idx = np.flatnonzero(has_fg), np.flatnonzero(~has_fg)
    steps = max(1, len(xtr) // batch)
    best, best_state = -1.0, None

    for ep in range(epochs):
        model.train()
        tot, seen = 0.0, 0
        for idx in make_batches(rng, len(fg_idx), len(bg_idx), batch, fg_frac, fg_idx, bg_idx, steps):
            xb = torch.from_numpy(xtr[idx]).to(dev, non_blocking=True)
            yb = torch.from_numpy(ytr[idx]).to(dev, non_blocking=True)
            xb, yb = augment(torch, xb, yb)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=dev.startswith("cuda")):
                out = model(xb)
                # Dice-weighted: BCE alone is dominated by the 99.8% background.
                loss = 0.3 * bce(out, yb) + 0.7 * dice_loss(torch, out, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
            scaler.step(opt)
            scaler.update()
            tot += float(loss) * len(idx); seen += len(idx)
        sched.step()

        if (ep + 1) % eval_every == 0 or ep == epochs - 1:
            d = evaluate_dice(torch, model, xva, yva, dev, batch)
            if d > best:
                best, best_state = d, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"      seed {seed} epoch {ep + 1:3d}/{epochs}  loss {tot / max(1, seen):.4f}  "
                  f"val slice-Dice {d:.4f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best


def evaluate_dice(torch, model, x, y, dev, batch) -> float:
    """Mean Dice over slices that contain lesion in the reference."""
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(x[i:i + batch]).to(dev)
            yb = torch.from_numpy(y[i:i + batch]).to(dev)
            p = (torch.sigmoid(model(xb)) > 0.5).float()
            inter = (p * yb).sum(dim=(1, 2, 3))
            den = p.sum(dim=(1, 2, 3)) + yb.sum(dim=(1, 2, 3))
            has = yb.sum(dim=(1, 2, 3)) > 0
            if has.any():
                scores.append(((2 * inter[has] + 1e-6) / (den[has] + 1e-6)).cpu().numpy())
    return float(np.concatenate(scores).mean()) if scores else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--sequence", default="T1WI", help="T1WI | T2WI | CE-T1WI | all")
    ap.add_argument("--members", type=int, default=5, help="ensemble members")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4,
                    help="1e-3 was unstable: val Dice oscillated between 0.21 and 0.00")
    ap.add_argument("--fg-frac", type=float, default=0.34,
                    help="foreground share per batch; nnU-Net uses ~0.33")
    ap.add_argument("--eval-every", type=int, default=2)
    args = ap.parse_args()

    import torch
    import torch.nn as nn

    cfg = load_config(args.config, verify=True, allow_missing=True)
    dev = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    splits = pd.read_csv(cfg.out("splits_dir") / "patient_splits.csv", dtype={"patient_id": str})
    part = {k: set(v) for k, v in splits.groupby("split")["patient_id"].apply(set).items()}
    print(f"device {dev} | train {len(part.get('train', []))} "
          f"calib {len(part.get('calib', []))} test {len(part.get('test', []))} patients")

    seqs = ["T1WI", "T2WI", "CE-T1WI"] if args.sequence == "all" else [args.sequence]
    out_dir = cfg.repo_root / "models"
    out_dir.mkdir(exist_ok=True)
    summary = {}

    for seq in seqs:
        print(f"\n{'=' * 70}\n{seq}\n{'=' * 70}")
        t0 = time.time()
        tr = SliceDataset(cfg, part["train"], seq)
        va = SliceDataset(cfg, part["calib"], seq)
        print(f"  train slices {len(tr)} ({tr.n_resampled} masks resampled to image grid)")
        print(f"  calib slices {len(va)}")
        if not len(tr) or not len(va):
            print("  empty split; skipping")
            continue
        xtr, ytr = tr.arrays()
        xva, yva = va.arrays()
        print(f"  tensors {xtr.shape} / {xva.shape}  lesion fraction {ytr.mean():.4f}")

        members, scores = [], []
        for k in range(args.members):
            print(f"    -- member {k + 1}/{args.members}")
            model, best = train_member(torch, nn, xtr, ytr, xva, yva,
                                       seed=cfg.seed + k, epochs=args.epochs,
                                       batch=args.batch, dev=dev,
                                       lr=args.lr, fg_frac=args.fg_frac,
                                       eval_every=args.eval_every)
            path = out_dir / f"{seq}_member{k}.pt"
            torch.save(model.state_dict(), path)
            members.append(path.name)
            scores.append(best)
            print(f"       saved {path.name}  best val slice-Dice {best:.4f}")

        summary[seq] = {"members": members, "val_dice": scores,
                        "val_dice_mean": float(np.mean(scores)),
                        "train_slices": len(tr), "calib_slices": len(va),
                        "minutes": round((time.time() - t0) / 60, 1)}
        print(f"  {seq} done in {summary[seq]['minutes']} min, "
              f"mean val Dice {summary[seq]['val_dice_mean']:.4f}")

    with open(out_dir / "training_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nWrote {out_dir}/training_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
