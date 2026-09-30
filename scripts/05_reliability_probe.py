#!/usr/bin/env python3
"""Probe the reliability subset: real 3D geometry, volumes, and cross-sequence spread.

Runs on the 3D NIfTI masks shipped in `inner_reliability.zip`, which carry real
voxel spacing and therefore permit tumour volume in cm³ — unlike the 2D TIFF
export. Three questions are answered:

  1. GEOMETRY. What are the true in-plane and through-plane spacings, and are the
     three sequences on a common grid? (They are not.)

  2. SLICE SELECTION. Does the 2D TIFF export contain the full acquired stack or
     only tumour-bearing slices? This determines whether every slice-level Dice
     published on this dataset is inflated by lesion-only selection, so it is
     tested rather than assumed: TIFF slice count is compared against both the
     full NIfTI stack depth and the count of lesion-bearing slices.

  3. CROSS-SEQUENCE VOLUME. How much does the same patient's tumour volume change
     depending on which sequence it was delineated on? Tested with Friedman
     across the three sequences and pairwise Wilcoxon.

If a second rater's masks are also available (primary_data.zip supplies
MRI-Segments/<id>/ROI-*.nii), `--rater2-root` adds the inter-rater comparison and
decomposes sequence effect against rater effect. Without it the script reports
the single-pass result and says so.

Usage:
    python3 scripts/05_reliability_probe.py --nifti-root ../dataset_v2/inner_reliability
    python3 scripts/05_reliability_probe.py --nifti-root ../dataset_v2/inner_reliability \
        --rater2-root ../dataset_v2/primary_data/data/MRI-Segments
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import load_config  # noqa: E402

SEQ_MAP = {"ROI-T1": "T1WI", "ROI-T2": "T2WI", "ROI-CE-T1": "CE-T1WI"}


def load_mask(path: Path):
    """Return (binary array, zooms) or (None, None)."""
    try:
        import nibabel as nib

        im = nib.load(str(path))
        arr = np.asanyarray(im.dataobj)
        return (arr > 0), tuple(float(z) for z in im.header.get_zooms()[:3])
    except Exception as exc:  # noqa: BLE001
        print(f"    WARN could not read {path.name}: {type(exc).__name__}: {exc}")
        return None, None


def dice(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("nan")
    inter = float(np.logical_and(a, b).sum())
    tot = float(a.sum() + b.sum())
    return 2 * inter / tot if tot else float("nan")


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("nan")
    union = float(np.logical_or(a, b).sum())
    return float(np.logical_and(a, b).sum()) / union if union else float("nan")


def scan(root: Path, images_root: Path | None) -> pd.DataFrame:
    rows = []
    for pdir in sorted(p for p in root.iterdir() if p.is_dir()):
        pid = pdir.name
        for stem, tok in SEQ_MAP.items():
            path = pdir / f"{stem}.nii"
            if not path.exists():
                path = pdir / f"{stem}.nii.gz"
            if not path.exists():
                continue
            arr, zooms = load_mask(path)
            if arr is None:
                continue
            vox_mm3 = float(np.prod(zooms))
            lesion_per_slice = arr.sum(axis=(0, 1)) if arr.ndim == 3 else np.array([arr.sum()])
            n_tif = -1
            if images_root is not None:
                d = images_root / pid
                n_tif = len(list(d.glob(f"{tok}_[0-9][0-9][0-9].tif"))) if d.is_dir() else 0
            rows.append(
                {
                    "patient_id": pid,
                    "sequence": tok,
                    "shape": "x".join(str(s) for s in arr.shape),
                    "spacing_x_mm": round(zooms[0], 4),
                    "spacing_z_mm": round(zooms[2], 3) if len(zooms) > 2 else np.nan,
                    "nii_slices": int(arr.shape[2]) if arr.ndim == 3 else 1,
                    "nii_lesion_slices": int((lesion_per_slice > 0).sum()),
                    "tif_slices": n_tif,
                    "voxels": int(arr.sum()),
                    "volume_cm3": round(float(arr.sum()) * vox_mm3 / 1000.0, 3),
                }
            )
    return pd.DataFrame(rows)


def _paired_effect_comparison(inter: pd.DataFrame, root: Path, r2: Path, cfg) -> None:
    """The study's central test: is the sequence effect larger than the reader effect?

    Both effects are measured on the SAME patients, so they are compared with a
    paired test rather than by eyeballing two marginal medians:

      reader effect   = |V_reader1 - V_reader2| / mean, averaged over sequences
      sequence effect = within-reader spread across the three sequences,
                        averaged over the two readers

    Averaging over the nuisance axis in each case keeps one value per patient per
    effect, which is what makes the pairing valid.
    """
    per = {}
    for pid, grp in inter.groupby("patient_id"):
        if len(grp) < 3:
            continue
        reader = grp["vol_diff_pct"].mean()
        seq = []
        for col in ("vol_rater1_cm3", "vol_rater2_cm3"):
            v = grp[col].to_numpy(dtype=float)
            if v.size >= 3 and np.mean(v) > 0:
                seq.append((v.max() - v.min()) / np.mean(v) * 100)
        if seq:
            per[pid] = (reader, float(np.mean(seq)))

    if len(per) < 5:
        print("\n  Too few complete patients for the paired effect comparison.")
        return

    d = pd.DataFrame(
        [{"patient_id": k, "reader_effect_pct": a, "sequence_effect_pct": b}
         for k, (a, b) in sorted(per.items())]
    )
    re_, se = d.reader_effect_pct, d.sequence_effect_pct
    print(f"\n  CENTRAL TEST — sequence effect vs reader effect (n={len(d)} patients)")
    print(f"    reader effect   (same sequence, 2 readers): median {re_.median():.1f}%  "
          f"IQR {re_.quantile(.25):.1f}-{re_.quantile(.75):.1f}")
    print(f"    sequence effect (same reader, 3 sequences): median {se.median():.1f}%  "
          f"IQR {se.quantile(.25):.1f}-{se.quantile(.75):.1f}")
    try:
        from scipy import stats

        w = stats.wilcoxon(se, re_)
        print(f"    paired Wilcoxon: W={w.statistic:.1f}, p={w.pvalue:.3g}")
    except Exception as exc:  # noqa: BLE001
        print(f"    (test unavailable: {exc})")

    ratio = se.median() / re_.median() if re_.median() else float("nan")
    rng = np.random.default_rng(cfg.seed)
    boot = []
    for _ in range(10000):
        idx = rng.integers(0, len(d), len(d))
        s = d.iloc[idx]
        if s.reader_effect_pct.median():
            boot.append(s.sequence_effect_pct.median() / s.reader_effect_pct.median())
    lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan)
    n_bigger = int((se > re_).sum())
    print(f"    ratio of medians: {ratio:.1f}x  (bootstrap 95% CI {lo:.2f}-{hi:.2f}, "
          f"{len(boot)} resamples, seed {cfg.seed})")
    print(f"    sequence effect exceeds reader effect in {n_bigger}/{len(d)} patients "
          f"({100 * n_bigger / len(d):.0f}%)")

    d.round(3).to_csv(cfg.out("tables_dir") / "sequence_vs_reader_effect.csv", index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--allow-missing-roots", action="store_true",
                    help="downgrade absent dataset roots to a warning (partially-staged dataset)")
    ap.add_argument("--nifti-root", required=True, help="extracted inner_reliability directory")
    ap.add_argument("--rater2-root", default=None, help="MRI-Segments dir holding the other rater")
    args = ap.parse_args()

    cfg = load_config(args.config, verify=True, allow_missing=args.allow_missing_roots)
    repo = cfg.repo_root
    root = (repo / args.nifti_root).resolve() if not Path(args.nifti_root).is_absolute() else Path(args.nifti_root)
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory", file=sys.stderr)
        return 2

    print("=" * 78)
    print("RELIABILITY SUBSET PROBE (3D NIfTI)")
    print("=" * 78)
    print(f"Root: {root}\n")

    df = scan(root, cfg.images_root)
    if df.empty:
        print("ERROR: no readable NIfTI masks found.", file=sys.stderr)
        return 3
    print(f"Read {len(df)} masks for {df.patient_id.nunique()} patients\n")

    # ---- 1. geometry ---------------------------------------------------
    print("1. GEOMETRY")
    g = df.groupby("sequence").agg(
        n=("patient_id", "size"),
        inplane_median=("spacing_x_mm", "median"),
        inplane_min=("spacing_x_mm", "min"),
        inplane_max=("spacing_x_mm", "max"),
        slice_median=("spacing_z_mm", "median"),
    )
    print(g.round(3).to_string())
    grids = df.pivot_table(index="patient_id", columns="sequence", values="spacing_x_mm")
    if {"T1WI", "T2WI"}.issubset(grids.columns):
        differ = int((~np.isclose(grids["T1WI"], grids["T2WI"], rtol=1e-3)).sum())
        print(f"\n  patients where T1WI and T2WI sit on DIFFERENT in-plane grids: "
              f"{differ}/{len(grids)}")
        print("  -> the sequences are not co-registered; slice-index stacking is invalid")

    # ---- 2. slice selection -------------------------------------------
    print("\n2. SLICE SELECTION (is the TIFF export the full stack or lesion-only?)")
    cmp_df = df[df.tif_slices > 0]
    if cmp_df.empty:
        print("  no TIFF counterparts available — cannot test")
    else:
        full = int((cmp_df.tif_slices == cmp_df.nii_slices).sum())
        lesion = int((cmp_df.tif_slices == cmp_df.nii_lesion_slices).sum())
        print(f"  comparable masks              : {len(cmp_df)}")
        print(f"  TIFF count == full NIfTI stack: {full}/{len(cmp_df)}")
        print(f"  TIFF count == lesion slices   : {lesion}/{len(cmp_df)}")
        frac = cmp_df.nii_lesion_slices / cmp_df.nii_slices
        print(f"  lesion-bearing fraction of stack: median {frac.median():.2f} "
              f"(so lesion-only export would be far smaller)")
        verdict = ("FULL STACK — no lesion-only selection bias" if full > lesion
                   else "LESION-ONLY — slice-level Dice on this export is inflated")
        print(f"  VERDICT: {verdict}")

    # ---- 3. cross-sequence volume -------------------------------------
    print("\n3. CROSS-SEQUENCE TUMOUR VOLUME (same patient, same annotation pass)")
    piv = df.pivot_table(index="patient_id", columns="sequence", values="volume_cm3")
    seqs = [s for s in ("T1WI", "T2WI", "CE-T1WI") if s in piv.columns]
    piv = piv.dropna(subset=seqs)
    if len(piv) >= 3 and len(seqs) == 3:
        mean = piv[seqs].mean(axis=1)
        spread = (piv[seqs].max(axis=1) - piv[seqs].min(axis=1)) / mean * 100
        print(f"  patients: {len(piv)}")
        print("  median volume per sequence (cm3): "
              + ", ".join(f"{s} {piv[s].median():.2f}" for s in seqs))
        print(f"  within-patient spread across sequences: median {spread.median():.1f}%, "
              f"IQR {spread.quantile(.25):.1f}-{spread.quantile(.75):.1f}%, max {spread.max():.1f}%")
        print(f"  patients with >20% spread: {int((spread > 20).sum())}/{len(spread)}")
        print(f"  patients with >30% spread: {int((spread > 30).sum())}/{len(spread)}")
        try:
            from scipy import stats

            fr = stats.friedmanchisquare(*[piv[s] for s in seqs])
            print(f"  Friedman across sequences: chi2={fr.statistic:.3f}, p={fr.pvalue:.4g}")
            for a, b in (("CE-T1WI", "T1WI"), ("CE-T1WI", "T2WI"), ("T1WI", "T2WI")):
                if a in piv and b in piv:
                    w = stats.wilcoxon(piv[a], piv[b])
                    print(f"    Wilcoxon {a} vs {b}: p={w.pvalue:.4g}")
        except Exception as exc:  # noqa: BLE001
            print(f"  (statistics unavailable: {exc})")
        piv["spread_pct"] = spread.round(2)
    else:
        print("  insufficient data for the cross-sequence comparison")

    # ---- 4. inter-rater (optional) ------------------------------------
    inter = pd.DataFrame()
    if args.rater2_root:
        r2 = (repo / args.rater2_root).resolve() if not Path(args.rater2_root).is_absolute() else Path(args.rater2_root)
        print(f"\n4. INTER-RATER AGREEMENT vs {r2}")
        if not r2.is_dir():
            print(f"  ERROR: {r2} not found — skipping")
        else:
            rows = []
            for pid in sorted(df.patient_id.unique()):
                for stem, tok in SEQ_MAP.items():
                    p1 = r2 / pid / f"{stem}.nii"
                    p2 = root / pid / f"{stem}.nii"
                    if not (p1.exists() and p2.exists()):
                        continue
                    a1, z1 = load_mask(p1)
                    a2, z2 = load_mask(p2)
                    if a1 is None or a2 is None:
                        continue
                    v1 = float(a1.sum()) * float(np.prod(z1)) / 1000.0
                    v2 = float(a2.sum()) * float(np.prod(z2)) / 1000.0
                    rows.append(
                        {
                            "patient_id": pid, "sequence": tok,
                            "dice": round(dice(a1, a2), 4),
                            "jaccard": round(jaccard(a1, a2), 4),
                            "vol_rater1_cm3": round(v1, 3),
                            "vol_rater2_cm3": round(v2, 3),
                            "vol_diff_pct": round(abs(v1 - v2) / ((v1 + v2) / 2) * 100, 2) if (v1 + v2) else np.nan,
                        }
                    )
            inter = pd.DataFrame(rows)
            if inter.empty:
                print("  no overlapping masks found — the two roots may hold the same rater")
            else:
                print(f"  paired masks: {len(inter)}")
                st = inter.groupby("sequence").agg(
                    n=("dice", "size"), dice_median=("dice", "median"),
                    dice_q1=("dice", lambda s: s.quantile(.25)),
                    dice_q3=("dice", lambda s: s.quantile(.75)),
                    jaccard_median=("jaccard", "median"),
                    vol_diff_median_pct=("vol_diff_pct", "median"),
                )
                print(st.round(4).to_string())
                _paired_effect_comparison(inter, root, r2, cfg)
    else:
        print("\n4. INTER-RATER AGREEMENT — skipped (no --rater2-root)")
        print("  The reliability archive holds ONE mask set per sequence per patient.")
        print("  The second reader's masks live in primary_data.zip under")
        print("  MRI-Segments/<id>/ROI-*.nii, so inter-rater agreement for this")
        print("  dataset can only be computed by pairing the two archives.")

    # ---- outputs -------------------------------------------------------
    tables = cfg.out("tables_dir")
    df.to_csv(tables / "reliability_geometry.csv", index=False)
    if len(piv):
        piv.round(3).to_csv(tables / "reliability_cross_sequence_volume.csv")
    if not inter.empty:
        inter.to_csv(tables / "inter_rater_agreement.csv", index=False)
    print(f"\nWrote tables to {tables}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
