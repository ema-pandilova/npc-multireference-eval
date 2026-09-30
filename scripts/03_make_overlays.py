#!/usr/bin/env python3
"""Render image/mask overlays to verify that masks align with their images.

Overlays are the only way to catch a mask that passes every numeric geometry
check yet is semantically wrong (shifted, flipped, or copied from the wrong
sequence). When pixel data is unavailable a diagnostic panel is written instead,
so a missing overlay is never mistaken for a correct empty one.

Usage:
    python3 scripts/03_make_overlays.py
    python3 scripts/03_make_overlays.py --n 12 --slices 4
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.discovery import discover_patients, discover_reliability  # noqa: E402
from src.overlays import render_patient_overlay, render_rater_comparison  # noqa: E402
from src.paths import load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--allow-missing-roots", action="store_true",
                    help="downgrade absent dataset roots to a warning (partially-staged dataset)")
    ap.add_argument("--n", type=int, default=8, help="number of patients to render")
    ap.add_argument("--slices", type=int, default=3, help="slices per modality")
    ap.add_argument("--patients", default=None, help="comma-separated patient IDs to force")
    args = ap.parse_args()

    cfg = load_config(args.config, verify=True, allow_missing=args.allow_missing_roots)
    patients = discover_patients(cfg, cfg.images_root)
    reliability = discover_reliability(cfg)
    fig_dir = cfg.out("figures_dir") / "overlays"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Prefer patients that actually have bytes; fall back to any with files so
    # the diagnostic panels still document the problem.
    with_pixels = [p for p, r in patients.items() if r.has_pixel_data]
    with_files = [p for p, r in patients.items() if not r.is_empty_dir]
    pool = with_pixels or with_files

    if args.patients:
        chosen = [p.strip() for p in args.patients.split(",") if p.strip() in patients]
    else:
        rng = random.Random(cfg.seed)
        chosen = sorted(rng.sample(pool, min(args.n, len(pool)))) if pool else []

    print("=" * 78)
    print("OVERLAY GENERATION")
    print("=" * 78)
    print(f"Patients with pixel data : {len(with_pixels)}")
    print(f"Patients with files      : {len(with_files)}")
    print(f"Rendering                : {len(chosen)} -> {fig_dir}")
    print()

    rows = []
    for pid in chosen:
        out = fig_dir / f"patient_{pid}_overlay.png"
        ok, msg = render_patient_overlay(cfg, patients[pid], out, n_slices=args.slices)
        print(f"  {pid}  {'OK  ' if ok else 'FAIL'}  {msg}")
        rows.append({"patient_id": pid, "figure": out.name, "rendered": ok, "message": msg})

    # Multi-rater figures for the reliability subset: the inter-rater envelope
    # the study calibrates against.
    rel_rendered = 0
    for pid in sorted(set(reliability) & set(patients))[: args.n]:
        if reliability[pid].n_files == 0:
            continue
        out = fig_dir / f"patient_{pid}_raters.png"
        ok, msg = render_rater_comparison(cfg, patients[pid], reliability[pid], out)
        rows.append({"patient_id": pid, "figure": out.name, "rendered": ok, "message": msg})
        rel_rendered += int(ok)
    print(f"\nMulti-rater figures rendered: {rel_rendered}")
    if rel_rendered == 0:
        print("  (the reliability directories contain no annotation files)")

    log = cfg.out("figures_dir") / "overlay_log.csv"
    pd.DataFrame(rows).to_csv(log, index=False)
    n_ok = sum(r["rendered"] for r in rows)
    print(f"\n{n_ok}/{len(rows)} figures contain real image data.")
    print(f"Log: {log}")
    if n_ok == 0:
        print("\nNo overlay could be rendered: the dataset has no pixel data. "
              "Diagnostic panels were written instead.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
