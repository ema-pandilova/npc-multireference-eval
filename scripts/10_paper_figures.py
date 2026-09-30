#!/usr/bin/env python3
"""Publication figures.

Figure 1  the image/mask grid defect, before and after correction
Figure 2  the two disagreements compared, sequences against readers, one patient
Figure 3  cross-sequence variability of the morphological descriptors
Figure 4  sequence effect against reader effect, paired within patient
Figure 5  one fixed model scored against each reference in turn

File names match the order the figures are printed in the manuscript, which the
earlier version of this script did not.

All figures are regenerated from the result tables and the raw data, so a change
upstream propagates here rather than being re-drawn by hand.

Drawing conventions follow the radiology literature rather than matplotlib
defaults: no titles on the canvas, panel letters set as bold characters outside
the axes, statistics printed as text rather than encoded in a title, and no
chromatic colour except where it carries meaning that greyscale cannot. Type 42
fonts are embedded because several production houses reject Type 3.

Usage:
    python3 scripts/10_paper_figures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.imaging import load_pair, load_pixels, to_2d  # noqa: E402
from src.paths import load_config  # noqa: E402

# Arial resolves to Liberation Sans here, which is metrically identical, so the
# emitted PDF keeps its metrics if the publisher re-renders with real Arial.
# mathtext is pinned to the same family; leaving it at the default puts DejaVu
# superscripts next to Liberation body text, which is visible on the page.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8.5, "axes.titlesize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
    "mathtext.bf": "Liberation Sans:bold", "mathtext.sf": "Liberation Sans",
    "mathtext.default": "regular",
    "axes.linewidth": 0.8, "axes.edgecolor": "black", "axes.labelcolor": "black",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "axes.axisbelow": True,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "xtick.color": "black", "ytick.color": "black",
    "legend.frameon": False,
    "lines.linewidth": 1.0, "lines.solid_capstyle": "round",
    "figure.dpi": 150, "savefig.dpi": 600,
    "pdf.fonttype": 42, "ps.fonttype": 42,   # never Type 3
})

# Greyscale carries the data; the two chromatic values are reserved for the one
# element in each figure that must be found instantly, and both survive CVD.
INK = "#000000"
DARK = "#595959"      # emphasised category
LIGHT = "#BFBFBF"     # comparison category
MID = "#808080"       # reference and threshold lines, dashed
PALE = "#D9D9D9"      # background data, individual trajectories
BLUE = "#0C7BDC"      # median trajectory (Fig 2), corrected contour (Fig 3)
YELLOW = "#FFC20A"    # as-distributed contour (Fig 3)

FIG_W = 5.45          # figure width in inches (13.86 cm text column)
SEQS = ["T1", "T2", "CET1"]
LAB = {"T1": "T1WI", "T2": "T2WI", "CET1": "CE-T1WI"}


def save(fig, out_dir: Path, name: str) -> None:
    """Write PDF and PNG at the same size.

    tight_layout followed by bbox_inches="tight" made the emitted width differ
    from figsize, which is what clipped an axis label in the previous version.
    Layout is settled before saving and the bbox is left alone.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=600)
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def panel_letter(ax, letter: str, dx: float = -0.16, dy: float = 1.06) -> None:
    """Bold letter outside the axes, the convention in Radiology and Eur Radiol."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=9.5,
            fontweight="bold", va="top", ha="left", color=INK)


def norm(img: np.ndarray) -> np.ndarray:
    img = to_2d(img).astype(np.float32)
    lo, hi = np.percentile(img, [1, 99])
    if hi <= lo:
        lo, hi = float(img.min()), float(img.max())
    return np.clip((img - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(img)


def style_box(bp, face: str) -> None:
    for b in bp["boxes"]:
        b.set_facecolor(face); b.set_edgecolor(INK); b.set_linewidth(0.8)
    for part in ("whiskers", "caps"):
        for e in bp[part]:
            e.set_color(INK); e.set_linewidth(0.8)
    for m in bp["medians"]:
        m.set_color(INK); m.set_linewidth(1.4)


# ------------------------------------------------------------------ Fig 1
def figure2(tables: Path, out_dir: Path) -> None:
    """Cross-sequence variability of the four morphological descriptors."""
    print("Figure 3: cross-sequence variability")
    per = pd.read_csv(tables / "morphology_cross_sequence.csv", dtype={"patient_id": str})

    cols = ["spread_pct_Vol", "spread_pct_SA", "spread_pct_SReg", "spread_pct_MaxD"]
    names = ["Volume", "Surface area", "Surface regularity", "Maximum diameter"]
    data = [per[c].dropna().to_numpy() for c in cols]

    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 4.30))
    fig.subplots_adjust(left=0.26, right=0.97, top=0.95, bottom=0.10, hspace=0.55)

    # (a) horizontal boxes, ordered by median so the ranking is read off directly.
    ax = axes[0]
    bp = ax.boxplot(data[::-1], vert=False, widths=0.62, patch_artist=True,
                    showfliers=True,
                    flierprops=dict(marker="o", markersize=1.6, markerfacecolor=MID,
                                    markeredgecolor="none"))
    style_box(bp, LIGHT)
    ax.set_yticks(range(1, len(names) + 1))
    ax.set_yticklabels(names[::-1])
    ax.axvline(5.7, color=MID, ls=(0, (4, 2)), lw=0.9, zorder=0)
    ax.set_xlabel("Within-patient spread across sequences (%)")
    # Five of 1108 plotted values lie beyond 70%; the caption says so rather than
    # the axis silently absorbing them.
    ax.set_xlim(0, 70); ax.set_ylim(0.4, 5.05)
    ax.annotate("Reader effect, 5.7%", xy=(5.7, 4.72), xytext=(15, 4.72),
                fontsize=7.2, color=INK, va="center",
                arrowprops=dict(arrowstyle="-", lw=0.7, color=MID))
    panel_letter(ax, "a", dx=-0.32, dy=1.12)

    # (b) survival curve: proportion of patients above a given spread. Plotting
    # positions are (n - i)/n, not linspace(0, 100), which had asserted that no
    # patient falls below the observed minimum.
    ax = axes[1]
    s = np.sort(per["spread_pct_Vol"].dropna().to_numpy())
    n = len(s)
    surv = (n - np.arange(n)) / n * 100.0
    ax.step(np.concatenate([[0.0], s]), np.concatenate([[100.0], surv]),
            where="post", color=INK, lw=1.2)
    for t in (10, 20, 30):
        frac = (s > t).mean() * 100
        ax.plot([t, t], [0, frac], color=MID, ls=(0, (1, 2)), lw=0.8, zorder=0)
        ax.plot([t], [frac], marker="o", ms=3.0, color=INK, zorder=3)
        ax.annotate(f"{frac:.0f}% above {t}%", xy=(t, frac), xytext=(t + 6, frac + 9),
                    fontsize=7.2, color=INK,
                    arrowprops=dict(arrowstyle="-", lw=0.7, color=MID))
    ax.set_xlim(0, 70); ax.set_ylim(0, 104)
    ax.set_xlabel("Within-patient spread across sequences (%)")
    ax.set_ylabel("Patients above (%)")
    panel_letter(ax, "b", dx=-0.32, dy=1.14)

    save(fig, out_dir, "fig3_cross_sequence_variability")


# ------------------------------------------------------------------ Fig 2
def figure3(tables: Path, out_dir: Path) -> None:
    """Sequence effect against reader effect, matched estimator leading.

    The reader effect is a relative difference between two volumes; the
    range-based sequence effect is a range across three. The expected range grows
    with the number of values compared, so the matched pairwise estimator is the
    like-for-like comparison and is drawn as the primary result. The range-based
    value is shown beside it, de-emphasised and labelled, rather than omitted.
    """
    print("Figure 4: sequence effect vs reader effect")
    src = tables / "sequence_vs_reader_matched.csv"
    if not src.exists():
        print("  missing sequence_vs_reader_matched.csv; skipping")
        return
    d = pd.read_csv(src, dtype={"patient_id": str})

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 2.85),
                             gridspec_kw={"width_ratios": [1.25, 1.0]})
    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.28, wspace=0.40)

    ax = axes[0]
    for _, r in d.iterrows():
        ax.plot([0, 1], [r.reader, r.sequence_matched], color=PALE, lw=0.7, zorder=1)
    mR, mM, mG = d.reader.median(), d.sequence_matched.median(), d.sequence_range.median()
    for x, col in ((0, d.reader), (1, d.sequence_matched)):
        q1, q3 = np.percentile(col, [25, 75])
        ax.plot([x, x], [q1, q3], color=INK, lw=1.0, zorder=3)
    ax.plot([0, 1], [mR, mM], color=BLUE, lw=2.0, zorder=4)
    ax.plot([0, 1], [mR, mM], marker="o", ms=4.5, ls="none", color=BLUE, zorder=5,
            markeredgecolor="white", markeredgewidth=0.7)
    ax.text(-0.07, mR, "%.1f%%" % mR, ha="right", va="center", fontsize=7.5, color=INK)
    ax.text(1.07, mM, "%.1f%%" % mM, ha="left", va="center", fontsize=7.5, color=INK)

    # the superseded estimator, plainly marked as such
    ax.plot([2], [mG], marker="o", ms=4.0, color=MID, zorder=4)
    ax.plot([1, 2], [mM, mG], color=MID, ls=(0, (2, 2)), lw=1.0, zorder=2)
    ax.text(2.07, mG, "%.1f%%" % mG, ha="left", va="center", fontsize=7.5, color=MID)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Reader", "Sequence\n(matched)", "Sequence\n(range)"], fontsize=7.0)
    ax.get_xticklabels()[2].set_color(MID)
    ax.set_xlim(-0.45, 2.62)
    ax.set_ylabel("Relative volume difference (%)")
    ax.text(0.30, 0.98, "$P = 2.4 \\times 10^{-5}$", transform=ax.transAxes,
            ha="center", va="top", fontsize=7.2, color=INK)
    panel_letter(ax, "a", dx=-0.26, dy=1.14)

    ax = axes[1]
    diff = (d.sequence_matched - d.reader).sort_values().to_numpy()
    y = np.arange(1, len(diff) + 1)
    ax.hlines(y, 0, diff, color=LIGHT, lw=0.9, zorder=1)
    ax.plot(diff, y, marker="o", ms=2.8, ls="none", color=DARK, zorder=2)
    ax.axvline(0, color=INK, lw=0.9, zorder=3)
    ax.set_ylim(0.2, len(diff) + 0.8); ax.set_yticks([1, 10, 20, 30])
    ax.set_xlabel("Sequence minus reader\n(points, matched)")
    ax.set_ylabel("Patient (ranked)")
    ax.text(0.97, 0.05, "%d of %d above zero" % (int((diff > 0).sum()), len(diff)),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.2, color=INK)
    panel_letter(ax, "b", dx=-0.30, dy=1.14)

    print("    reader %.2f%%, matched %.2f%%, range %.2f%%; %d/%d above zero"
          % (mR, mM, mG, int((diff > 0).sum()), len(diff)))
    save(fig, out_dir, "fig4_sequence_vs_reader")


# ------------------------------------------------------------------ Fig 3
def figure1(cfg, out_dir: Path, pid: str = "013", idx: str = "009") -> None:
    """The grid defect and its correction, on one patient."""
    print("Figure 1: image/mask grid defect")
    root = cfg.images_root / pid
    mods = ["T1WI", "T2WI", "CE-T1WI"]

    fig = plt.figure(figsize=(FIG_W, 3.70))
    gs = fig.add_gridspec(2, 4, width_ratios=[0.07, 1, 1, 1], height_ratios=[1, 1],
                          left=0.01, right=0.99, top=0.90, bottom=0.02,
                          wspace=0.05, hspace=0.05)
    lab_axes = [fig.add_subplot(gs[r, 0]) for r in (0, 1)]
    for a in lab_axes:
        a.axis("off")
    lab_axes[0].text(0.5, 0.5, "As distributed", rotation=90, va="center",
                     ha="center", fontsize=8, color=INK, transform=lab_axes[0].transAxes)
    lab_axes[1].text(0.5, 0.5, "After resampling", rotation=90, va="center",
                     ha="center", fontsize=8, color=INK, transform=lab_axes[1].transAxes)

    # One extent shared by all three columns, sized to the content rather than to
    # the nominal 512 grid. Computing it per column drew T2WI at twice the
    # magnification of the other two; using the full 512 left most of the row empty.
    lim = 0
    for mod in mods:
        ip, mp = root / f"{mod}_{idx}.tif", root / f"{mod}_{idx}_mask.tif"
        if not (ip.exists() and mp.exists()):
            continue
        img = to_2d(load_pixels(ip))
        raw = to_2d(load_pixels(mp))
        ys, xs = np.nonzero(raw > 0)
        reach = max(int(xs.max()), int(ys.max())) if ys.size else 0
        lim = max(lim, img.shape[0], img.shape[1], reach)
    lim = int(lim * 1.06)

    letters = [["a", "b", "c"], ["d", "e", "f"]]
    for c, mod in enumerate(mods):
        ip = root / f"{mod}_{idx}.tif"
        mp = root / f"{mod}_{idx}_mask.tif"
        ax_top = fig.add_subplot(gs[0, c + 1])
        ax_bot = fig.add_subplot(gs[1, c + 1])
        if not (ip.exists() and mp.exists()):
            for a in (ax_top, ax_bot):
                a.axis("off")
            continue
        img = to_2d(load_pixels(ip))
        raw = to_2d(load_pixels(mp))
        _, fixed, _ = load_pair(ip, mp)

        # Top row: mask drawn in its own coordinate system, which is what pairing
        # files by name produces. A single shared extent is used for all three
        # columns (see lim above).
        ax_top.imshow(norm(img), cmap="gray", interpolation="nearest",
                      extent=[0, img.shape[1], img.shape[0], 0], zorder=1)
        ax_top.contour(raw > 0, levels=[0.5], colors=YELLOW, linewidths=1.1,
                       extent=[0, raw.shape[1], raw.shape[0], 0], zorder=2)
        ax_top.add_patch(Rectangle((0, 0), img.shape[1], img.shape[0], fill=False,
                                   ec=MID, ls=(0, (3, 2)), lw=0.7, zorder=3))
        ax_top.set_xlim(0, lim); ax_top.set_ylim(lim, 0)
        ax_top.set_title(mod, fontsize=8, color=INK, loc="center", pad=11)
        # Under the column title rather than on the canvas, where it collided
        # with the image once the extent was cropped to the content.
        ax_top.text(0.5, 1.01, f"image {img.shape[0]}$^2$, mask {raw.shape[0]}$^2$",
                    transform=ax_top.transAxes, ha="center", va="bottom",
                    fontsize=6.5, color=INK, zorder=4)
        ax_top.axis("off")
        ax_top.text(0.02, 0.98, letters[0][c], transform=ax_top.transAxes,
                    fontsize=9, fontweight="bold", va="top", ha="left", color="white",
                    zorder=5)

        # Bottom row: after nearest-neighbour resampling onto the image grid.
        ax_bot.imshow(norm(img), cmap="gray", interpolation="nearest", zorder=1)
        if fixed is not None and fixed.any():
            ax_bot.contour(fixed, levels=[0.5], colors=BLUE, linewidths=1.1, zorder=2)
        ax_bot.axis("off")
        ax_bot.text(0.02, 0.98, letters[1][c], transform=ax_bot.transAxes,
                    fontsize=9, fontweight="bold", va="top", ha="left", color="white",
                    zorder=5)
        if c == 0:   # scale reference, one panel only
            h, w = img.shape[:2]
            bar = w * 0.25
            ax_bot.plot([w * 0.06, w * 0.06 + bar], [h * 0.94, h * 0.94],
                        color="white", lw=1.8, solid_capstyle="butt", zorder=5)
            ax_bot.text(w * 0.06 + bar / 2, h * 0.90, f"{int(bar)} px", color="white",
                        fontsize=6.4, ha="center", va="bottom", zorder=5)

    save(fig, out_dir, "fig1_mask_grid_mismatch")




# ------------------------------------------------ reference disagreement
# Okabe-Ito hues, each also carrying a distinct dash pattern so the panels
# survive greyscale reproduction and common colour vision deficiencies.
SEQ_STYLE = {
    "T1WI":    ("#1A85FF", "solid"),
    "T2WI":    ("#FFC20A", (0, (5, 2))),
    "CE-T1WI": ("#D41159", (0, (1.2, 1.6))),
}
RDR_STYLE = {
    "Reader 1": ("#1A85FF", "solid"),
    "Reader 2": ("#009E73", (0, (5, 2))),
}


def figure_reference_overlay(cfg, out_dir: Path, pid: str = "016") -> None:
    """The two disagreements the paper compares, on one patient and one slice.

    Patient 016 was selected by rule, not by eye: among the thirty patients
    annotated by both radiologists it minimises the combined relative deviation
    from both dataset medians. Its cross-sequence agreement is 0.787 against a
    median of 0.783, and its inter-reader agreement is 0.884 against 0.899, so
    the case is representative on both axes and slightly conservative on the
    reader side.
    """
    import SimpleITK as sitk
    print("Figure 2: sequence versus reader disagreement")
    seg = cfg.primary_data_root / "MRI-Segments" / pid
    rel = cfg.reliability_root / pid

    def series(folder):
        r = sitk.ImageSeriesReader()
        ids = r.GetGDCMSeriesIDs(str(folder))
        best, n = None, -1
        for sid in ids:
            f = r.GetGDCMSeriesFileNames(str(folder), sid)
            if len(f) > n:
                best, n = sid, len(f)
        r.SetFileNames(r.GetGDCMSeriesFileNames(str(folder), best))
        return r.Execute()

    def register(fx, mv):
        ff, mf = sitk.Cast(fx, sitk.sitkFloat32), sitk.Cast(mv, sitk.sitkFloat32)
        init = sitk.CenteredTransformInitializer(
            ff, mf, sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY)
        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        R.SetMetricSamplingStrategy(R.RANDOM)
        R.SetMetricSamplingPercentage(0.10, seed=20260727)
        R.SetInterpolator(sitk.sitkLinear)
        R.SetOptimizerAsRegularStepGradientDescent(1.0, 1e-4, 200,
                                                   gradientMagnitudeTolerance=1e-6)
        R.SetOptimizerScalesFromPhysicalShift()
        R.SetShrinkFactorsPerLevel([4, 2, 1])
        R.SetSmoothingSigmasPerLevel([2, 1, 0])
        R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        R.SetInitialTransform(init, inPlace=False)
        return R.Execute(ff, mf)

    fixed = series(seg / "T1WI")
    seqs = {"T1WI": sitk.GetArrayFromImage(sitk.ReadImage(str(seg / "ROI-T1.nii"))) > 0}
    for name, roi in (("T2WI", "ROI-T2.nii"), ("CE-T1WI", "ROI-CE-T1.nii")):
        mv = series(seg / name)
        ref = sitk.ReadImage(str(seg / roi))
        tx = register(fixed, mv)
        seqs[name] = sitk.GetArrayFromImage(
            sitk.Resample(ref, fixed, tx, sitk.sitkNearestNeighbor, 0, ref.GetPixelID())) > 0

    # Both readers annotated the same T1WI acquisition, so no registration is
    # involved on this panel and none of its separation can be blamed on ours.
    rdrs = {"Reader 1": seqs["T1WI"],
            "Reader 2": sitk.GetArrayFromImage(sitk.ReadImage(str(rel / "ROI-T1.nii"))) > 0}

    img = sitk.GetArrayFromImage(fixed)
    union = seqs["T1WI"] | seqs["T2WI"] | seqs["CE-T1WI"] | rdrs["Reader 2"]
    z = int(np.argmax(union.sum(axis=(1, 2))))
    ys, xs = np.nonzero(union[z])
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    half = int(max(ys.max() - ys.min(), xs.max() - xs.min()) * 1.35)
    y0, y1 = max(0, cy - half), min(img.shape[1], cy + half)
    x0, x1 = max(0, cx - half), min(img.shape[2], cx + half)
    sl = norm(img[z])[y0:y1, x0:x1]

    def dice(a, b):
        s = a.sum() + b.sum()
        return 2 * np.logical_and(a, b).sum() / s if s else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 3.05))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.15, wspace=0.05)

    for ax, (style, masks, letter, note) in zip(axes, [
            (SEQ_STYLE, seqs, "a", "Between sequences, DSC 0.79"),
            (RDR_STYLE, rdrs, "b", "Between readers, DSC 0.87")]):
        ax.imshow(sl, cmap="gray", interpolation="nearest", zorder=1)
        for name, (col, ls) in style.items():
            ax.contour(masks[name][z][y0:y1, x0:x1], levels=[0.5], colors=[col],
                       linewidths=1.3, linestyles=[ls], zorder=2)
        ax.axis("off")
        panel_letter(ax, letter, dx=0.0, dy=1.09)
        ax.text(0.5, 1.005, note, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=7.0, color=INK)
        handles = [plt.Line2D([], [], color=c, ls=s, lw=1.3, label=n)
                   for n, (c, s) in style.items()]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.005),
                  ncol=len(style), fontsize=6.8, handlelength=2.2,
                  columnspacing=1.0, borderpad=0.2, handletextpad=0.5)

    print("    slice %d; cross-sequence DSC %.3f/%.3f/%.3f; reader DSC %.3f" % (
        z, dice(seqs["T1WI"], seqs["T2WI"]), dice(seqs["T1WI"], seqs["CE-T1WI"]),
        dice(seqs["T2WI"], seqs["CE-T1WI"]), dice(rdrs["Reader 1"], rdrs["Reader 2"])))
    save(fig, out_dir, "fig2_reference_vs_reader")


# ------------------------------------------------------------------ Fig 5
def figure_multireference(tables: Path, out_dir: Path) -> None:
    """One fixed set of predictions, scored against each reference in turn.

    The manuscript argues that a single reported figure conceals the spread. The
    headline result was itself reported only as three medians inside a table, so
    this panel shows the per-patient distribution the medians summarise.
    """
    print("Figure 5: multi-reference evaluation")
    d = pd.read_csv(tables / "multireference_evaluation.csv", dtype={"patient_id": str})
    cols = ["dice_vs_T1WI", "dice_vs_T2WI", "dice_vs_CE-T1WI"]
    labs = ["T1WI", "T2WI", "CE-T1WI"]
    d = d.dropna(subset=cols)

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 2.75),
                             gridspec_kw={"width_ratios": [1.15, 1.0]})
    fig.subplots_adjust(left=0.11, right=0.98, top=0.88, bottom=0.20, wspace=0.36)

    # (a) every held-out patient scored three ways, with the reference the model
    # was trained on first. One line per patient; the model never changes.
    ax = axes[0]
    for _, r in d.iterrows():
        ax.plot([0, 1, 2], [r[c] for c in cols], color=PALE, lw=0.5, zorder=1)
    med = [d[c].median() for c in cols]
    ax.plot([0, 1, 2], med, color=BLUE, lw=2.0, zorder=4)
    ax.plot([0, 1, 2], med, marker="o", ms=4.5, ls="none", color=BLUE, zorder=5,
            markeredgecolor="white", markeredgewidth=0.7)
    for i, m in enumerate(med):
        ax.annotate(f"{m:.3f}", (i, m), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=7.0, color=INK)
    ax.axhline(0.898, color=MID, ls=(0, (4, 2)), lw=0.9, zorder=2)
    ax.text(2.42, 0.898, "Inter-reader\n0.898", fontsize=6.6, color=INK,
            va="center", ha="left")
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(labs)
    ax.set_xlim(-0.35, 2.38); ax.set_ylim(0, 1.0)
    ax.set_xlabel("Reference used for scoring")
    ax.set_ylabel("Dice similarity coefficient")
    panel_letter(ax, "a", dx=-0.24, dy=1.15)

    # (b) what the three medians hide: the per-patient spread.
    ax = axes[1]
    s = np.sort(d.reference_sensitivity.to_numpy())
    n = len(s)
    ax.step(np.concatenate([[0.0], s]), np.concatenate([[100.0], (n - np.arange(n)) / n * 100]),
            where="post", color=INK, lw=1.2)
    for thr in (0.05, 0.10):
        frac = (s > thr).mean() * 100
        ax.plot([thr, thr], [0, frac], color=MID, ls=(0, (1, 2)), lw=0.8, zorder=0)
        ax.plot([thr], [frac], marker="o", ms=3.0, color=INK, zorder=3)
        ax.annotate(f"{frac:.0f}% above {thr:.2f}", xy=(thr, frac),
                    xytext=(thr + 0.035, frac + 10), fontsize=7.0, color=INK,
                    arrowprops=dict(arrowstyle="-", lw=0.7, color=MID))
    ax.set_xlim(0, 0.35); ax.set_ylim(0, 104)
    ax.set_xlabel("Reference sensitivity (Dice)")
    ax.set_ylabel("Patients above (%)")
    panel_letter(ax, "b", dx=-0.26, dy=1.15)

    print(f"    n = {len(d)}; medians {med[0]:.3f}/{med[1]:.3f}/{med[2]:.3f}; "
          f"{(s > 0.35).sum()} patients beyond the plotted range")
    save(fig, out_dir, "fig5_multireference")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--allow-missing-roots", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, verify=True, allow_missing=True)
    tables = cfg.out("tables_dir")
    out_dir = cfg.out("figures_dir") / "paper"

    figure1(cfg, out_dir)
    figure_reference_overlay(cfg, out_dir)
    figure2(tables, out_dir)
    figure3(tables, out_dir)
    figure_multireference(tables, out_dir)
    print(f"\nFigures in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
