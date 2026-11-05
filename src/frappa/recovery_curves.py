"""Recovery curves of the bleached droplets.

  A = spot(t) / ref(t)        B = A(t) / A(pre)
  C = tot(t)  / ref(t)        D = C(t) / C(pre)
  final = B(t) / D(t)

spot = bleached region, tot = whole bleached droplet, ref = reference droplets (not moved, not
bleached; all their pixels pooled). All are mean intensities per frame. No background subtraction.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator


def measure(stack, labels, spots, tab, frame_pre):
    """Intensities and normalization of every bleached droplet of one file.
    tab: table of all droplets from bleach_detection (with 'moved' and 'bleached').
    Returns a long table: one row per bleached droplet and frame."""

    ref_ids = tab.loc[~tab["moved"] & ~tab["bleached"], "droplet"]
    ref = stack[:, np.isin(labels, ref_ids)].mean(axis=1)       # one value per frame
    frames = np.arange(stack.shape[0])

    rows = []
    for d in tab.loc[tab["bleached"], "droplet"]:
        spot = stack[:, spots == d].mean(axis=1)          # bleached region
        tot = stack[:, labels == d].mean(axis=1)          # whole bleached droplet
        A = spot / ref
        B = A / A[frame_pre]
        C = tot / ref
        D = C / C[frame_pre]
        rows.append(pd.DataFrame({"droplet": d, "frame": frames, "spot": spot, "tot": tot, "ref": ref,
                                  "A": A, "B": B, "C": C, "D": D, "final": B / D}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def curves_figure(df, dashed=frozenset()):
    """Recovery curves of one file: each droplet in colour (ID in the legend), mean ± SD in black."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for d, dd in df.groupby("droplet"):
        ls = "--" if d in dashed else "-"
        ax.plot(dd["frame"], dd["final"], ls=ls, lw=0.8, label=d)
    mean = df.groupby("frame")["final"].mean()
    sd = df.groupby("frame")["final"].std()
    ax.plot(mean.index, mean, "k-o", markersize=3)
    ax.fill_between(mean.index, mean - sd, mean + sd, color="k", alpha=0.1)
    ax.axhline(1, color="gray", ls=":", lw=0.8)
    note = "(dashed: spot above the saturation threshold before the bleach)" if dashed else ""
    ax.set_title(f"{df['file'].iloc[0]} (n = {df['droplet'].nunique()}) {note}", fontsize=10)
    ax.set_xlabel("frame")
    ax.set_ylabel("final (normalized intensity)")
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.legend(title="droplet", fontsize=7, title_fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def all_files_curves_figure(curves):
    """Recovery curves of all files, one panel per file: droplets in light gray, mean ± SD in
    black. Panels share the y-axis, so the files can be compared."""
    names = curves["file"].unique()
    fig, axes = plt.subplots(1, len(names), squeeze=False, sharey=True, figsize=(5 * len(names), 4))
    for ax, name in zip(axes[0], names):
        df = curves[curves["file"] == name]
        for _, dd in df.groupby("droplet"):
            ax.plot(dd["frame"], dd["final"], color="0.8", lw=0.6)
        mean = df.groupby("frame")["final"].mean()
        sd = df.groupby("frame")["final"].std()
        ax.plot(mean.index, mean, "k-o", markersize=3)
        ax.fill_between(mean.index, mean - sd, mean + sd, color="k", alpha=0.1)
        ax.axhline(1, color="gray", ls=":", lw=0.8)
        ax.set_title(f"{name} (n = {df['droplet'].nunique()})", fontsize=10)
        ax.set_xlabel("frame")
        ax.xaxis.set_major_locator(MultipleLocator(5))
    axes[0, 0].set_ylabel("final (normalized intensity)")
    fig.tight_layout()
    return fig


def steps_figure(df, frame_pre, dashed=frozenset()):
    """Normalization steps of one file. Panel 1: whole bleached droplets, tot(t)/tot(pre), and the
    reference, ref(t)/ref(pre) (thick black); then B, D and final normalization (with the mean ± SD).
    Each droplet has the same colour in every panel."""
    ref = df.groupby("frame")["ref"].first()                 # the same for every droplet of the file
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for d, dd in df.groupby("droplet"):
        ls = "--" if d in dashed else "-"
        axes[0].plot(dd["frame"], dd["tot"] / dd["tot"].iloc[frame_pre], ls=ls, lw=0.8)
        axes[1].plot(dd["frame"], dd["B"], ls=ls, lw=0.8)
        axes[2].plot(dd["frame"], dd["D"], ls=ls, lw=0.8)
        axes[3].plot(dd["frame"], dd["final"], ls=ls, lw=0.8, label=d)
    axes[0].plot(ref.index, ref / ref[frame_pre], "k-", lw=2, label="reference")
    mean = df.groupby("frame")["final"].mean()
    sd = df.groupby("frame")["final"].std()
    axes[3].plot(mean.index, mean, "k-", lw=2, label="mean")
    axes[3].fill_between(mean.index, mean - sd, mean + sd, color="k", alpha=0.15)

    titles = ["whole droplets tot(t)/tot(pre), reference ref(t)/ref(pre) in black", 
              "B = A / A(pre)", "D = C / C(pre)", "final = B / D"]
    for ax, title in zip(axes, titles):
        ax.axhline(1, color="gray", ls=":", lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("frame")
        ax.xaxis.set_major_locator(MultipleLocator(5))
    axes[0].set_ylabel("normalized intensity")
    axes[3].legend(title="droplet", fontsize=6, title_fontsize=7, ncol=2)
    note = "(dashed: spot above the saturation threshold before the bleach)" if dashed else ""
    fig.suptitle(f"{df['file'].iloc[0]}: normalization steps {note}", fontsize=11)
    fig.tight_layout()
    return fig