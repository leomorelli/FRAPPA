"""Saturation report (optional, with --saturation-report): how many pixels of each droplet and
spot are saturated, i.e. at the maximum value of the data type (255 for 8-bit images).

Considers for the bleached droplets, the whole droplet and its spot; for the references, the whole droplet.
For each region (whole droplet or spot), reports the fraction of saturated pixels in every frame,
the fraction in the pre-bleach frame, and the worst post-bleach fraction with its frame.

Bleached droplets whose spot or whole droplet has a saturated fraction above the threshold before
the bleach are flagged. Nothing is excluded: the report only shows where saturation may affect the results.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MultipleLocator
RED = ListedColormap(["red"])

from .common import crop_around


def saturation_per_frame(saturated, label_img, ids):
    """Fraction of saturated pixels in each region (whole droplet or spot), frame by frame.
    saturated: True/False stack (frames, rows, columns); label_img: droplet or spot labels.
    Returns a table with one row per ID and one column per frame."""
    fractions = {d: saturated[:, label_img == d].mean(axis=1) for d in ids}
    table = pd.DataFrame.from_dict(fractions, orient="index", columns=range(saturated.shape[0]))
    table.columns.name = "frame"
    return table


def report(stack, sat_value, labels, spots, tab, frame_pre, threshold):
    """Saturation report of one file, for the droplets that did not move.
    Returns: a table with one row per droplet (role, saturated fractions before and after the bleach,
    flags above the threshold), the per-frame tables {"droplet": ..., "spot": ...},
    and the counts for the summary."""

    static = tab.loc[~tab["moved"]]
    sat = pd.DataFrame({"droplet": static["droplet"], "role": np.where(static["bleached"], "bleached", "reference")})
    is_bleached = sat["role"] == "bleached"
    bleached_ids = sat.loc[is_bleached, "droplet"]
    saturated = stack >= sat_value
    per_frame = {"droplet": saturation_per_frame(saturated, labels, sat["droplet"]),
                 "spot": saturation_per_frame(saturated, spots, bleached_ids)}

    counts = {"bleached": int(is_bleached.sum())}
    for region, table in per_frame.items():
        post = table.loc[:, frame_pre + 1:]                    # post-bleach frames only
        worst_post = post.max(axis=1)
        sat[f"sat_{region}_pre"] = sat["droplet"].map(table[frame_pre])
        sat[f"sat_{region}_post_worst"] = sat["droplet"].map(worst_post)
        sat[f"sat_{region}_post_worst_frame"] = sat["droplet"].map(post.idxmax(axis=1).where(worst_post > 0))
        sat[f"{region}_pre_above"] = is_bleached & (sat[f"sat_{region}_pre"] > threshold)
        counts[f"{region}_above_before"] = int(sat[f"{region}_pre_above"].sum())
        counts[f"{region}_above_after"] = int((post.loc[bleached_ids] > threshold).any(axis=1).sum())
    
    refs = sat.loc[~is_bleached]
    counts["references"] = len(refs)
    counts["references_above_before"] = int((refs["sat_droplet_pre"] > threshold).sum())
    counts["references_above_after"] = int((refs["sat_droplet_post_worst"] > threshold).sum())
    counts["references_any_saturated"] = int(
        (refs[["sat_droplet_pre", "sat_droplet_post_worst"]].max(axis=1) > 0).sum())

    return sat.reset_index(drop=True), per_frame, counts


def summary_text(counts, threshold):
    """Lines summarizing the saturation report of one file, for the terminal.
    'after' means in at least one post-bleach frame; 'any saturated pixel' means in any frame."""
    n, th = counts["bleached"], f"{threshold:.0%}"
    return (f"    saturation (pixels at the maximum value):\n"
            f"      spots more than {th} saturated: {counts['spot_above_before']} before the bleach, "
            f"{counts['spot_above_after']} after\n"
            f"      whole droplets more than {th} saturated: {counts['droplet_above_before']} before "
            f"the bleach, {counts['droplet_above_after']} after\n"
            f"      references with any saturated pixel: {counts['references_any_saturated']} "
            f"of {counts['references']}\n"
            f"      references more than {th} saturated: {counts['references_above_before']} before "
            f"the bleach, {counts['references_above_after']} after")
            

def over_time_figure(sat_tables, per_frame_tables, threshold):
    """Saturated fraction of each droplet over time, one panel per file (coloured: bleached
    droplets, legend = droplet ID; gray: references; droplets never saturated are not drawn).
    sat_tables: file name -> per-droplet table from report()
    per_frame_tables: file name -> per-frame table of the whole droplets."""
    fig, axes = plt.subplots(1, len(sat_tables), squeeze=False, figsize=(5 * len(sat_tables), 4))    
    for ax, (name, sf) in zip(axes[0], per_frame_tables.items()):
        roles = sat_tables[name].set_index("droplet")["role"]
        for d, row in sf.iterrows():
            if row.max() == 0:                        # never saturated: not drawn
                continue
            if roles[d] == "bleached":
                ax.plot(row.index, row.values, "-", lw=1.2, label=str(d))
            else:
                ax.plot(row.index, row.values, "-", lw=0.6, color="0.7", zorder=1)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("frame")
        ax.set_ylabel("saturated fraction of the droplet")
        ax.set_ylim(-0.02, 1.02)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.axhline(threshold, color="red", ls="--", lw=0.8)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(title="bleached", fontsize=6, title_fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


def gallery_figure(name, stack, sat_value, labels, spots, sat_table, per_frame, frame_pre, threshold, pad=8):
    """Bleached droplets: pre-bleach frame (top) and first post-bleach frame (bottom), saturated
    pixels in red, droplet (green) and spot (cyan) outlines. The values in each panel are the
    saturated fractions of the spot and the whole droplet in that frame (the worst post-bleach
    values are in the report table). * = above the threshold before the bleach."""
    flags = sat_table.set_index("droplet")
    ids = flags.index[flags["role"] == "bleached"]
    fig, axes = plt.subplots(2, len(ids), squeeze=False, figsize=(2.5 * len(ids), 5.5))
    for j, d in enumerate(ids):
        sl = crop_around(labels, d, pad)
        for i, t in enumerate((frame_pre, frame_pre + 1)):
            img = stack[t][sl]
            saturated = img >= sat_value
            ax = axes[i, j]
            ax.imshow(img, cmap="gray", vmin=0, vmax=sat_value)
            if saturated.any():
                ax.imshow(np.ma.masked_where(~saturated, saturated), cmap=RED, alpha=0.7,
                          interpolation="nearest")
            ax.contour(labels[sl] == d, levels=[0.5], colors="lime", linewidths=0.7)
            ax.contour(spots[sl] == d, levels=[0.5], colors="cyan", linewidths=0.7)
            ax.text(0.03, 0.97, f"spot {per_frame['spot'].loc[d, t]:.0%}\n"
                                f"droplet {per_frame['droplet'].loc[d, t]:.0%}",
                    transform=ax.transAxes, color="yellow", fontsize=7, va="top")
            ax.axis("off")
        mark = " *" if (flags.loc[d, "droplet_pre_above"] or flags.loc[d, "spot_pre_above"]) else ""
        axes[0, j].set_title(f"droplet {d}{mark}", fontsize=8)
    axes[0, 0].text(-0.15, 0.5, "pre-bleach", transform=axes[0, 0].transAxes, rotation=90, va="center")
    axes[1, 0].text(-0.15, 0.5, "first post-bleach", transform=axes[1, 0].transAxes, rotation=90,
                    va="center")
    fig.suptitle(f"{name}: bleached droplets (red = saturated; values = saturated fraction in that "
                 f"frame; * = above {threshold:.0%} before the bleach)", fontsize=10)
    fig.tight_layout()
    return fig