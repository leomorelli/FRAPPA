"""Bleach detection: compares the pre-bleach frame with the first post-bleach frame.

- motion check: droplets that gained signal just outside their mask moved and are excluded
- relative drop: fraction of the pre-bleach signal lost, pixel by pixel
- bleached droplets: peak relative drop (95th percentile) of at least min_peak_drop
- bleached spot: pixels whose drop is at least spot_fraction x the droplet's peak drop
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import filters, measure
from skimage.segmentation import expand_labels, find_boundaries
from .common import crop_around


def droplet_motion(pre_s, post_s, labels, territory, gain_frac):
    """For every droplet, the signal gained just outside it between the (smoothed) pre- and post-bleach frames.
    Bleaching removes signal: gained signal means the droplet moved, became brighter, or a bright neighbour came close.
    - ring: pixels within `territory` pixels outside the droplet, not closer to another droplet
    - gained pixel: post - pre larger than gain_frac x the droplet's median pre-bleach intensity
    - shift_px: gained pixels / droplet diameter, a rough displacement estimate (0 = stayed)
    Returns the table and an image of the gained pixels, labelled with their droplet's ID."""
    terr = expand_labels(labels, distance=territory)    # each droplet + its own ring
    gain = post_s - pre_s
    gained_img = np.zeros_like(labels)    # empty image of the same shape and type as the labels
    rows = []
    for d in np.unique(labels)[1:]:
        droplet = labels == d
        ring = (terr == d) & ~droplet
        level = np.median(pre_s[droplet])    # median pre-bleach intensity
        gained = ring & (gain > gain_frac * level)
        gained_img[gained] = d    # writes the droplet’s ID on its gained pixels
        diameter = 2 * np.sqrt(droplet.sum() / np.pi)
        rows.append({"droplet": d, "gain_px": int(gained.sum()), "shift_px": gained.sum() / diameter})
    return pd.DataFrame(rows), gained_img


def relative_drop(pre_s, post_s, labels):
    """Fraction of the pre-bleach signal lost in the first post-bleach frame, pixel by pixel.
    1 = fully bleached, 0 = unchanged. Set to 0 outside droplets.
    No background subtraction: the microscope clips the background at 0."""
    drop = (pre_s - post_s) / np.maximum(pre_s, 1e-6)
    drop[labels == 0] = 0
    return drop


def droplet_drop_table(drop, labels):
    """For every droplet: its size and how strongly its most-bleached part dropped."""
    rows = []
    for d in np.unique(labels)[1:]:                  # droplet IDs (0 = background is skipped)
        vals = drop[labels == d]                     # drop values of this droplet's pixels
        rows.append({"droplet": d, "area": vals.size,
                     "peak_drop": np.percentile(vals, 95),      # robust maximum
                     "mean_drop": vals.mean()})
    return pd.DataFrame(rows)


def bleach_spot(drop, labels, droplet_id, fraction):
    """Bleached spot of one droplet: pixels whose drop is at least `fraction` of the droplet's
    peak drop. Largest connected region, holes filled."""
    inside = labels == droplet_id
    peak = np.percentile(drop[inside], 95)
    spot = inside & (drop >= fraction * peak)
    lab = measure.label(spot, connectivity=2)
    if lab.max() > 1:    # if more than one "spot" is detected only the largest one is kept
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0    # sets the background to 0
        spot = lab == sizes.argmax()
    return ndi.binary_fill_holes(spot) & inside


def detect(pre, post, labels, p):
    """Bleached droplets and spots of one file. p: dict with bl_sigma, territory, gain_frac, max_shift, min_peak_drop, spot_fraction.
    Returns: table of all droplets (with 'moved' and 'bleached'), spot label image (each spot has the ID of its droplet),
    table of the bleached droplets, the gained image"""

    pre_s = filters.gaussian(pre, sigma=p["bl_sigma"], preserve_range=True)
    post_s = filters.gaussian(post, sigma=p["bl_sigma"], preserve_range=True)
    
    motion, gained = droplet_motion(pre_s, post_s, labels, p["territory"], p["gain_frac"])
    drop = relative_drop(pre_s, post_s, labels)
    tab = droplet_drop_table(drop, labels).merge(motion, on="droplet")
    tab["moved"] = tab["shift_px"] > p["max_shift"]
    tab["bleached"] = (tab["peak_drop"] >= p["min_peak_drop"]) & ~tab["moved"]

    spots = np.zeros_like(labels)
    rows = []
    for d in tab.loc[tab["bleached"], "droplet"]:
        spot = bleach_spot(drop, labels, d, p["spot_fraction"])
        spots[spot] = d                                       # spot keeps the ID of its droplet
        droplet = labels == d
        edge = find_boundaries(droplet, mode="inner")          # outermost pixels of the droplet
        cy, cx = ndi.center_of_mass(droplet)
        sy, sx = ndi.center_of_mass(spot)
        rows.append({"droplet": d,
                     "droplet_area": int(droplet.sum()),
                     "spot_area": int(spot.sum()),
                     "spot_to_droplet_area": spot.sum() / droplet.sum(),
                     "bleach_depth": drop[spot].mean(),            # mean relative drop in the spot
                     "spot_offset": np.hypot(sy - cy, sx - cx) / np.sqrt(droplet.sum() / np.pi),
                     "spot_touches_edge": bool((spot & edge).any())})
    return tab, spots, pd.DataFrame(rows), gained

## Figures ##

def moved_figure(name, pre, post, labels, gained, tab, pad=10):
    """Droplets that moved: pre- and post-bleach crops with the droplet mask (green) and the gained signal (red)."""
    ids = tab.loc[tab["moved"], "droplet"]
    lo, hi = np.percentile(pre, (1, 99.5))
    fig, axes = plt.subplots(2, len(ids), squeeze=False, figsize=(3 * len(ids), 6))
    for j, d in enumerate(ids):
        sl = crop_around(labels, d, pad)
        for i, img in enumerate((pre, post)):
            ax = axes[i, j]
            ax.imshow(img[sl], cmap="gray", vmin=lo, vmax=hi)
            ax.contour(labels[sl] == d, levels=[0.5], colors="lime", linewidths=0.8)
            if (gained[sl] == d).any():
                ax.contour(gained[sl] == d, levels=[0.5], colors="red", linewidths=0.8)
            ax.axis("off")
        shift = tab.loc[tab["droplet"] == d, "shift_px"].iloc[0]
        axes[0, j].set_title(f"droplet {d}\nshift ~{shift:.1f} px", fontsize=9)
    fig.suptitle(f"{name}: moved droplets (top: pre-bleach, bottom: post-bleach)\n"
                 f"green = droplet, red = gained signal", fontsize=10)
    fig.tight_layout()
    return fig


def peak_drop_figure(tables, min_peak_drop):
    """Peak relative drop of the droplets that did not move, sorted from highest to lowest, with the threshold
    One panel per file"""
    fig, axes = plt.subplots(1, len(tables), squeeze=False, figsize=(5 * len(tables), 4))
    for ax, (name, tab) in zip(axes[0], tables.items()):
        drops = np.sort(tab.loc[~tab["moved"], "peak_drop"])[::-1]
        ax.plot(range(1, len(drops) + 1), drops, "o", markersize=4)
        ax.axhline(min_peak_drop, color="red", ls="--", lw=0.8)
        ax.set_title(f"{name}: {(drops >= min_peak_drop).sum()} bleached", fontsize=10)
        ax.set_xlabel("droplets (not moved), sorted")
        ax.set_ylabel("peak relative drop")
    fig.tight_layout()
    return fig


def spots_figure(name, pre, post, labels, spots, tab, pad=8):
    """Bleached droplets: pre-bleach, first post-bleach, and post-bleach with the droplet (green) and its spot (cyan)."""
    ids = tab.loc[tab["bleached"], "droplet"]
    lo, hi = np.percentile(pre, (1, 99.5))
    fig, axes = plt.subplots(3, len(ids), squeeze=False, figsize=(2.5 * len(ids), 8))
    for j, d in enumerate(ids):
        sl = crop_around(labels, d, pad)
        for i, img in enumerate((pre, post, post)):
            ax = axes[i, j]
            ax.imshow(img[sl], cmap="gray", vmin=lo, vmax=hi)
            ax.axis("off")
        axes[2, j].contour(labels[sl] == d, levels=[0.5], colors="lime", linewidths=0.7)
        axes[2, j].contour(spots[sl] == d, levels=[0.5], colors="cyan", linewidths=0.9)
        axes[0, j].set_title(f"droplet {d}", fontsize=9)
    fig.suptitle(f"{name}: bleached droplets (rows: pre-bleach, first post-bleach, spot in cyan)", fontsize=10)
    fig.tight_layout()
    return fig


def overview_figure(name, pre, post, labels, spots, tab):
    """Pre-bleach and first post-bleach frames; on the post-bleach frame: all droplets (green),
    bleached spots (magenta) with the droplet IDs, moved droplets (orange)."""
    lo, hi = np.percentile(pre, (1, 99.5))                  # same contrast for both frames
    moved = np.isin(labels, tab.loc[tab["moved"], "droplet"])
    bleached_ids = tab.loc[tab["bleached"], "droplet"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(pre, cmap="gray", vmin=lo, vmax=hi)
    axes[0].set_title("pre-bleach")
    axes[1].imshow(post, cmap="gray", vmin=lo, vmax=hi)
    axes[1].set_title("first post-bleach")
    axes[2].imshow(post, cmap="gray", vmin=lo, vmax=hi)
    for d in np.unique(labels)[1:]:
        axes[2].contour(labels == d, levels=[0.5], colors="lime", linewidths=0.4)
    if moved.any():
        axes[2].contour(moved, levels=[0.5], colors="orange", linewidths=1.0)
    if spots.any():
        axes[2].contour(spots > 0, levels=[0.5], colors="magenta", linewidths=0.8)
    for d in bleached_ids:                                  # ID above each bleached droplet
        droplet = labels == d
        cy, cx = ndi.center_of_mass(droplet)
        radius = np.sqrt(droplet.sum() / np.pi)
        axes[2].text(cx, cy - radius - 3, str(d), color="yellow", fontsize=8, ha="center")
    axes[2].set_title(f"{len(bleached_ids)} bleached (magenta), {int(tab['moved'].sum())} moved (orange)")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(name, fontsize=13)
    fig.tight_layout()
    return fig