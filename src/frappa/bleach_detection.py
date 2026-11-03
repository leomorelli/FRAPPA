"""Bleach detection: compares the pre-bleach frame with the first post-bleach frame.

- motion check: droplets that gained signal just outside their mask moved and are excluded
- relative drop: fraction of the pre-bleach signal lost, pixel by pixel
- bleached droplets: peak relative drop (95th percentile) of at least min_peak_drop
- bleached spot: pixels whose drop is at least spot_fraction x the droplet's peak drop
"""

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import filters, measure
from skimage.segmentation import expand_labels, find_boundaries


def droplet_motion(pre_s, post_s, labels, territory, gain_frac):
    """For every droplet, the signal gained just outside it between the (smoothed) pre- and post-bleach frames.
    Bleaching removes signal: gained signal means the droplet moved, became brighter, or a bright neighbour came close.
    - ring: pixels within `territory` pixels outside the droplet, not closer to another droplet
    - gained pixel: post - pre larger than gain_frac x the droplet's median pre-bleach intensity
    - shift_px: gained pixels / droplet diameter, a rough displacement estimate (0 = stayed)"""
    terr = expand_labels(labels, distance=territory)    # each droplet + its own ring
    gain = post_s - pre_s
    rows = []
    for d in np.unique(labels)[1:]:
        droplet = labels == d
        ring = (terr == d) & ~droplet
        level = np.median(pre_s[droplet])    # median pre-bleach intensity
        gained = ring & (gain > gain_frac * level)
        diameter = 2 * np.sqrt(droplet.sum() / np.pi)
        rows.append({"droplet": d, "gain_px": int(gained.sum()), "shift_px": gained.sum() / diameter})
    return pd.DataFrame(rows)


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
    table of the bleached droplets."""

    pre_s = filters.gaussian(pre, sigma=p["bl_sigma"], preserve_range=True)
    post_s = filters.gaussian(post, sigma=p["bl_sigma"], preserve_range=True)
    
    motion = droplet_motion(pre_s, post_s, labels, p["territory"], p["gain_frac"])
    drop = relative_drop(pre, post, labels)
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
    return tab, spots, pd.DataFrame(rows)