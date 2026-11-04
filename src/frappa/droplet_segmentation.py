"""Droplet segmentation on the pre-bleach frame.

percentile normalization -> Gaussian smoothing -> CLAHE -> Otsu threshold -> removal of small objects 
-> hole filling -> distance-transform watershed -> removal of droplets touching the border
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage import exposure, filters, measure, morphology, segmentation
from skimage.feature import peak_local_max
from skimage.color import label2rgb


def normalize(img, pct=(2, 98)):
    """Percentile normalization to [0, 1] (needed by CLAHE)."""
    low, high = np.percentile(img, pct)
    return np.clip((img - low) / (high - low), 0, 1)


def remove_small(mask, min_droplet_area):
    """Keep objects with at least min_droplet_area pixels, then fill holes inside droplets."""
    clean = morphology.remove_small_objects(mask, max_size=min_droplet_area - 1, connectivity=2)
    return ndi.binary_fill_holes(clean)


def split_touching(clean_mask, min_peak_distance, min_droplet_area):
    """Distance-transform watershed, then removal of fragments smaller than min_droplet_area."""
    distance = ndi.distance_transform_edt(clean_mask)
    components = measure.label(clean_mask, connectivity=2)
    peaks = peak_local_max(distance, min_distance=min_peak_distance, labels=components, exclude_border=False)
    markers = np.zeros(clean_mask.shape, dtype=np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    labels = segmentation.watershed(-distance, markers=markers, mask=clean_mask, connectivity=2)
    labels = morphology.remove_small_objects(labels, max_size=min_droplet_area - 1)
    labels, _, _ = segmentation.relabel_sequential(labels)
    return labels, distance, peaks


def segment(frame, p):
    """Segment of the pre-bleach frame.
    p: dict with sigma, kernel_size, clip_limit, min_droplet_area, min_peak_distance.
    Returns: a dict with every intermediate image (for the summary figure), and
    the droplet label image in steps["final"], one integer ID per droplet, 0 = background). """

    steps = {"raw": frame}
    # create mask: Gaussian smoothing -> CLAHE -> Otsu threshold. Input must be normalized to [0, 1].
    smooth = filters.gaussian(normalize(frame), sigma=p["sigma"], preserve_range=True)
    steps["smooth"] = np.clip(smooth, 0, 1)
    steps["clahe"] = exposure.equalize_adapthist(steps["smooth"], kernel_size=p["kernel_size"], clip_limit=p["clip_limit"])
    steps["otsu"] = steps["clahe"] > filters.threshold_otsu(steps["clahe"])
    
    # mask cleaning and watersheding
    steps["clean"] = remove_small(steps["otsu"], p["min_droplet_area"])
    steps["watershed"], steps["distance"], steps["peaks"] = split_touching(steps["clean"], p["min_peak_distance"], p["min_droplet_area"])
    final = segmentation.clear_border(steps["watershed"])                  # drop droplets touching the border
    steps["final"], _, _ = segmentation.relabel_sequential(final)
    return steps


def summary_figure(name, steps, p):
    """Figure with the segmentation steps."""
    img = steps["raw"]
    lo, hi = np.percentile(img, (1, 99.5))
    removed = steps["otsu"] & ~steps["clean"]                         # too small
    border = (steps["watershed"] > 0) & (steps["final"] == 0)         # touching the edge

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    ax = axes.ravel()
    ax[0].imshow(img, cmap="gray", vmin=lo, vmax=hi)
    ax[0].set_title("1. pre-bleach frame")
    ax[1].imshow(steps["smooth"], cmap="gray")
    ax[1].set_title(f"2. normalized + Gaussian (sigma={p['sigma']})")
    ax[2].imshow(steps["clahe"], cmap="gray")
    ax[2].set_title(f"3. CLAHE (k={p['kernel_size']}, cl={p['clip_limit']})")
    ax[3].imshow(steps["otsu"], cmap="gray")
    ax[3].set_title("4. Otsu threshold")
    ax[4].imshow(steps["clean"], cmap="gray")
    if removed.any():
        ax[4].contour(removed, levels=[0.5], colors="red", linewidths=0.7)
    ax[4].set_title(f"5. small objects removed (red, < {p['min_droplet_area']} px)")
    ax[5].imshow(steps["distance"], cmap="magma")
    ax[5].plot(steps["peaks"][:, 1], steps["peaks"][:, 0], "c.", markersize=4)
    ax[5].set_title(f"6. distance map + {len(steps['peaks'])} seeds (d={p['min_peak_distance']})")
    ax[6].imshow(segmentation.mark_boundaries(label2rgb(steps["watershed"], bg_label=0), steps["watershed"], color=(1, 1, 1)))
    ax[6].set_title(f"7. watershed: {steps['watershed'].max()} droplets")
    display = np.clip((img - lo) / (hi - lo), 0, 1)                   # scaled to [0, 1] for mark_boundaries
    ax[7].imshow(segmentation.mark_boundaries(display, steps["final"], color=(0, 1, 0)))
    if border.any():
        ax[7].contour(border, levels=[0.5], colors="red", linewidths=0.7)
    ax[7].set_title(f"8. final: {steps['final'].max()} droplets (red = touching the edge)")
    for a in ax:
        a.axis("off")
        a.title.set_fontsize(9)
    fig.suptitle(f"{name}: droplet segmentation", fontsize=13)
    fig.tight_layout()
    return fig