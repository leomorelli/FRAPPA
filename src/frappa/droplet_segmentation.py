"""Droplet segmentation on the pre-bleach frame.

percentile normalization -> Gaussian smoothing -> CLAHE -> Otsu threshold -> removal of small objects 
-> hole filling -> distance-transform watershed -> removal of droplets touching the border
"""

import numpy as np
from scipy import ndimage as ndi
from skimage import exposure, filters, measure, morphology, segmentation
from skimage.feature import peak_local_max

SIGMA = 0.5
KERNEL_SIZE = 64
CLIP_LIMIT = 0.01
MIN_DROPLET_AREA = 150
MIN_PEAK_DISTANCE = 6


def normalize(img, pct=(2, 98)):
    """Percentile normalization to [0, 1] (needed by CLAHE)."""
    low, high = np.percentile(img, pct)
    return np.clip((img - low) / (high - low), 0, 1)


def create_mask(normalized, sigma, kernel_size, clip_limit):
    """Gaussian smoothing -> CLAHE -> Otsu threshold. Input must be normalized to [0, 1]."""
    smooth = filters.gaussian(normalized, sigma=sigma, preserve_range=True)
    smooth = np.clip(smooth, 0, 1)
    local = exposure.equalize_adapthist(smooth, kernel_size=kernel_size, clip_limit=clip_limit)
    return local > filters.threshold_otsu(local)


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
    return labels


def segment(frame):
    """Droplet label image of one pre-bleach frame (one integer ID per droplet, 0 = background)."""
    mask = create_mask(normalize(frame), SIGMA, KERNEL_SIZE, CLIP_LIMIT)
    labels = split_touching(remove_small(mask, MIN_DROPLET_AREA), MIN_PEAK_DISTANCE, MIN_DROPLET_AREA)
    labels = segmentation.clear_border(labels)              # drop droplets touching the border
    labels, _, _ = segmentation.relabel_sequential(labels)
    return labels