# FRAP image preprocessing and cell segmentation: notebook workflow

This guide prepares FRAP microscopy images for segmentation, using small steps that can be inspected interactively in a notebook.

Use the **first, prebleach frame** to identify the fluorescent objects before bleaching creates dark regions inside them. In your acquisition, frame 1 is prebleach, frame 2 is the first postbleach observation, and frames 3-21 show recovery.

The workflow is:

**Prebleach image → Gaussian smoothing → Otsu foreground mask → mask cleanup → distance transform → watershed separation.**

Gaussian filtering prepares the image. Otsu thresholding is already the first segmentation step: it separates foreground from background.

Keep the original intensity images unchanged. Use a separate copy for segmentation; later FRAP measurements must use the original or appropriately corrected, unsmoothed intensity images.

## Before starting

The examples assume a grayscale TIFF array called `raw`, with shape `(frames, height, width)`. Select channels or Z planes explicitly if your data contain them.

Install the required libraries in your notebook environment if necessary:

```bash
python -m pip install numpy scipy matplotlib scikit-image tifffile
```

Run these imports:

```python
import numpy as np
import matplotlib.pyplot as plt
import tifffile

from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.feature import peak_local_max
from skimage.segmentation import watershed, mark_boundaries
```

If the data are not already loaded, a single multipage grayscale TIFF can be read as follows:

```python
raw = tifffile.imread("experiment.tif")

print("Shape:", raw.shape)
print("Data type:", raw.dtype)

if raw.ndim != 3 or raw.shape[0] != 21:
    raise ValueError("Expected 21 grayscale frames with shape (21, height, width).")
```

Use the original acquisition TIFFs, not RGB display exports. If frames are separate TIFFs, assemble them in verified acquisition order before proceeding.

## 1. Select the prebleach frame

### Purpose

Use the intact fluorescence pattern to define object geometry. Converting to floating point allows filtering without integer rounding.

### Notebook code

```python
image = raw[0].astype(float, copy=True)

if image.ndim != 2 or not np.isfinite(image).all():
    raise ValueError("Use a finite 2D prebleach image.")

plt.figure(figsize=(7, 7))
plt.imshow(image, cmap="gray")
plt.title("Prebleach image")
plt.show()
```

### What to check

- The expected fluorescent objects are visible.
- The selected image is the prebleach frame.
- Image axes and orientation are correct.
- The array contains a single grayscale channel.

If registered images contain NaN borders, first crop to the common valid region and retain the crop coordinates. The functions below expect finite image values.

**Output:** `image`, a floating-point copy of the prebleach frame.

## 2. Reduce noise with Gaussian filtering

### Purpose

A Gaussian filter replaces each pixel with a weighted average of nearby pixels. This reduces small intensity fluctuations that could otherwise create small foreground objects or irregular boundaries.

`sigma` controls the spatial scale of smoothing, in pixels. Larger values produce stronger smoothing.

### Function

```python
def smooth_image(image, sigma=1.0):
    """Gaussian smoothing for segmentation only."""
    image = np.asarray(image, dtype=float)

    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("Sigma must be nonnegative and finite.")

    return ndi.gaussian_filter(image, sigma=sigma)
```

### Notebook code

```python
sigma = 1.0
smoothed = smooth_image(image, sigma=sigma)

vmin, vmax = np.percentile(image, [1, 99.5])

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

for ax, data, title in zip(
    axes,
    [image, smoothed],
    ["Original", f"Gaussian sigma = {sigma} pixels"],
):
    ax.imshow(data, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(title)

plt.show()
```

### What to check

Noise should decrease while object edges and narrow gaps between neighboring objects remain visible. The shared display scale makes the two images comparable.

Try `sigma = 0.5, 1.0, 1.5, 2.0` as an initial sensitivity check. These are starting values, not universal settings. Excessive smoothing can erase gaps and merge nearby objects.

**Output:** `smoothed`, a segmentation-only intensity image.

## 3. Create the foreground mask with Otsu thresholding

### Purpose

Otsu selects a threshold from the intensity histogram by maximizing separation between two intensity classes. Here, those classes are intended to represent background and fluorescent foreground.

A global threshold is a reasonable starting point when background intensity is relatively uniform. A threshold does not yet assign a separate identity to every touching cell.

### Function

```python
def otsu_foreground(image, threshold_factor=1.0):
    """
    Return a foreground mask and the applied threshold.

    factor=1 uses the Otsu threshold directly.
    Other factors are manual adjustments.
    """
    if not np.isfinite(threshold_factor) or threshold_factor <= 0:
        raise ValueError("threshold_factor must be positive and finite.")

    if np.ptp(image) == 0:
        raise ValueError("Image has no intensity contrast.")

    threshold = threshold_otsu(image) * threshold_factor
    mask = image > threshold

    return mask, threshold
```

### Notebook code

Start with the unmodified Otsu threshold:

```python
threshold_factor = 1.0

foreground, threshold = otsu_foreground(
    smoothed,
    threshold_factor=threshold_factor,
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(smoothed.ravel(), bins=256)
axes[0].axvline(threshold, color="red", label=f"Threshold = {threshold:.2f}")
axes[0].set_yscale("log")
axes[0].set_xlabel("Intensity")
axes[0].set_ylabel("Pixel count")
axes[0].legend()

axes[1].imshow(foreground, cmap="gray")
axes[1].set_title("White = foreground; black = background")

plt.show()
```

### What to check

- Are dim objects missing?
- Are object boundaries reasonably complete?
- Has background been included?
- Have neighboring objects become connected?

If dim objects are missed, try a lower factor, such as `0.8` or `0.6`. This is a tunable modification, not part of standard Otsu threshold selection. Lowering the threshold may also include background or connect neighboring objects.

If one threshold cannot handle different parts of the image, investigate uneven background or local thresholding rather than continually lowering the global threshold. Do not add background-flattening or contrast transformations automatically without first checking whether they are needed.

**Output:** `foreground`, a boolean image in which `True` represents foreground.

## 4. Clean the binary mask

### Purpose

Two useful operations are:

- **Remove small components:** discard isolated regions too small to represent the target objects.
- **Fill enclosed holes:** useful when the intended object masks should be solid.

### Function

```python
def clean_foreground(mask, min_area=40, fill_holes=True):
    """Remove small connected regions and optionally fill enclosed holes."""
    mask = np.asarray(mask, dtype=bool).copy()

    if min_area < 1:
        raise ValueError("min_area must be positive.")

    if fill_holes:
        mask = ndi.binary_fill_holes(mask)

    components, _ = ndi.label(mask)
    sizes = np.bincount(components.ravel())

    keep = sizes >= min_area
    keep[0] = False  # Never retain the background component.

    return keep[components]
```

### Notebook code

```python
cleaned = clean_foreground(
    foreground,
    min_area=40,
    fill_holes=True,
)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(foreground, cmap="gray")
axes[0].set_title("Before cleanup")

axes[1].imshow(cleaned, cmap="gray")
axes[1].set_title("After cleanup")

plt.show()
```

### What to check

Small artifacts should disappear while genuine small objects remain. `min_area` is an area in pixels, not an intensity threshold. A small artifact attached to a large object will not be removed by this connected-component filter.

Hole filling can erase meaningful internal cavities. Set `fill_holes=False` if the target structures are genuinely hollow. Start without morphological dilation or closing: these operations can connect neighboring cells.

**Output:** `cleaned`, a foreground mask suitable for preparing watershed markers.

## 5. Calculate the distance transform

### Purpose

For each foreground pixel, the distance transform measures its distance to the nearest background pixel. Values are generally higher near object centers.

### Notebook code

```python
distance = ndi.distance_transform_edt(cleaned)

plt.figure(figsize=(7, 7))
plt.imshow(distance, cmap="magma")
plt.colorbar(label="Distance to background, pixels")
plt.title("Distance transform")
plt.show()
```

### What to check

Individual rounded objects should produce distinct elevated regions. Touching objects may share one foreground component but still contain separate distance peaks.

**Output:** `distance`, a floating-point distance map.

## 6. Place seeds and separate touching objects with watershed

### Purpose

Seeds represent candidate object centers. Watershed expands those seeds within the foreground mask to assign individual labels.

This step assumes that distance peaks provide useful candidate centers. Irregular objects can contain several peaks, so visual validation remains necessary.

### Notebook code

```python
min_peak_distance = 6

# Separate component IDs ensure nearby disconnected objects are
# considered independently when finding their distance peaks.
components, _ = ndi.label(cleaned)

coordinates = peak_local_max(
    distance,
    min_distance=min_peak_distance,
    labels=components,
    exclude_border=False,
)

markers = np.zeros(cleaned.shape, dtype=np.int32)

if len(coordinates) == 0:
    raise ValueError("No seeds detected. Inspect the foreground mask.")

markers[tuple(coordinates.T)] = np.arange(
    1, len(coordinates) + 1
)

cell_labels = watershed(
    -distance,
    markers=markers,
    mask=cleaned,
)
```

Inspect the seeds:

```python
plt.figure(figsize=(7, 7))
plt.imshow(distance, cmap="magma")
plt.scatter(
    coordinates[:, 1],
    coordinates[:, 0],
    color="cyan",
    s=15,
)
plt.title("Candidate object centers")
plt.show()
```

### What to check

Aim for approximately **one seed per intended object**.

| Observation | Adjustment to test |
| --- | --- |
| Multiple seeds inside one cell | Increase `min_peak_distance` |
| Touching cells receive only one seed | Decrease `min_peak_distance` |
| Seeds occur in background artifacts | Revisit thresholding and mask cleanup |
| A dim object has no foreground mask | Revisit the foreground detection before adjusting seeds |

`min_peak_distance` is a distance in pixels. It does not specify a cell radius. Because `exclude_border=False`, partial objects at image borders can be included; decide how those objects will be handled in later analysis.

**Outputs:** `coordinates`, `markers`, and `cell_labels`. Label 0 represents background; positive integer labels identify separate objects.

## 7. Inspect final boundaries on the original image

### Purpose

Evaluate whether the complete sequence of processing steps produced plausible object boundaries and individual-object identities.

### Notebook code

```python
display_image = (
    (image - image.min())
    / max(np.ptp(image), 1e-12)
)

overlay = mark_boundaries(
    display_image,
    cell_labels,
    color=(1, 0, 0),
)

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

axes[0].imshow(cell_labels, cmap="nipy_spectral", interpolation="nearest")
axes[0].set_title(f"Labelled objects: {cell_labels.max()}")

axes[1].imshow(overlay)
axes[1].set_title("Boundaries on the prebleach image")

plt.show()
```

The normalization used for `display_image` is for visualization only. It does not change `raw`, `image`, or the data used for later fluorescence measurements.

### What to check

- Each intended cell or fluorescent object has one label.
- Touching objects are separated where the image supports separation.
- Boundaries follow the intended fluorescent structures.
- Dim objects have not been systematically omitted.
- Large objects have not been split into many fragments.
- Background artifacts have not been retained.

The labels follow the fluorescent structures visible in the selected channel. If fluorescence labels nuclei, the result represents nuclear masks; full cell boundaries cannot be inferred reliably from that channel alone.

**Output:** a reviewed instance-label image, `cell_labels`.

## Parameter adjustment order

Change one stage at a time and rerun every downstream stage after an upstream change.

| Order | Parameter | Goal |
| --- | --- | --- |
| 1 | `sigma` | Reduce noise while preserving object boundaries and gaps |
| 2 | `threshold_factor` | Include the intended fluorescent foreground |
| 3 | `min_area` | Remove small disconnected artifacts |
| 4 | `min_peak_distance` | Obtain appropriate individual-object splits |

These parameters interact. Visual inspection is an initial development tool, not an objective accuracy benchmark. Later, compare predictions against manually annotated objects on representative development and held-out images.

## Arrays produced by this workflow

| Array | Meaning | Suitable for fluorescence measurement? |
| --- | --- | --- |
| `raw` | Original TIFF sequence | Yes, after appropriate QC and any necessary quantitative corrections |
| `image` | Floating-point copy of the prebleach frame | Only the prebleach observation; preserve appropriate intensity calibration |
| `smoothed` | Gaussian-filtered prebleach image | No; segmentation only |
| `foreground` | Initial binary foreground mask | A geometric mask, not an intensity image |
| `cleaned` | Cleaned binary foreground mask | A geometric mask |
| `distance` | Distance to foreground boundary | No |
| `markers` | Candidate object-center IDs | No |
| `cell_labels` | Separate object IDs | Use to define identity and geometry, not as intensity data |

## FRAP-specific next steps

The resulting `cell_labels` defines object identities. Bleached-cell identification comes afterward, by comparing prebleach and postbleach images within those objects.

Do not independently threshold every recovery frame to define measurement ROIs: recovery itself would change their apparent size. Account for drift and, when necessary, object motion before following measurement regions over time.

Selecting a bleached cell does not mean that recovery should automatically be measured over the entire cell. A local bleach generally requires a separate bleach-spot ROI within that cell.

Saturated images can still be useful for segmentation development. Before quantitative FRAP fitting, check clipping within each selected bleach ROI and reference region, particularly in the prebleach frame. Segmentation preprocessing cannot reconstruct clipped intensity values.

## Documentation

- [scikit-image thresholding guide](https://scikit-image.org/docs/stable/auto_examples/applications/plot_thresholding_guide.html)
- [scikit-image watershed and segmentation API](https://scikit-image.org/docs/stable/api/skimage.segmentation.html)
- [scikit-image peak detection API](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.peak_local_max)
- [SciPy Gaussian filtering](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html)
- [SciPy distance transform](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html)
