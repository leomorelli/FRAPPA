"""Small helper functions shared by the steps."""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")                     # figures are only saved, never displayed
import matplotlib.pyplot as plt


def save_figure(fig, folder, name, formats=("png",), dpi=150):
    """Save a figure in each of the formats (e.g. png, pdf, svg) in folder (created if needed)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(folder / f"{name}.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def crop_around(labels, droplet_id, pad):
    """Slices of a box around one droplet, with pad pixels of margin (clipped at the image edge)."""
    rows, cols = np.nonzero(labels == droplet_id)      # row and column coordinates of all the droplet’s pixels
    return (slice(max(rows.min() - pad, 0), rows.max() + 1 + pad),
            slice(max(cols.min() - pad, 0), cols.max() + 1 + pad))