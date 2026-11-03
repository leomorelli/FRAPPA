"""Small helper functions shared by the steps."""

from pathlib import Path

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