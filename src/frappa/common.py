"""Small helper functions shared by the steps."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")                     # figures are only saved, never displayed
import matplotlib.pyplot as plt


def save_figure(fig, folder, name):
    """Save a figure as PNG in folder (created if needed) and close it."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)