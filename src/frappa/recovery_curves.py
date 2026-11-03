"""Recovery curves of the bleached droplets.

  A = spot(t) / ref(t)        B = A(t) / A(pre)
  C = tot(t)  / ref(t)        D = C(t) / C(pre)
  final = B(t) / D(t)

spot = bleached region, tot = whole bleached droplet, ref = reference droplets (not moved, not
bleached; all their pixels pooled). All are mean intensities per frame. No background subtraction.
"""

import numpy as np
import pandas as pd


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