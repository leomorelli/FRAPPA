"""Saturation report (optional, with --saturation-report): how many pixels of each droplet and
spot are saturated, i.e. at the maximum value of the data type (255 for 8-bit images).

Considers for the bleached droplets, the whole droplet and its spot; for the references, the whole droplet.
For each region (whole droplet or spot), reports the fraction of saturated pixels in every frame,
the fraction in the pre-bleach frame, and the worst post-bleach fraction with its frame.

Bleached droplets whose spot or whole droplet has a saturated fraction above the threshold before
the bleach are flagged. Nothing is excluded: the report only shows where saturation may affect the results.
"""


import numpy as np
import pandas as pd


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
            
