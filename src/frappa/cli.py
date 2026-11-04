"""frappa: FRAP analysis of bleached droplets."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from . import __version__
from . import bleach_detection, droplet_segmentation, recovery_curves
from .common import save_figure


FRAME_PRE = 0 
FRAME_POST = 1    # index of the first post-bleach frame


def build_parser():
    parser = argparse.ArgumentParser(prog="frappa", description=__doc__)
    parser.add_argument("-i", "--input", type=Path, required=True, help="folder with the TIFF stacks")
    parser.add_argument("-o", "--output", type=Path, required=True, help="output folder")

    seg = parser.add_argument_group("segmentation parameters")
    seg.add_argument("--sigma", type=float, default=0.5, help="Gaussian smoothing before CLAHE, in pixels (default: %(default)s)")
    seg.add_argument("--kernel-size", type=int, default=64, help="CLAHE kernel size, in pixels (default: %(default)s)")
    seg.add_argument("--clip-limit", type=float, default=0.01, help="CLAHE clip limit (default: %(default)s)")
    seg.add_argument("--min-droplet-area", type=int, default=150, help="smaller objects are removed, in pixels (default: %(default)s)")
    seg.add_argument("--min-peak-distance", type=int, default=6, help="minimum distance between watershed seeds, in pixels (default: %(default)s)")

    bl = parser.add_argument_group("bleach detection parameters")
    bl.add_argument("--bl-sigma", type=float, default=1.0, help="smoothing of the pre/post frames, in pixels (default: %(default)s)")
    bl.add_argument("--territory", type=int, default=5,
                    help="width of the ring around each droplet checked for gained signal, in pixels (default: %(default)s)")
    bl.add_argument("--gain-frac", type=float, default=0.25,
                    help="a ring pixel counts as gained if it brightened by more than this fraction of the droplet intensity (default: %(default)s)")
    bl.add_argument("--max-shift", type=float, default=1.0,
                    help="droplets with a larger estimated shift, in pixels, count as moved and are excluded (default: %(default)s)")
    bl.add_argument("--min-peak-drop", type=float, default=0.4,
                    help="a droplet is bleached if its peak relative drop is at least this (default: %(default)s)")
    bl.add_argument("--spot-fraction", type=float, default=0.75,
                    help="spot = pixels whose drop is at least this fraction of the droplet's peak drop (default: %(default)s)")

    fig = parser.add_argument_group("figures")
    fig.add_argument("--all-plots", action="store_true", help="also save the diagnostic figures of every step")
    fig.add_argument("--figure-format", nargs="+", default=["png"], choices=["png", "pdf", "svg"],
                     help="file format(s) of the figures (default: %(default)s), space separated for multiple formats")
    fig.add_argument("--dpi", type=int, default=150, help="resolution of the figures (default: %(default)s)")

    return parser


def main():
    args = build_parser().parse_args()
    seg_params = {"sigma": args.sigma, "kernel_size": args.kernel_size, "clip_limit": args.clip_limit,
                  "min_droplet_area": args.min_droplet_area, "min_peak_distance": args.min_peak_distance}
    bleach_params = {"bl_sigma": args.bl_sigma, "territory": args.territory, "gain_frac": args.gain_frac, "max_shift": args.max_shift,
                     "min_peak_drop": args.min_peak_drop, "spot_fraction": args.spot_fraction}
    fig_opts = {"formats": args.figure_format, "dpi": args.dpi}

    files = sorted(p for p in args.input.iterdir() if p.suffix.lower() in (".tif", ".tiff"))
    print(f"{len(files)} TIFF file(s) in {args.input}")

    # record how this run was made
    args.output.mkdir(parents=True, exist_ok=True)
    run_info = {"frappa_version": __version__,
                "date": datetime.now().isoformat(timespec="seconds"),
                "command": "frappa " + " ".join(sys.argv[1:]),
                "input_files": [f.name for f in files],
                "figures": {"all_plots": args.all_plots, **fig_opts},
                "parameters": {"segmentation": seg_params}}
    with open(args.output / "run_info.json", "w") as fh:
        json.dump(run_info, fh, indent=2)

    seg_dir = args.output / "segmentation"
    bleach_dir = args.output / "bleach_detection"
    curves_dir = args.output / "recovery_curves"
    for folder in (seg_dir, bleach_dir, curves_dir):
        folder.mkdir(exist_ok=True)

    print("Segmentation of the pre-bleach frame: " + ", ".join(f"{k}={v}" for k, v in seg_params.items()))
    print("Bleach detection: " + ", ".join(f"{k}={v}" for k, v in bleach_params.items()))

    all_curves = []
    for f in files:
        name = f.stem
        stack = tifffile.imread(f).astype(np.float64)              # (frames, rows, columns)
        pre, post = stack[FRAME_PRE], stack[FRAME_POST]

        # 1. droplets in the pre-bleach frame
        steps = droplet_segmentation.segment(pre, seg_params)
        labels = steps["final"]
        tifffile.imwrite(seg_dir / f"{name}_droplet_labels.tif", labels.astype(np.uint16))
        if args.all_plots:
            save_figure(droplet_segmentation.summary_figure(name, steps, seg_params),
                        seg_dir / "figures", f"{name}_segmentation_summary", **fig_opts)

        # 2. bleached droplets and spots
        tab, spots, spot_tab = bleach_detection.detect(pre, post, labels, bleach_params)
        tifffile.imwrite(bleach_dir / f"{name}_spot_labels.tif", spots.astype(np.uint16))
        tab.to_csv(bleach_dir / f"{name}_all_droplets.csv", index=False)
        spot_tab.to_csv(bleach_dir / f"{name}_bleached_droplets.csv", index=False)
        save_figure(bleach_detection.overview_figure(name, pre, post, labels, spots, tab),
                    bleach_dir / "figures", f"{name}_bleach_overview", **fig_opts)

        # 3. recovery curves of the bleached droplets
        curves = recovery_curves.measure(stack, labels, spots, tab, FRAME_PRE)
        curves.insert(0, "file", name)
        curves.to_csv(curves_dir / f"{name}_recovery_curves.csv", index=False)
        if not curves.empty:
            save_figure(recovery_curves.curves_figure(curves),
                        curves_dir / "figures", f"{name}_recovery_curves", **fig_opts)
        all_curves.append(curves)

        print(f"  {name}: {labels.max()} droplets, {int(tab['moved'].sum())} moved (excluded), {int(tab['bleached'].sum())} bleached")

    all_curves = pd.concat(all_curves, ignore_index=True)
    all_curves.to_csv(curves_dir / "recovery_curves.csv", index=False)
    if not all_curves.empty:
        save_figure(recovery_curves.all_files_curves_figure(all_curves),
                    curves_dir / "figures", "mean_final_curves", **fig_opts)

    print(f"Done. Results in {args.output}")


if __name__ == "__main__":
    main()