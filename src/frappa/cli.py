"""frappa: FRAP analysis of bleached droplets."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile

from . import __version__
from . import droplet_segmentation


FRAME_PRE = 0 


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
    return parser


def main():
    args = build_parser().parse_args()
    seg_params = {"sigma": args.sigma, "kernel_size": args.kernel_size, "clip_limit": args.clip_limit,
                  "min_droplet_area": args.min_droplet_area, "min_peak_distance": args.min_peak_distance}

    files = sorted(p for p in args.input.iterdir() if p.suffix.lower() in (".tif", ".tiff"))
    print(f"{len(files)} TIFF file(s) in {args.input}")

    # record how this run was made
    args.output.mkdir(parents=True, exist_ok=True)
    run_info = {"frappa_version": __version__,
                "date": datetime.now().isoformat(timespec="seconds"),
                "command": "frappa " + " ".join(sys.argv[1:]),
                "input_files": [f.name for f in files],
                "parameters": {"segmentation": seg_params}}
    with open(args.output / "run_info.json", "w") as fh:
        json.dump(run_info, fh, indent=2)

    seg_dir = args.output / "segmentation"
    seg_dir.mkdir(exist_ok=True)

    print("Segmentation of the pre-bleach frame: " + ", ".join(f"{k}={v}" for k, v in seg_params.items()))
    for f in files:
        stack = tifffile.imread(f)                              # (frames, rows, columns)
        pre = stack[FRAME_PRE].astype(np.float64)
        labels = droplet_segmentation.segment(pre, seg_params)
        tifffile.imwrite(seg_dir / f"{f.stem}_droplet_labels.tif", labels.astype(np.uint16))
        print(f"  {f.stem}: {labels.max()} droplets")

    print(f"Done. Results in {args.output}")


if __name__ == "__main__":
    main()