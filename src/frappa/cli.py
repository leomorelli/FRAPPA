import argparse
from pathlib import Path

import numpy as np
import tifffile

from . import droplet_segmentation


FRAME_PRE = 0 


def main():
    parser = argparse.ArgumentParser(prog="frappa", description=__doc__)
    parser.add_argument("-i", "--input", type=Path, required=True, help="folder with the TIFF stacks")
    parser.add_argument("-o", "--output", type=Path, required=True, help="output folder")
    args = parser.parse_args()

    files = []
    for p in args.input.iterdir():
        if p.suffix.lower() in (".tif", ".tiff"):
            files.append(p)
    files = sorted(files)

    print(f"{len(files)} TIFF file(s) in {args.input}:")

    seg_dir = args.output / "segmentation"
    seg_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        stack = tifffile.imread(f)                              # (frames, rows, columns)
        pre = stack[FRAME_PRE].astype(np.float64)
        labels = droplet_segmentation.segment(pre)
        tifffile.imwrite(seg_dir / f"{f.stem}_droplet_labels.tif", labels.astype(np.uint16))
        print(f"  {f.stem}: {labels.max()} droplets")

    print(f"Done. Results in {args.output}")


if __name__ == "__main__":
    main()