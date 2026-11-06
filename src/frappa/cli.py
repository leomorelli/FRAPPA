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
from . import bleach_detection, droplet_segmentation, recovery_curves, saturation_report
from .common import save_figure


FRAME_PRE = 0 
FRAME_POST = 1    # index of the first post-bleach frame


def build_parser():
    parser = argparse.ArgumentParser(prog="frappa", description=__doc__)
    parser.add_argument("-i", "--input", type=Path, nargs="+", required=True,
                        help="TIFF stacks: one or more files and/or folders (with the .tif/.tiff files in it)")
    parser.add_argument("-o", "--output", type=Path, required=True, help="output folder")
    parser.add_argument("--exclude", type=Path,
                        help="CSV file with the droplets to exclude. Columns: file name, droplet, and optionally the reason")

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

    sr = parser.add_argument_group("saturation report parameters")
    sr.add_argument("--saturation-report", action="store_true", help="write the saturation report: tables per file, summary and figures")
    sr.add_argument("--saturation-threshold", type=float, default=0.75,
                    help="report droplets and spots whose saturated fraction is above this (default: %(default)s)")

    return parser

def find_files(inputs):
    """TIFF stacks to analyse: all .tif/.tiff files in the given folders and/or the given files."""
    files = []
    for path in inputs:
        if path.is_dir():
            files += [p for p in path.iterdir() if p.suffix.lower() in (".tif", ".tiff")]
        elif path.is_file():
            files.append(path)
        else:
            sys.exit(f"Not found: {path}")
    if not files:
        sys.exit("No TIFF files found")
    return sorted(files)

def read_exclusions(path):
    """Droplets to exclude, from a CSV file with columns file, droplet (and optionally reason).
    Without a file (path None), an empty table."""
    if path is None:
        return pd.DataFrame(columns=["file", "droplet", "reason"])
    if not path.is_file():
        sys.exit(f"Exclusion file not found: {path}")
    exclusions = pd.read_csv(path, skipinitialspace=True)
    missing = {"file", "droplet"} - set(exclusions.columns)    # required columns
    if missing:
        sys.exit(f"{path}: missing column(s) {', '.join(sorted(missing))}. "
                 f"The first line must be the header: file,droplet,reason")
    if "reason" not in exclusions.columns:
        exclusions["reason"] = ""
    return exclusions

def main():
    args = build_parser().parse_args()
    seg_params = {"sigma": args.sigma, "kernel_size": args.kernel_size, "clip_limit": args.clip_limit,
                  "min_droplet_area": args.min_droplet_area, "min_peak_distance": args.min_peak_distance}
    bleach_params = {"bl_sigma": args.bl_sigma, "territory": args.territory, "gain_frac": args.gain_frac, "max_shift": args.max_shift,
                     "min_peak_drop": args.min_peak_drop, "spot_fraction": args.spot_fraction}
    fig_opts = {"formats": args.figure_format, "dpi": args.dpi}

    files = find_files(args.input)
    print(f"{len(files)} file(s): {', '.join(f.name for f in files)}")
    exclusions = read_exclusions(args.exclude)
    unknown = set(exclusions["file"]) - {f.stem for f in files}
    if unknown:
        print(f"Warning: exclusion file lists files that are not in the input: {', '.join(sorted(unknown))}")
    if len(exclusions):
        print(f"Excluding {len(exclusions)} droplet(s) listed in {args.exclude}")

    # record how this run was made
    args.output.mkdir(parents=True, exist_ok=True)
    run_info = {"frappa_version": __version__,
                "date": datetime.now().isoformat(timespec="seconds"),
                "command": "frappa " + " ".join(sys.argv[1:]),
                "input_files": [str(f) for f in files],
                "figures": {"all_plots": args.all_plots, **fig_opts},
                "parameters": {"segmentation": seg_params,
                               "bleach_detection": bleach_params},
                "saturation report": {"enabled": args.saturation_report, 
                                      "threshold": args.saturation_threshold},
                "exclusion_file": str(args.exclude) if args.exclude else None,                      
                "exclusions": exclusions.to_dict(orient="records")}
    with open(args.output / "run_info.json", "w") as fh:
        json.dump(run_info, fh, indent=2)

    seg_dir = args.output / "segmentation"
    bleach_dir = args.output / "bleach_detection"
    curves_dir = args.output / "recovery_curves"
    for folder in (seg_dir, bleach_dir, curves_dir):
        folder.mkdir(exist_ok=True)
    if args.saturation_report:
        saturation_dir = args.output / "saturation_report"
        saturation_dir.mkdir(exist_ok=True)

    print("Segmentation of the pre-bleach frame: " + ", ".join(f"{k}={v}" for k, v in seg_params.items()))
    print("Bleach detection: " + ", ".join(f"{k}={v}" for k, v in bleach_params.items()))
    if args.saturation_report:
        print(f"Saturation report enabled: threshold={args.saturation_threshold}")

    all_curves = []
    all_tables = {}
    saturation_counts = []
    all_sat_tables = {}
    all_per_frame = {}

    for f in files:
        name = f.stem
        raw = tifffile.imread(f)                 # (frames, rows, columns)
        sat_value = int(np.iinfo(raw.dtype).max)         # 255
        stack = raw.astype(np.float64) 
        pre, post = stack[FRAME_PRE], stack[FRAME_POST]

        ## 1. droplets in the pre-bleach frame ##
        steps = droplet_segmentation.segment(pre, seg_params)
        labels = steps["final"]
        tifffile.imwrite(seg_dir / f"{name}_droplet_labels.tif", labels.astype(np.uint16))
        if args.all_plots:
            save_figure(droplet_segmentation.summary_figure(name, steps, seg_params),
                        seg_dir / "figures", f"{name}_segmentation_summary", **fig_opts)

        ## 2. bleached droplets and spots ##
        tab, spots, spot_tab, gained = bleach_detection.detect(pre, post, labels, bleach_params)
        
        # manual curation, mark droplets excluded by hand
        ex = exclusions[exclusions["file"] == name]
        tab["excluded"] = tab["droplet"].isin(ex["droplet"])
        tab["exclusion_reason"] = tab["droplet"].map(ex.set_index("droplet")["reason"])
        not_found = set(ex["droplet"]) - set(tab["droplet"])
        if not_found:
            print(f"  Warning: {name}: excluded droplet(s) not found: {sorted(not_found)}")

        all_tables[name] = tab
        tifffile.imwrite(bleach_dir / f"{name}_spot_labels.tif", spots.astype(np.uint16))
        tab.to_csv(bleach_dir / f"{name}_all_droplets.csv", index=False)
        spot_tab.to_csv(bleach_dir / f"{name}_bleached_droplets.csv", index=False)
        save_figure(bleach_detection.overview_figure(name, pre, post, labels, spots, tab),
                    bleach_dir / "figures", f"{name}_bleach_overview", **fig_opts)
        if args.all_plots and tab["moved"].any():
            save_figure(bleach_detection.moved_figure(name, pre, post, labels, gained, tab),
                        bleach_dir / "figures", f"{name}_moved_droplets", **fig_opts)
        if args.all_plots and tab["bleached"].any():
            save_figure(bleach_detection.spots_figure(name, pre, post, labels, spots, tab),
                        bleach_dir / "figures", f"{name}_bleached_spots", **fig_opts)

        # saturation report (optional)
        dashed = set()                     # droplets drawn dashed in the curve figures
        if args.saturation_report:
            sat_table, per_frame, counts = saturation_report.report(stack, sat_value, labels, spots, tab,
                                                                    FRAME_PRE, args.saturation_threshold)
            dashed = set(sat_table.loc[sat_table["spot_pre_above"], "droplet"])
            sat_table.to_csv(saturation_dir / f"{name}_saturation.csv", index=False)
            per_frame["droplet"].to_csv(saturation_dir / f"{name}_saturation_per_frame.csv",
                                        index_label="droplet")
            per_frame["spot"].to_csv(saturation_dir / f"{name}_spot_saturation_per_frame.csv",
                                     index_label="droplet")
            saturation_counts.append({"file": name, **counts})
            all_sat_tables[name] = sat_table
            all_per_frame[name] = per_frame["droplet"]
            if counts["bleached"]:
                save_figure(saturation_report.gallery_figure(name, stack, sat_value, labels, spots,
                                                             sat_table, per_frame, FRAME_PRE,
                                                             args.saturation_threshold),
                            saturation_dir / "figures", f"{name}_saturation_gallery", **fig_opts)

        ## 3. recovery curves of the bleached droplets ##
        curves = recovery_curves.measure(stack, labels, spots, tab, FRAME_PRE)
        curves.insert(0, "file", name)
        curves.to_csv(curves_dir / f"{name}_recovery_curves.csv", index=False)
        if not curves.empty:
            save_figure(recovery_curves.curves_figure(curves, dashed),
                        curves_dir / "figures", f"{name}_recovery_curves", **fig_opts)
            if args.all_plots:
                save_figure(recovery_curves.steps_figure(curves, FRAME_PRE, dashed),
                        curves_dir / "figures", f"{name}_normalization_steps", **fig_opts)
        all_curves.append(curves)
        
        # print message
        moved = tab["moved"]
        excluded = tab["excluded"] & ~moved              # a moved droplet is counted as moved
        with_curve = tab["bleached"] & ~tab["excluded"]
        references = ~moved & ~tab["bleached"] & ~tab["excluded"]
        print(f"  {name}: {len(tab)} droplets\n"
              f"    analysed: {int(with_curve.sum())} bleached (recovery curves), {int(references.sum())} references\n"
              f"    left out: {int(moved.sum())} moved, {int(excluded.sum())} excluded by hand")

        if args.saturation_report:
            print(saturation_report.summary_text(counts, args.saturation_threshold))
    
    if args.all_plots:
        save_figure(bleach_detection.peak_drop_figure(all_tables, args.min_peak_drop),    # peak drop, one panel per file
                    bleach_dir / "figures", "peak_drop", **fig_opts)
    all_curves = pd.concat(all_curves, ignore_index=True)
    all_curves.to_csv(curves_dir / "recovery_curves.csv", index=False)
    if not all_curves.empty:
        save_figure(recovery_curves.all_files_curves_figure(all_curves),        # recovery curves, one panel per file
                    curves_dir / "figures", "mean_final_curves", **fig_opts)
    if args.saturation_report:
        pd.DataFrame(saturation_counts).to_csv(saturation_dir / "saturation_summary.csv", index=False)
        save_figure(saturation_report.over_time_figure(all_sat_tables, all_per_frame, 
                                                       args.saturation_threshold),
                    saturation_dir / "figures", "saturation_over_time", **fig_opts)

    print(f"Done. Results in {args.output}")


if __name__ == "__main__":
    main()