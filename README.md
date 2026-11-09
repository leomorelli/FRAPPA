# FRAPPA

**F**luorescence **R**ecovery **A**fter **P**hotobleaching **P**rocessing and **A**nalysis:
from the image stacks of droplets  to normalized recovery curves.

For each stack, the pipeline:

1. **segments the droplets** in the pre-bleach frame;
2. **detects the bleached droplets and their bleached spot**, comparing the pre-bleach and the first post-bleach frame;
3. **computes the recovery curves** of the bleached droplets, normalized to the whole droplet.

Optional: **saturation report** (`--saturation-report`), **diagnostic figures** for every step
(`--all-plots`), and **manual curation** to exclude droplets by hand (`--exclude`).

The analysis was first developed in `notebooks/`, where the default parameters were chosen.

## Installation

Requires [conda](https://docs.conda.io). From the repository folder:

```bash
git clone https://github.com/leomorelli/FRAPPA.git
cd FRAPPA
conda env create -f environment.yml
conda activate frappa
frappa --help
```

## Usage

The input is one multi-frame TIFF per acquisition (frames × rows × columns). The first frame is the pre-bleach frame, the second the first post-bleach frame.

```bash
frappa -i data/ -o results/                                            # all TIFFs in a folder
frappa -i data/PEG_3.tif data/Apt2_1.tif -o results_two/               # selected files
frappa -i data/ -o results_full/ --saturation-report --all-plots       # full output
frappa -i data/ -o results_cl005/ --clip-limit 0.005                   # a different parameter
```

### Options

**Input and output**

| option | meaning |
|---|---|
| `-i, --input` | TIFF stacks: one or more files and/or folders (a folder means all its `.tif`/`.tiff` files) |
| `-o, --output` | output folder |
| `--exclude` | CSV file with droplets to exclude by hand (see [Manual curation](#manual-curation)) |

**Segmentation** (pre-bleach frame)

| option | default | meaning |
|---|---|---|
| `--sigma` | 0.5 | Gaussian smoothing before CLAHE (pixels) |
| `--kernel-size` | 64 | CLAHE kernel size (pixels) |
| `--clip-limit` | 0.01 | CLAHE clip limit |
| `--min-droplet-area` | 150 | smaller objects are removed (pixels) |
| `--min-peak-distance` | 6 | minimum distance between watershed seeds (pixels); lower = more splitting |

**Bleach detection** (pre-bleach vs first post-bleach frame)

| option | default | meaning |
|---|---|---|
| `--bl-sigma` | 1.0 | Gaussian smoothing of the two frames (pixels) |
| `--territory` | 5 | ring around each droplet checked for gained signal (pixels; used for motion detection) |
| `--gain-frac` | 0.25 | a ring pixel counts as gained if it brightened by more than this fraction of the droplet intensity (used for motion detection) |
| `--max-shift` | 1.0 | droplets with a larger estimated shift (pixels) count as moved and are left out (used for motion detection) |
| `--min-peak-drop` | 0.4 | a droplet is bleached if its peak relative drop is at least this |
| `--spot-fraction` | 0.75 | consider in the bleached spot pixels whose drop is at least this fraction of the droplet’s peak drop; lower values give larger spots (e.g. 0.5 = half maximum)
|

**Saturation report**

| option | default | meaning |
|---|---|---|
| `--saturation-report` | off | write the saturation report (tables, summary, figures) |
| `--saturation-threshold` | 0.75 | droplets and spots with a larger saturated fraction are reported |

**Figures**

| option | default | meaning |
|---|---|---|
| `--all-plots` | off | also save the diagnostic figures of every step |
| `--figure-format` | png | one or more of `png`, `pdf`, `svg` |
| `--dpi` | 150 | resolution of the figures |

## Outputs

```
results/
├── run_info.json         version, date, command, input files, exclusions, all parameters
├── segmentation/
├── bleach_detection/
├── saturation_report/    (with --saturation-report)
└── recovery_curves/
```

Each step folder has a `figures/` sub-folder. Figures marked *all-plots* are saved only with
`--all-plots`.

**`segmentation/`**

- `<file>_droplet_labels.tif`: one integer ID per droplet (0 = background)
- figures: `<file>_segmentation_summary` (every step of the segmentation) *all-plots*

**`bleach_detection/`**

- `<file>_spot_labels.tif`: the bleached spots, each with the ID of its droplet
- `<file>_all_droplets.csv`: every droplet: area, peak and mean relative drop, estimated shift,
  `moved`, `bleached`, `excluded`, `exclusion_reason`
- `<file>_bleached_droplets.csv`: the bleached droplets: spot area, bleach depth, spot position
- figures: `<file>_bleach_overview` (droplets, bleached spots with IDs, moved and excluded droplets); `<file>_moved_droplets`, `<file>_bleached_spots`, `peak_drop` *all-plots*

**`saturation_report/`** (with `--saturation-report`)

- `<file>_saturation.csv`: per droplet, the saturated fraction before the bleach, the worst after
  the bleach and its frame, for the whole droplet and the spot; flags above the threshold
- `<file>_saturation_per_frame.csv`, `<file>_spot_saturation_per_frame.csv`: saturated fraction in every frame per droplet and per spot
- `saturation_summary.csv`: counts per file
- figures: `<file>_saturation_gallery` (bleached droplets, saturated pixels in red), `saturation_over_time`

**`recovery_curves/`**

- `<file>_recovery_curves.csv` and `recovery_curves.csv` (all files): one row per file, droplet
  and frame, with `spot`, `tot`, `ref`, `A`, `B`, `C`, `D`, `final`
- figures: `<file>_recovery_curves` (each droplet, mean ± SD), `recovery_curves_all_files`;
  `<file>_normalization_steps` *all-plots*

## Manual curation

Some droplets can't be analysed reliably, for example when overlapping droplets are segmented as one. They can be excluded by hand:

1. Run the pipeline, ideally with `--all-plots`, and look at the bleach overviews and the bleached-spots galleries.
2. Write the IDs of the droplets to exclude in a CSV file. The first line must be the header; `reason` is optional:

   ```
   file,droplet,reason
   Apt2+Ca_2,94,wrong segmentation
   ```

3. Run again **with the same input files and segmentation parameters** (to ensure that the droplets IDs are always the same), into a new output folder:

   ```bash
   frappa -i data/ -o results_curated/ --exclude exclusions.csv
   ```

Excluded bleached droplets get no recovery curve and excluded references are left out of the reference pool. The bleach detection overview figure outlines excluded droplets in red. The saturation report is not affected. 

## Method notes

- **Normalization.** spot = mean intensity of the bleached spot, tot = whole bleached droplet,
  ref = all pixels of the reference droplets, per frame:

  `A(t) = spot(t)/ref(t))`

  `B(t) = A(t)/A(pre)`
  
  `C(t) = tot(t)/ref(t)`
  
  `D(t) = C(t)/C(pre)`
  
  `final(t) = B(t)/D(t)`.

  The reference cancels in `final`, which equals `[spot(t)/tot(t)] / [spot(pre)/tot(pre)]`

  A, B, C, D and ref are kept in `<file>_recovery_curves` and figure `<file>_normalization_steps` for inspection.

- **No background subtraction:** the microscope clips the background at 0.
- **Motion check.** A droplet whose surroundings gain signal between the pre-bleach and the first post-bleach frame likely moved, brightened, or was approached by a neighbour; it is left out of both bleached and reference droplets.
- **Saturation.** Saturated pixels (the maximum value of the data type, 255 for 8-bit images) cannot record intensity changes and flatten the curves. The report flags them; nothing is excluded automatically.

## Reproducibility

Every run writes `run_info.json`, with the pipeline version, the date, the full command, the input files, the exclusions and all parameters. 

## Repository layout

```
src/frappa/
├── cli.py                    the frappa command: options, input, the steps in order
├── droplet_segmentation.py   segmentation of the pre-bleach frame
├── bleach_detection.py       motion check, bleached droplets, spots
├── saturation_report.py      saturation report
├── recovery_curves.py        normalization and curves
└── common.py                 helpers shared by the steps
notebooks/                    development notebooks
environment.yml               conda environment
pyproject.toml                package definition and the frappa command
```
