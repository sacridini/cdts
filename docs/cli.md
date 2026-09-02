# Command Line Interface (CLI)

The `cdts` package includes a powerful Command Line Interface (CLI) that allows you to run our core change detection algorithms directly from your terminal. This is especially useful for processing large GeoTIFF stacks in automated bash scripts or HPC environments without writing Python code.

## Basic Usage

The CLI is structured around subcommands for each algorithm. You can invoke the CLI using the `cdts` command (if installed via pip) or by executing the python module directly:

```bash
cdts <algorithm> [OPTIONS] <input_file> <output_directory>
```
*(Alternatively: `python -m cdts.cli <algorithm> ...`)*

To get general help or see the list of available commands:
```bash
cdts --help
```

---

## `landtrendr`

The `landtrendr` command processes a multi-band GeoTIFF representing a time series of a single spectral index (e.g., NBR, NDVI) and extracts change events.

### Syntax
```bash
cdts landtrendr <input> <output_dir> [OPTIONS]
```

### Positional Arguments
* **`input`**: Path to the input multi-band GeoTIFF. Each band should represent a single year or time step in chronological order.
* **`output_dir`**: Directory where the resulting maps (e.g., Year of Detection, Magnitude, Duration) will be saved.

### Options
* `--start-year` (int): The calendar year corresponding to the first band in your stack. Default: `2000`.
* `--max-segments` (int): The maximum number of segments the algorithm can fit per pixel. Default: `6`.
* `--jobs` (int): Number of CPU cores to use. Use `-1` for all available cores. Default: `-1`.
* `--chunk-size` (int): Size of the image chunks (in pixels) processed simultaneously to manage memory. Default: `512`.
* `--save-vertices` (flag): If provided, saves the raw multi-band GeoTIFF containing all fitted vertices, which can be useful for manual inspection.

#### Event Extraction Options
These options control how the specific change event is extracted from the temporal trajectory:
* `--event-type`: The type of event to map. Choices: `loss` (value decreases) or `gain` (value increases). Default: `loss`.
* `--sort-by`: How to select the event if multiple events occur. Choices: `greatest` (highest magnitude), `newest` (most recent), `fastest` (highest rate), `longest` (longest duration). Default: `greatest`.
* `--min-mag` (float): Filter out events with a magnitude lower than this threshold. Default: `0.0`.
* `--min-dur` (int): Filter out events shorter than this duration in years. Default: `1`.
* `--pre-val-thresh` (float): Filter out events if the starting value of the pixel was already below this threshold (useful to avoid detecting "loss" in already deforested areas). Default: `0.0`.
* `--prefix` (str): Prefix added to all output file names. Default: `lt_event`.
* `--output-scale` (float): A scale factor applied to the output values. Useful for converting scaled integer data back to floating-point index values (e.g., `0.0001`). Default: `1.0`.

### Example
Run LandTrendr on a 30-year NBR stack, extracting the greatest vegetation loss, using all CPU cores, and converting the output back to floating point (assuming NBR was scaled by 10000):
```bash
cdts landtrendr ./data/nbr_stack_1990_2020.tif ./results \
    --start-year 1990 \
    --max-segments 6 \
    --event-type loss \
    --sort-by greatest \
    --output-scale 0.0001 \
    --jobs -1
```

---

## `ccdc`

The `ccdc` command runs the Continuous Change Detection and Classification algorithm (or its variant, COLD) on a highly dense, multi-band, and multi-date GeoTIFF stack.

### Syntax
```bash
cdts ccdc <input> <output_dir> [OPTIONS]
```

### Positional Arguments
* **`input`**: Path to the input stacked GeoTIFF. The bands must be interleaved by date (e.g., Date1-Band1, Date1-Band2, Date2-Band1, etc.).
* **`output_dir`**: Directory where the resulting harmonic coefficients and structural break dates will be saved.

### Options
* `--num-bands` (int): The number of spectral bands provided per observation date. Default: `6`.
* `--qa-band` (int): The zero-based index of the Quality Assessment (QA) band within the block of bands for a single date. For example, if you have 6 bands and the 7th is QA, set this to 6 (since `--num-bands` would be 7). Default: `-1` (no QA masking).
* `--dates-file` (str): Path to a plain text file containing the observation dates (one integer per line, usually in Julian days or ordinal dates). If omitted, the tool assumes a regular 16-day interval (like Landsat) and prints a warning.
* `--max-segments` (int): The maximum number of distinct change segments to retain per pixel. Default: `6`.
* `--chunk-size` (int): Size of the image chunks to process simultaneously. Default: `512`.
* `--jobs` (int): Number of CPU cores to use. Use `-1` for all available cores. Default: `-1`.
* `--cold` (flag): Switches the internal logic from standard CCDC to the **COLD** (Continuous monitoring of Land Disturbance) algorithm. This increases the required consecutive anomalies for a break from 3 to 6.
* `--prefix` (str): Prefix added to the output GeoTIFF files. Default: `ccdc`.

### Example
Run CCDC on an image with 6 spectral bands and 1 QA band (7 total bands per date). The 7th band (index 6) is the QA mask. The dates are provided in `dates.txt`:
```bash
cdts ccdc ./data/dense_stack.tif ./results \
    --num-bands 7 \
    --qa-band 6 \
    --dates-file ./data/dates.txt \
    --max-segments 8 \
    --jobs -1
```

Run the COLD algorithm (more conservative, requires 6 anomalies) on the same dataset:
```bash
cdts ccdc ./data/dense_stack.tif ./results_cold \
    --num-bands 7 \
    --qa-band 6 \
    --dates-file ./data/dates.txt \
    --cold \
    --jobs -1
```
