# Command Line Interface (CLI)

The `cdts` package includes a powerful Command Line Interface (CLI) that allows you to run our core change detection algorithms directly from your terminal. This is especially useful for processing large GeoTIFF stacks in automated bash scripts, cron jobs, or High-Performance Computing (HPC) environments without writing any Python code.

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

## 1. LandTrendr (`landtrendr`)

The `landtrendr` command processes a multi-band GeoTIFF representing a time series of a single spectral index (e.g., NBR, NDVI) and extracts spatial change events.

### Syntax
```bash
cdts landtrendr <input> <output_dir> [OPTIONS]
```

### Positional Arguments
| Argument | Type | Description |
| :--- | :---: | :--- |
| **`input`** | `filepath` | Path to the input multi-band GeoTIFF. Each band must represent a single year or time step in chronological order. |
| **`output_dir`** | `dirpath` | Directory where the resulting event maps (Year of Detection, Magnitude, Duration) will be saved. |

### Configuration Options

| Option | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--start-year` | `int` | `2000` | The calendar year corresponding to the first band in your stack. |
| `--max-segments`| `int` | `6` | The maximum number of line segments the algorithm can fit per pixel. |
| `--jobs` | `int` | `-1` | Number of CPU cores to use for parallel processing. `-1` uses all available cores. |
| `--chunk-size` | `int` | `512` | Size of the image chunks (in pixels) processed simultaneously to manage RAM. |
| `--save-vertices`| `flag` | `False` | If provided, saves the raw multi-band GeoTIFF containing all fitted vertices. |

### Event Extraction Options

These options control how specific change events are extracted from the temporal trajectory:

| Option | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--event-type` | `str` | `loss` | Type of event to map. Choices: `loss` (value decreases) or `gain` (value increases). |
| `--sort-by` | `str` | `greatest` | How to select the event if multiple occur. Choices: `greatest`, `newest`, `fastest`, `longest`. |
| `--min-mag` | `float`| `0.0` | Filter out events with a magnitude lower than this threshold. |
| `--min-dur` | `int` | `1` | Filter out events shorter than this duration in years. |
| `--pre-val-thresh`| `float`| `0.0` | Filter out events if the starting value was already below this threshold. |
| `--output-scale` | `float`| `1.0` | Scale factor applied to the output. Useful for converting scaled integers back to floats (e.g., `0.0001`). |
| `--prefix` | `str` | `lt_event` | Prefix added to all output file names. |

### End-to-End Example
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

## 2. Continuous Change Detection (`ccdc`)

The `ccdc` command runs the Continuous Change Detection and Classification algorithm (or its more conservative variant, COLD) on a highly dense, multi-band, and multi-date GeoTIFF stack.

### Syntax
```bash
cdts ccdc <input> <output_dir> [OPTIONS]
```

### Positional Arguments
| Argument | Type | Description |
| :--- | :---: | :--- |
| **`input`** | `filepath` | Path to the input stacked GeoTIFF. Bands must be interleaved by date (e.g., Date1-Band1, Date1-Band2, Date2-Band1, etc.). |
| **`output_dir`** | `dirpath` | Directory where the resulting harmonic coefficients and structural break dates will be saved. |

### Configuration Options

| Option | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--num-bands` | `int` | `6` | The number of spectral bands provided per observation date. |
| `--qa-band` | `int` | `-1` | The zero-based index of the Quality Assessment (QA) mask band within the block of bands for a single date. `-1` disables QA masking. |
| `--dates-file` | `str` | *None* | Path to a plain text file containing the observation dates (one integer per line, usually in Julian days). If omitted, assumes a 16-day Landsat interval. |
| `--max-segments`| `int` | `6` | The maximum number of distinct change segments to retain per pixel. |
| `--chunk-size` | `int` | `512` | Size of the image chunks to process simultaneously. |
| `--jobs` | `int` | `-1` | Number of CPU cores to use. `-1` uses all available cores. |
| `--cold` | `flag` | `False` | Switches logic to the **COLD** algorithm variant, increasing the required consecutive anomalies for a break from 3 to 6. |
| `--prefix` | `str` | `ccdc` | Prefix added to all output GeoTIFF files. |

### End-to-End Example
Run CCDC on an image with 6 spectral bands and 1 QA band (7 total bands per date). The 7th band (index 6) is the QA mask. The dates are provided in a text file:
```bash
cdts ccdc ./data/dense_stack.tif ./results \
    --num-bands 7 \
    --qa-band 6 \
    --dates-file ./data/dates.txt \
    --max-segments 8 \
    --jobs -1
```

Run the COLD algorithm on the exact same dataset:
```bash
cdts ccdc ./data/dense_stack.tif ./results_cold \
    --num-bands 7 \
    --qa-band 6 \
    --dates-file ./data/dates.txt \
    --cold \
    --jobs -1
```

---

## Note on AI Tools (Deep Learning)

Currently, the AI tools (`cdts.ai`) are **not** exposed via the CLI. 

**Why?** 
Deep learning architectures (like UTAE, TempCNN, or Siamese Networks) require highly specific initializations based on your dataset (e.g., number of input bands, number of target classes, path to pre-trained `.pth` weights, and GPU allocation strategies). These configurations are too complex and dynamic to be safely passed as simple terminal arguments.

To use the AI tools, please utilize the [Python API](tutorials/ai.md) which allows full flexibility in defining PyTorch DataLoaders, Loss Functions, and Training Loops.
