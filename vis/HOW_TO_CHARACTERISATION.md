# How to run characterisation plots

`characterisation_run.py` loads a single data object and saves a batch of characterisation
plots as PNG files into an output folder. It wraps the plotting functions in `vis_plotting/`.

---

## Requirements

Activate the project environment before running:

```bash
conda activate gates_env
```

Run from the repository base directory (`graphnet_LPDM_emulator/`):

```bash
cd graphnet_LPDM_emulator
```

---

## Basic usage

```bash
python vis/characterisation_run.py --date 201602 --region BRAZIL
```

This loads footprints, meteorology, topography and land cover for Brazil, February 2016,
and saves all plots to:

```
vis/outputs/characterisation_201602_southamerica_brazil_<timestamp>/
```

---

## All options

| Flag | Default | Description |
|---|---|---|
| `--date` | *(required)* | Date/pattern passed to the loader, e.g. `201602`, `2016`, `2016*`, `201[4-6]*` |
| `--region` | `BRAZIL` | Region name, e.g. `BRAZIL`, `SAHARA`, `INDIA` |
| `--domain` | auto | Override the UM domain (normally inferred from region) |
| `--list-regions` | off | Print available built-in regions and their default domains, then exit |
| `--out-dir` | auto timestamped | Custom output folder for PNGs |
| `--countries` | none | Country or countries for topography/land-use plots — comma-separated (`BRAZIL,ARGENTINA`) or JSON (`["BRAZIL","ARGENTINA"]`). If omitted, no country mask is applied. |
| `--no-align-domains` | off | Skip cropping met/fp to their intersection |
| `--no-domain` | off | Skip domain boundary map plots |
| `--no-footprint` | off | Skip seasonal footprint histogram plots |
| `--no-topography` | off | Skip topography and land-use plots |
| `--no-windrose` | off | Skip wind rose plot |
| `--no-metrics` | off | Skip wind-speed metrics text file output |
| `--quiet` | off | Suppress data loader progress messages |

---

## Examples

### Sahara, single month, custom output folder
```bash
python vis/characterisation_run.py --date 201603 --region SAHARA --out-dir ~/plots/sahara_mar2016
```

### Brazil, full year, multiple countries for land-use plots
```bash
python vis/characterisation_run.py --date 2016 --region BRAZIL --countries BRAZIL,ARGENTINA,PERU
```

### Multi-year run in one go (all together)
```bash
python vis/characterisation_run.py --date "201[4-6]*" --region SOUTHAMERICA
```

### Show available region names and default domains
```bash
python vis/characterisation_run.py --list-regions
```

### Domain and footprint plots only (skip topography and met)
```bash
python vis/characterisation_run.py --date 201602 --region BRAZIL --no-topography --no-windrose
```

### Optional: run via SLURM launch script
From the repository base directory:

```bash
cd graphnet_LPDM_emulator
sbatch launch_characterise.sh --date 201602 --region SAHARA
```

Multi-year example:

```bash
sbatch launch_characterise.sh --date "201[4-6]*" --region SOUTHAMERICA --countries BRAZIL,PERU
```

List regions:

```bash
sbatch launch_characterise.sh --list-regions
```

The SLURM script is optional. It forwards arguments directly to
`vis/characterisation_run.py`, so CLI flags are the same as local runs.

### Calling from Python (e.g. in a notebook)
```python
import sys
sys.path.insert(0, "..")  # repo root
sys.path.insert(0, ".")   # vis/

from pathlib import Path
from characterisation_run import run_characterisation_batch

saved = run_characterisation_batch(
    date="201602",
    region="BRAZIL",
    out_dir=Path("outputs/my_run"),
    countries=["BRAZIL", "ARGENTINA"],
    include_windrose=False,
)

print("Saved:", saved)
```

---

## Regions and countries

- Region names are case-insensitive (`sahara`, `Sahara`, `SAHARA` all work).
- Built-in regions currently map as follows:
    - `BRAZIL` -> `SOUTHAMERICA`
    - `SOUTHAMERICA` -> `SOUTHAMERICA`
    - `SAHARA` -> `NORTHAFRICA`
    - `INDIA` -> `SOUTHASIA`
- If you need another region naming scheme, pass `--domain` explicitly.
- During each run, the script now prints the full list of available countries
    for the loaded region/domain so you can pick values for `--countries`.

---

## Output files

All files are saved as PNGs. The filename encodes the plot type and country/scope.

| Filename | Description |
|---|---|
| `domain_met.png` | Met domain boundary on a global map |
| `domain_fp.png` | Footprint domain boundary on a global map |
| `domain_topo.png` | Topography domain boundary on a global map |
| `domain_overlay_met_fp_topo.png` | All three domains overlaid |
| `footprint_seasonal_count.png` | Seasonal map of footprint observation counts |
| `footprint_seasonal_mean_sum.png` | Seasonal map of mean footprint sums |
| `topography_map_global.png` | Discrete elevation map (global, when no country specified) |
| `topography_map_<domain>_<country>.png` | Discrete elevation map for the country (includes domain) |
| `topography_histogram_global.png` | Histogram of land surface altitude (global) |
| `topography_histogram_<domain>_<country>.png` | Histogram of land surface altitude for the country |
| `landuse_frequency_global.png` | Bar chart of land-use class frequencies (global) |
| `landuse_frequency_<domain>_<country>.png` | Bar chart of land-use class frequencies for the country |
| `landuse_majority_map_global.png` | Spatial map of majority land-use class (global) |
| `landuse_majority_map_<domain>_<country>.png` | Spatial map of majority land-use class for the country |
| `met_windrose_level_1.png` | Wind rose at the median release point, level 1 |
| `wind_speed_metrics.txt` | Domain-wide and seasonal wind-speed summary stats (mean, std, etc.) |

---

## Adding new plots

1. Add (or import) a plotting function in the appropriate `vis_plotting/` module.
2. Add a `_run_<group>_plots(data, out_dir, ...)` helper function in `characterisation_run.py`
   following the pattern of the existing ones — call `_save_fig(fig, out_dir, "filename_stem")`
   for each figure and return the list of saved paths.
3. Call the new helper inside `run_characterisation_batch()`.
4. Optionally add a `--no-<group>` CLI flag in `_build_arg_parser()` and wire it up in `main()`.
