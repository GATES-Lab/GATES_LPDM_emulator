# How-To: Generate and configure `config.yml`

## Overview
This guide explains how to generate, set up and use the `config.yml` file for the GATES LPDM emulator.

`config.yml` lives in the repository root and centralises data paths, domain mappings, and known problematic files, removing the need for hard‑coded values scattered throughout the codebase. It is generated once for your machine (from a set of platform defaults) and then edited to point at your data.

---
## Getting Started

### Step 1 — Generate `config.yml`
From the repository root, generate a `config.yml` with default paths for **local** usage:
```bash
python gates/config.py
```

GATES has been developed on several HPC platforms. To start from the path set for one of them, pass it with `--platform`:
```bash
python gates/config.py --platform bp
```

Available platforms: `local` (default), `bp` (University of Bristol BluePebble), `oracle`, and `isambard-ai`. The names `bluepebble` and `isambard-ai` are accepted as aliases for `bp` and `isambard_ai`. Only the `data_paths` section differs between platforms — `domains` and `bad_fp_files` are shared.

Once generated, open `config.yml` and edit the paths to match your system.

### Step 2 — Load the config
When executing scripts, the repository config is imported with:
```python
from gates.config import get_config
cfg = get_config()
print(cfg.fp_datadir)          # fully-resolved Path
print(cfg.domains["BRAZIL"])   # domain mapping dict
```
`get_config()` builds the `Config` object once and caches it, so repeated calls don't re-read the file. The instance is **read-only** after initialisation — to change a setting, edit `config.yml` and restart, rather than mutating `cfg` in code.

---
## Configuration Reference

The file has four **required** top-level sections — `data_paths`, `user_paths`, `domains`, and `bad_fp_files`. Loading raises an error if any are missing, or if the file isn't valid YAML.

### `data_paths`
Every data path is built by joining each entry onto `base_data_path`, so `base_data_path` is the single place to point at your data root. See the [data pipeline guide](HOW_TO_DATA.md) for the resulting on-disk directory structure.

| Key | Points to | Notes |
|-----|-----------|-------|
| `base_data_path` | data root | prepended to every path below |
| `fp_datadir` | footprint directory | organised into `<DOMAIN>/` subfolders |
| `met_datadir` | meteorology directory | one Zarr store per year |
| `topog_datadir` | topography file | single global NetCDF |
| `landcover_datadir` | land cover file | single global NetCDF; **optional** (may be `null`) |
| `flux_datadir` | flux / emissions directory | used for mole-fraction evaluation |

These are exposed as attributes on the loaded config, already joined onto `base_data_path` — e.g. `cfg.fp_datadir` returns the full `Path`. Leading slashes on the sub-paths are stripped before joining, so the result is correct regardless of whether an entry starts with `/`. Any *additional* keys placed under `data_paths` are still readable via `cfg.data_paths["..."]`, but are **not** automatically joined onto `base_data_path`.

### `user_paths`
Paths specific to your working setup, used verbatim (not joined onto `base_data_path`):

| Key | Meaning |
|-----|---------|
| `save_models_dir` | where trained models are written |
| `parameter_files_dir` | where training parameter JSONs live |

### `domains`
Maps a **region** (e.g. `BRAZIL`) to the **domain** it sits in, plus reference metadata. Each region entry looks like:
```yaml
BRAZIL:
  domain_name: "SOUTHAMERICA"                              # domain the region belongs to
  footprint: "GOSAT-BRAZIL-column_SOUTHAMERICA_201801.nc"  # example footprint file
  flux_suffix:                                             # optional, per-species
    ch4: "_SWAMPS-v32-5_Saunois-Annual-Mean"
```

- **`domain_name`** — the domain the region belongs to. This drives the `<DOMAIN>/` subfolder and the filename token used when resolving footprint and meteorology paths.
- **`footprint`** — an example footprint filename for the region, kept for reference.
- **`flux_suffix`** *(optional)* — maps a species (e.g. `ch4`) to the suffix appended to its flux filename. An empty string means no suffix.

A region not listed here must be passed with an explicit `domain` argument to `LoadSquareSatelliteData`.

### `bad_fp_files`
A list of footprint filenames known to be malformed. `load_fps` detects these and loads them with a workaround instead of failing:
```yaml
bad_fp_files:
- "GOSAT-BRAZIL-column_SOUTHAMERICA_201511.nc"
- "GOSAT-SAHARA-column_NORTHAFRICA_201409.nc"
```

---
## Troubleshooting
- **Invalid YAML** — keep indentation consistent (spaces, not tabs) after editing.
- **Missing sections** — loading raises if any of `data_paths`, `user_paths`, `domains`, or `bad_fp_files` is absent.
- **`Config is read-only`** — you tried to set an attribute after the config was loaded; edit `config.yml` and restart instead.
- **File not found** — run `python gates/config.py` from the repository root to (re)generate the file.

## See Also
- [Data pipeline guide](HOW_TO_DATA.md) — how these paths resolve to files on disk.
- [README.md](../README.md)
