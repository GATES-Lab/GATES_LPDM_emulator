# How-To: Data Pipeline

This guide covers how to load, process, and prepare data for the LPDM emulator using `load_data.py` and `datasets.py`.

The pipeline takes LPDM footprints + meteorology + static fields, cuts them to squares around each satellite measurement release point, scales them, and produces a PyTorch DataLoader for training.

---

## Required Data

| Data | Format | Required dimensions / variables |
|------|--------|--------------------------------|
| **Footprints** | NetCDF `.nc`, one file per month | dims: `time`, `lat`, `lon`; vars: `fp`, `release_lat`, `release_lon` |
| **Meteorology** | Zarr, one store per year | dims: `time`, `levels`, `lat`, `lon` |
| **Topography** | Single global NetCDF | dims: `lat`, `lon` |
| **Land cover** | Single global NetCDF | dims: `lat`, `lon`, `pseudo_level` |

### Where the data lives: `config.yml` paths

The loader builds every data path from the `data_paths` section of `config.yml` (see the [config guide](HOW_TO_CONFIG.md)). Each entry is joined onto `base_data_path`, so `base_data_path` is the single place to point at your data root:

```yaml
data_paths:
  base_data_path:    /group/chem/acrg          # prepended to every path below
  fp_datadir:        /LPDM/fp_NAME_pre20210701/
  met_datadir:       /met_archive/zarr_store/
  topog_datadir:     /LPDM/topog_NAME/TopogUMG_Mk8_global.nc
  landcover_datadir: /LPDM/topog_NAME/land_cover.nc     # optional
```

`fp_datadir` and `met_datadir` are **directories**; `topog_datadir` and `landcover_datadir` point at **single global files**.

### Expected directory structure (default paths)

Inside `fp_datadir` and `met_datadir`, files are organised into a **`<DOMAIN>/` subfolder** and named so the region, domain and date can be parsed out. Meteorology is stored as **one Zarr store per year** — the older monthly NetCDF met files are no longer supported (footprints, topography and land cover remain NetCDF):

```
<base_data_path>/
├── <fp_datadir>/
│   └── <DOMAIN>/
│       └── *<REGION>*<DOMAIN>_<YYYYMM>*.nc     # footprints, one file per month
├── <met_datadir>/
│   └── <DOMAIN>/
│       └── <DOMAIN>_Met_<YYYY>*.zarr           # meteorology, one store per year
├── <topog_datadir>                             # single global NetCDF file
└── <landcover_datadir>                         # single global NetCDF file (optional)
```

For example, loading `region="BRAZIL"` (domain `SOUTHAMERICA`) for January 2018 resolves to:

```
/group/chem/acrg/LPDM/fp_NAME_pre20210701/SOUTHAMERICA/*BRAZIL*SOUTHAMERICA_201801*.nc
/group/chem/acrg/met_archive/zarr_store/SOUTHAMERICA/SOUTHAMERICA_Met_2018*.zarr
```

When no `month` is given, the date part is just `<YYYY>`, so every month of that year matches.

### Overriding the default paths

To load data that doesn't follow this layout, pass a path **prefix** directly and the loader appends the date and extension itself:

- **Footprints** — pass `fp_datadir="/path/to/filename_"`; the loader appends `*<YYYYMM>*.nc`, so files must end in the date (e.g. `filename_201801.nc`).
- **Meteorology** — pass `met_args={"met_datadir": "/path/to/filename_"}`; the loader appends `*<YYYY>*.zarr` (e.g. `filename_2018.zarr`).
- **Topography / land cover** — pass `topog_args={"topog_path": ..., "landcover_path": ...}` to point at specific files.

### Built-in region → domain mappings

`<DOMAIN>` above is the *domain* that a *region* belongs to. These mappings live in the `domains` section of `config.yml`:

| Region | Domain |
|--------|--------|
| `BRAZIL` | `SOUTHAMERICA` |
| `SOUTHAMERICA` | `SOUTHAMERICA` |
| `SAHARA` | `NORTHAFRICA` |
| `INDIA` | `SOUTHASIA` |
| `CHINA` | `EASTASIA` |

Regions not listed in `config.yml` require passing `domain` explicitly to `LoadSquareSatelliteData`.

---

## Pipeline Steps

### Step 1 — Load and cut data: `LoadSquareSatelliteData`

```python
from gates.data.load_data import LoadSquareSatelliteData

data = LoadSquareSatelliteData(
    year=2016,
    region="BRAZIL",
    month="01",   # omit for a full year
    size=10,      # cut to 10×10 square around release point
    freq=3, # load one in every three footprints, sampled regularly along the time axis
)
```

What happens under the hood:
- `load_fps()` opens and concatenates footprint NetCDF files; handles a hardcoded list of known malformed files automatically.
- `_process_footprints()` cuts each footprint to a `size × size` square centred on its `release_lat`/`release_lon` (via `cut_satellite_data()`). Out-of-domain areas are filled with NaNs by default.
- `load_meteorology()` opens the yearly met Zarr store(s) with dask (dims are already `lat`/`lon`/`levels` and duplicates already removed in the store), slices to the requested month if one was given, and selects the requested levels/variables (missing ones are skipped).
- `_process_meteorology()` calls `cut_satellite_met()` to cut the met to the same square grid as the footprints, interpolated to each footprint timestamp.
- `load_topog()` loads topography and land cover, interpolates both to the footprint grid.

Key attributes after loading:

| Attribute | Contents |
|-----------|----------|
| `data.fp_data_full` | Full footprint Dataset (dims: `time`, `lat`, `lon`) 
| `data.fp_xr` | Footprints cut to square (dims: `time`, `lat`, `lon`, with artificial centred coordinates) |
| `data.met` | Met cut to square, aligned to footprint times |
| `data.topog` | Topography/landcover interpolated to footprint grid |

**Subsampling** — use `freq` to load every N-th timestep (e.g. `freq=2` loads half the data).

The xarrays have coordinates `[0,1,2,3 ..., size-1]`, with the release coordinate at `int(size/2)`

---

### Step 2 — Build the inputs array: `get_square_satellite_inputs_v2` *(v1 deprecated)*

`get_square_satellite_inputs_v2` is the current default. The old `get_square_satellite_inputs` (v1) is deprecated — do not use it for new runs.

Key differences from v1:
- `met_variables` is a **flat list** of variable names (not a dict mapping names to levels).
- `met_levels` is a **single shared list** of levels applied to all atmospheric variables — surface variables (those without a `levels` dimension in the met dataset) are detected automatically.
- All `time_deltas` are processed in a single dask compute call, eliminating repeated I/O.
- Adds `interp_to`: pass a pandas offset string (e.g. `"1h"`) to linearly interpolate met to rounded target times rather than snapping to the nearest timestamp.

```python
from gates.data.datasets import get_square_satellite_inputs_v2

met_variables = ["x_wind", "y_wind", "atmosphere_boundary_layer_thickness"]
met_levels = [15, 21]   # shared across all atmospheric variables; surface vars detected automatically

inputs, data = get_square_satellite_inputs_v2(
    data,
    met_variables=met_variables,
    met_levels=met_levels,
    time_deltas=[6],                                            # also extract met at t−6h
    static_variables=["topog", "lat_coords", "lon_coords"],     # optional static fields
    interp_to="1h",   # optional: interpolate met to 1h-rounded times (omit to snap to nearest)
)
```

This stacks all variables along a `variable_name` MultiIndex of tuples `(variable, level, time_delta)`. Static variables always have `level=0, time_delta=0`.

Returns `inputs`, an xarray DataArray of shape `(fp_time, lat, lon, variable_name)`, and the updated data object — fp_time indices where met interpolation failed are dropped from both.

Available `static_variables`: `topog`, `landcover`, `lat_coords`, `lon_coords`, and others — see `get_static_variables_functions()` in `load_data_helper_funs.py`.

---

### Step 3 — Scale inputs: `InputsDataset` / `DefaultInputsScaler` / `HandcraftedInputsScaler`

```python
from gates.data.datasets import InputsDataset

inp_ds = InputsDataset(inputs, scaler_params={"fit_on_subsample": 0.2})
inp_ds.fit()                          # fit on 20% of timesteps (faster)
scaled_inputs = inp_ds.transform(inputs)
```

`DefaultInputsScaler` applies:
- **Standard scaler** (zero mean, unit variance) per variable and level for met variables.
- **MinMax scaler** (to [0, 1]) for static fields: `topog`, `land_cover`, coordinate variables.

`fit_on_subsample` (0–1) controls what fraction of timesteps to use when fitting — useful for large datasets.

Two `scaler_params` keys control which variables get which treatment:
- `minmax_variables` — list of variable names to apply MinMax scaling instead of standard scaling (default includes `topog`, `land_cover`, coordinate fields).
- `ignore_variables` — list of variable names to pass through unchanged (assigned a `GhostScaler` internally). Variables in both lists are treated as ignored, with a warning.

```python
# Example: use standard scaling for topog, skip coordinate variables entirely
inp_ds = InputsDataset(
    inputs,
    scaler_params={
        "minmax_variables": [],                              # no MinMax overrides
        "ignore_variables": ["lat_coords", "lon_coords"],   # pass through unchanged
    }
)
```

**`HandcraftedInputsScaler`** — alternative to `DefaultInputsScaler` for when you want reproducible, externally validated statistics rather than computing them from the data. Pass a `stats_file` (path to a JSON file or a dict). Each variable entry has a `"type"` key (`"standard"`, `"minmax"`, or `"ghost"`) plus per-level statistics:

```json
{
    "x_wind": {"type": "standard", "15": {"mean": 5.21, "std": 3.14}},
    "topog":  {"type": "minmax",   "0":  {"min": -50.0, "max": 3200.0}},
    "lat_coords": {"type": "ghost"}
}
```

Level keys are strings (JSON requirement) and are cast to int internally. If a variable or level is missing from the stats file, a data-driven standard scaler is fitted instead. Example stats files are in `scaler_files/`.

```python
inp_ds = InputsDataset(inputs, scaler="HandcraftedInputsScaler",
                       scaler_params={"stats_file": "scaler_files/my_stats.json"})
inp_ds.fit()
scaled_inputs = inp_ds.transform(inputs)
```

**`GhostScaler`** — passthrough (identity) scaler. Useful for ablation runs where you want a variable included in the input tensor but not transformed.

---

### Step 4 — Scale footprints: `FootprintDataset`

```python
from gates.data.datasets import FootprintDataset

fp_ds = FootprintDataset(data.fp_xr)
scaled_fps = fp_ds.fit_transform()
# scaled_fps is an xarray Dataset with variables:
#   "fp_transformed"  — log-scaled footprints for training
#   "fp_original"     — originals kept for reference
```

The default scaler (`LogAndShiftMeanFpScaler`) takes log₁₀ of non-zero values and shifts so the mean of the log is ~0. It does not need manual parameter tuning.

To recover original values (e.g. for evaluation):
```python
fp_recovered = fp_ds.inverse_transform(scaled_fps)
```

---

### Step 5 — Build DataLoader: `make_dataloader`

Uses `xbatcher` to batch lazily — data is not all loaded into memory at once. Returns a standard `torch.utils.data.DataLoader` yielding `(inputs_batch, fp_batch)` pairs. `inputs_batch` always has shape `(batch_size, lat, lon, variable_name)`.

The shape of the `fp_batch` will depend on what footprint data was passed. Passing a DataArray will return batches of shape `(batch_size, lat, lon)`:

```python
from gates.data.datasets import make_dataloader

train_loader, fp_labels = make_dataloader(
    scaled_inputs,
    scaled_fps["fp_transformed"],
    batch_size=32,
    randomize=True,    # shuffle for training; set False for val/test
)
```
Passing a Dataset with multiple variables will stack them. For example, `scaled_fps` has variables `["fp_transformed", "fp_original"]`, so each `fp_batch` will have shape `(batch_size, lat, lon, n_fp_labels)`:

```python
# 5. DataLoader with multiple output labels
train_loader, fp_labels = make_dataloader(
    scaled_inputs, scaled_fps,
    batch_size=10, randomize=True,
)

train_features, train_labels = next(iter(dataloader))
```
You can pass parameters directly to Torch's dataloader by passing a dictionary 
```python
dataloader_params={"prefetch_factor": None, # whether to pre-load the following batches while the current one is being used 
    "num_workers": 0, # how many subprocesses to use in parallel
    "shuffle" : False # whether to shuffle the batches each epoch
    }
train_loader, fp_labels = make_dataloader(
    scaled_inputs, scaled_fps,
    batch_size=10, dataloader_params=dataloader_params
)
```
In a notebook (or for small tests) the above dataloader parameters are recommended. In computation settings that allow more memory, increasing these parameters should help with memory and latency (e.g. `dataloader_params={"prefetch_factor": 2, "num_workers": 2}` )


---

## Full Example

```python
from gates.data.load_data import LoadSquareSatelliteData
from gates.data.datasets import (get_square_satellite_inputs_v2, InputsDataset,
                                  FootprintDataset, make_dataloader)

# 1. Load
data = LoadSquareSatelliteData(year=2016, region="BRAZIL", month="01", size=10)

# 2. Build inputs (v2 API: flat list + shared met_levels)
met_variables = ["x_wind", "y_wind", "upward_air_velocity", "atmosphere_boundary_layer_thickness"]
met_levels = [3, 15]
inputs, data = get_square_satellite_inputs_v2(
    data,
    met_variables=met_variables,
    met_levels=met_levels,
    time_deltas=[6],
    static_variables=["topog", "lat_coords", "lon_coords"],
)

# 3. Scale inputs
inp_ds = InputsDataset(inputs, scaler_params={"fit_on_subsample": 0.2})
inp_ds.fit()
scaled_inputs = inp_ds.transform(inputs)

# 4. Scale footprints
fp_ds = FootprintDataset(data.fp_xr)
scaled_fps = fp_ds.fit_transform()

# 5. DataLoader
train_loader, _ = make_dataloader(
    scaled_inputs, scaled_fps["fp_transformed"],
    batch_size=10, randomize=True,
)

train_features, train_labels = next(iter(dataloader))
```

