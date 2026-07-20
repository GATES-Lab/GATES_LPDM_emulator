# How-To: Data Pipeline

This guide covers how to load, process, and prepare data for the LPDM emulator using `load_data.py` and `datasets.py`.

The pipeline takes LPDM footprints + meteorology + static fields, cuts them to squares around each satellite measurement release point, scales them, and produces a PyTorch DataLoader for training.

---

## Required Data

| Data | Format | Required dimensions / variables |
|------|--------|--------------------------------|
| **Footprints (satellite)** | NetCDF (one file per month) **or** Zarr (one store per year) | dims: `time`, `lat`, `lon`; vars: `fp`, `release_lat`, `release_lon` |
| **Footprints (receptor)** | Zarr, one store per year | dims: `time`, `receptor`, `lat`, `lon`; vars: `fp`, `release_lat`, `release_lon` (one per `time`×`receptor`) |
| **Meteorology** | Zarr, one store per year | dims: `time`, `levels`, `lat`, `lon` |
| **Topography** | Single global NetCDF | dims: `lat`, `lon` |
| **Land cover** | Single global NetCDF | dims: `lat`, `lon`, `pseudo_level` |

Footprints load in either NetCDF or Zarr (see [Footprints: NetCDF or Zarr](#footprints-netcdf-or-zarr)); meteorology is Zarr-only (see [Meteorology: yearly Zarr stores](#meteorology-yearly-zarr-stores)). Both footprint layouts — satellite and receptor — are put onto a shared `sample_id` index at load (see [The `sample_id` layout](#the-sample_id-layout)).

**File naming conventions** (if using default paths):
- Footprints (NetCDF): `*REGION*DOMAIN_*YYYYMM.nc`
- Footprints (Zarr): `*REGION*DOMAIN_*YYYY.zarr`
- Met (Zarr): `DOMAIN_Met_*YYYY.zarr`

Config roots: `fp_datadir` (NetCDF footprints), `fp_zarr_datadir` (Zarr footprints), `met_datadir` (met). If you're not using default paths, pass the root as `"/path/to/file/filename_"` — the loader appends the date and suffix.

**Built-in region → domain mappings:**

| Region | Domain |
|--------|--------|
| `BRAZIL` | `SOUTHAMERICA` |
| `SOUTHAMERICA` | `SOUTHAMERICA` |
| `SAHARA` | `NORTHAFRICA` |
| `INDIA` | `SOUTHASIA` |

Other regions require passing `domain` explicitly.

---

## The `sample_id` layout

Every dataset the loader produces — footprints, meteorology, topography, fluxes — is aligned along a single integer **`sample_id`** dimension. One `sample_id` is one datapoint: a cut footprint and everything that goes with it. `sample_id` is a stable, 1-based index assigned at load time.

This is what unifies the two footprint layouts:

- **Satellite footprints** have exactly one release per timestamp, so each `time` uniquely labels a datapoint. Loading just relabels the `time` axis to `sample_id`, keeping `time` as a coordinate along it.
- **Receptor footprints** come as `(time, receptor, lat, lon)`: at each timestep there are multiple receptors at different locations, so a datapoint is a `(time, receptor)` pair. Loading flattens `(time, receptor)` into `sample_id`, keeping both `time` and `receptor` as coordinates along it.

`time` (and `receptor`) are plain coordinates, **not** indexes — a secondary index on the sample dimension breaks xarray's alignment in the cropping/batching steps. So you can't `.sel(time=…)` directly; add the index first on whatever object you're inspecting:

```python
data.fp_xr.set_xindex("time").sel(time="2016-01-01T12:00:00")
# receptor data also: .set_xindex("receptor")
```

`data.remove_indeces()` accepts either `sample_id` values **or** timestamps (dropping every sample at those times — for receptor data, all receptors sharing the timestamp) and removes them from every aligned object at once.

---

## Footprints: NetCDF or Zarr

Footprints load from either **monthly NetCDF files** or **yearly Zarr stores**; the two can coexist. The format is chosen by `fp_format`:

- `"auto"` (default) — looks for a Zarr store first, falls back to NetCDF. Zarr wins for a year available in both.
- `"zarr"` / `"nc"` — force a format.

Passing only one of `fp_datadir` (NetCDF root) / `fp_zarr_datadir` (Zarr root) also selects that format.

**Layout & naming** (default paths)
```
<fp_datadir>/DOMAIN/*REGION*DOMAIN_YYYYMM.nc       # monthly NetCDF
<fp_zarr_datadir>/DOMAIN/*REGION*DOMAIN_YYYY.zarr  # yearly Zarr
```

The Zarr stores are written by `convert_fps_to_zarr()` — a conversion utility still **in progress and not yet available**. It bakes in the preprocessing the NetCDF path does on every load: `latitude/longitude → lat/lon` renamed, timestamps sorted and de-duplicated, and the known bad-file workarounds applied. So `chunk`, `parallel_loading` and `bad_files_list` apply to NetCDF only and are ignored for Zarr. A single-month load opens the whole-year store and slices to the month after opening.

---

## Meteorology: yearly Zarr stores

Meteorology loads from **one Zarr store per year**. The `met_datadir` config key points at the Zarr root, and a single-month load opens the whole-year store and slices to the month.

**Layout & naming**
```
<met_datadir>/DOMAIN/DOMAIN_Met_YYYY.zarr     # one store per year
```

**What the store already contains** — the conversion (`data_utils/convert_met_to_zarr.py`) bakes in, at write time, everything the loader used to redo on every load, so the Zarr is ready to use as-is:
- dims already renamed: `latitude/longitude → lat/lon`, `model_level_number → levels`
- unused UM variables dropped (`forecast_period`, `forecast_reference_time`,
  `level_height_0`, `sigma_0`)
- duplicate/unsorted timestamps and duplicate lat/lon removed
- native chunks `{time:1, lat:-1, lon:-1, levels:3}` — tuned so a single timestep
  (the dominant scattered-access pattern) is cheap to read

---

## Pipeline Steps

### Step 1 — Load and cut data: `LoadSquareSatelliteData` / `LoadReceptorData`

```python
from gates.data.load_data import LoadSquareSatelliteData

data = LoadSquareSatelliteData(
    year=2016,
    region="BRAZIL",
    month="01",   # omit for a full year
    size=10,      # cut to 10×10 square around release point
    freq=3,       # load one in every three samples, sampled regularly
    # fp_format="auto",  # "auto" (default) / "zarr" / "nc" — see Footprints: NetCDF or Zarr
)
```

Receptor-format footprints load the same way through `LoadReceptorData`, which reuses all of the square-cutting machinery and only differs in how the raw footprints are flattened onto `sample_id`:

```python
from gates.data.load_data import LoadReceptorData

data = LoadReceptorData(
    year=2016,
    region="BRAZIL",
    size=10,
    fp_zarr_datadir="/path/to/receptor_zarr/DOMAIN/",  # or rely on the config
    load_everything=True,   # also load + crop met and topog onto the sample_id grid
)
```
> Receptor support currently covers loading and cropping footprints/met/topog onto the `sample_id` grid; the flux and inputs/DataLoader steps are still being wired up for it.

What happens under the hood:
- `load_fps()` opens the footprints — monthly NetCDF files (concatenated, with the known-malformed files handled automatically) or yearly Zarr stores, depending on `fp_format`.
- `_prepare_samples()` puts the footprints onto the `sample_id` index: satellite data relabels `time` → `sample_id`; receptor data flattens `(time, receptor)` → `sample_id`.
- `_process_footprints()` cuts each footprint to a `size × size` square centred on its `release_lat`/`release_lon` (via `cut_satellite_data()`). Out-of-domain areas are filled with NaNs by default.
- `load_meteorology()` opens the yearly met Zarr store(s) with dask.
- `_process_meteorology()` calls `cut_satellite_met()` to cut the met to the same square grid as the footprints, matched to each sample's footprint time (`fp_time − time_delta`).
- `load_topog()` loads topography and land cover, interpolates both to the footprint grid.

Key attributes after loading (all aligned on `sample_id`):

| Attribute | Contents |
|-----------|----------|
| `data.fp_data_full` | Full (uncut) footprint Dataset (dims: `sample_id`, `lat`, `lon`) |
| `data.fp_xr` | Footprints cut to square (dims: `sample_id`, `lat`, `lon`, with artificial centred coordinates) |
| `data.met` | Met cut to square, aligned to footprint samples (dims: `sample_id`, `lat`, `lon`, `levels`) |
| `data.topog` | Topography/landcover cut to square (dims: `sample_id`, `lat`, `lon`) |

**Subsampling** — use `freq` to keep every N-th sample (e.g. `freq=2` loads half the data); operates along the `sample_id` index.

Each cut square has artificial `lat`/`lon` coordinates `[0, 1, 2, …, size−1]`, with the release point at `int(size/2)`. The real geographic coordinates are kept in the `lat_coords` (`sample_id`, `lat`) and `lon_coords` (`sample_id`, `lon`) variables.

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

Returns `inputs`, an xarray DataArray of shape `(sample_id, lat, lon, variable_name)` (with the footprint time kept as an `fp_time` coordinate along `sample_id`), and the updated data object — samples where met interpolation failed are dropped from both.

Available `static_variables`: `topog`, `landcover`, `lat_coords`, `lon_coords`, and others — see `get_static_variables_functions()` in `load_data_helper_funs.py`.

---

### Step 3 — Scale inputs: `InputsDataset` / `DefaultInputsScaler` / `HandcraftedInputsScaler`

```python
from gates.data.datasets import InputsDataset

inp_ds = InputsDataset(inputs, scaler_params={"fit_on_subsample": 0.2})
inp_ds.fit()                          # fit on 20% of samples (faster)
scaled_inputs = inp_ds.transform(inputs)
```

`DefaultInputsScaler` applies:
- **Standard scaler** (zero mean, unit variance) per variable and level for met variables.
- **MinMax scaler** (to [0, 1]) for static fields: `topog`, `land_cover`, coordinate variables.

`fit_on_subsample` (0–1) controls what fraction of samples to use when fitting — useful for large datasets.

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

