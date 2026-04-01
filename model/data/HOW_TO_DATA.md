# How-To: Data Pipeline

This guide covers how to load, process, and prepare data for the LPDM emulator using `load_data.py` and `datasets.py`.

The pipeline takes LPDM footprints + meteorology + static fields, cuts them to squares around each satellite measurement release point, scales them, and produces a PyTorch DataLoader for training.

---

## Required Data

| Data | Format | Required dimensions / variables |
|------|--------|--------------------------------|
| **Footprints** | NetCDF `.nc`, one file per month | dims: `time`, `lat`, `lon`; vars: `fp`, `release_lat`, `release_lon` |
| **Meteorology** | NetCDF `.nc`, one or more files per month | dims: `time`, `levels` (or `model_level_number`), `lat`/`latitude`, `lon`/`longitude` |
| **Topography** | Single global NetCDF | dims: `lat`, `lon` |
| **Land cover** | Single global NetCDF | dims: `lat`, `lon`, `pseudo_level` |

**File naming conventions** (if using default paths):
- Footprints: `*REGION*DOMAIN_*YYYYMM.nc`
- Met: `DOMAIN_Met_*YYYYMM.nc`
If you're not using default paths, your files still need to end in `YYYYMM.nc'`, and the path should be in the format `fp_dir = "/path/to/file/filename_"`

**Built-in region → domain mappings:**

| Region | Domain |
|--------|--------|
| `BRAZIL` | `SOUTHAMERICA` |
| `SOUTHAMERICA` | `SOUTHAMERICA` |
| `SAHARA` | `NORTHAFRICA` |
| `INDIA` | `SOUTHASIA` |

Other regions require passing `domain` explicitly.

---

## Pipeline Steps

### Step 1 — Load and cut data: `LoadSquareSatelliteData`

```python
from model.data.load_data import LoadSquareSatelliteData

data = LoadSquareSatelliteData(
    year=2016,
    region="BRAZIL",
    month="01",   # omit for a full year
    size=10,      # cut to 10×10 square around release point
)
```

What happens under the hood:
- `load_fps()` opens and concatenates footprint NetCDF files; handles a hardcoded list of known malformed files automatically.
- `_process_footprints()` cuts each footprint to a `size × size` square centred on its `release_lat`/`release_lon` (via `cut_satellite_data()`). Out-of-domain areas are filled with NaNs by default.
- `load_meteorology()` opens met files with dask, renames `latitude`/`longitude` → `lat`/`lon`, and drops duplicates.
- `_process_meteorology()` calls `cut_satellite_met()` to cut the met to the same square grid as the footprints, interpolated to each footprint timestamp.
- `load_topog()` loads topography and land cover, interpolates both to the footprint grid.

Key attributes after loading:

| Attribute | Contents |
|-----------|----------|
| `data.fp_data_full` | Full footprint Dataset (dims: `time`, `lat`, `lon`) NOTE: soon to be obsolete|
| `data.fp_xr` | Footprints cut to square (dims: `time`, `lat`, `lon`, with artificial centred coordinates) |
| `data.met` | Met cut to square, aligned to footprint times |
| `data.topog` | Topography/landcover interpolated to footprint grid |

**Subsampling** — use `freq` to load every N-th timestep (e.g. `freq=2` loads half the data).

The xarrays have coordinates `[0,1,2,3 ..., size-1]`, with the release coordinate at `int(size/2)`

---

### Step 2 — Build the inputs array: `get_square_satellite_inputs`

```python
from model.data.datasets import get_square_satellite_inputs

met_variables = {
    "x_wind": [15, 20],          # atmospheric variable: list of model levels to extract
    "y_wind": [15, 20],
    "atmosphere_boundary_layer_thickness": [0], # a 2D variable, with no levels
}

inputs = get_square_satellite_inputs(
    data,
    met_variables,
    time_deltas=[6],                       # also extract met at t-6h 
    static_variables=["topog", "lat_coords", "lon_coords"],  # optional static fields
)
```

This stacks all variables along a `variable_name` MultiIndex of tuples `(variable, level, time_delta)`. Static variables always have `level=0, time_delta=0`.

Returns an xarray DataArray of shape `(fp_time, lat, lon, variable_name)`.

Available `static_variables`: `topog`, `landcover`, `lat_coords`, `lon_coords`, and others — see `get_static_variables_functions()` in `load_data_helper_funs.py`.

---

### Step 3 — Scale inputs: `InputsDataset` / `DefaultInputsScaler`

```python
from model.data.datasets import InputsDataset

inp_ds = InputsDataset(inputs, scaler_params={"fit_on_subsample": 0.2})
inp_ds.fit()                          # fit on 20% of timesteps (faster)
scaled_inputs = inp_ds.transform(inputs)
```

`DefaultInputsScaler` applies:
- **Standard scaler** (zero mean, unit variance) per variable and level for met variables.
- **MinMax scaler** (to [0, 1]) for static fields: `topog`, `land_cover`, coordinate variables.

`fit_on_subsample` (0–1) controls what fraction of timesteps to use when fitting — useful for large datasets.

---

### Step 4 — Scale footprints: `FootprintDataset`

```python
from model.data.datasets import FootprintDataset

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

```python
from model.data.datasets import make_dataloader

train_loader = make_dataloader(
    scaled_inputs,
    scaled_fps["fp_transformed"],
    batch_size=32,
    randomize=True,    # shuffle for training; set False for val/test
)
```

Uses `xbatcher` to batch lazily — data is not all loaded into memory at once. Returns a standard `torch.utils.data.DataLoader` yielding `(inputs_batch, fp_batch)` pairs of shape `(batch_size, lat, lon, variable_name)` and `(batch_size, lat, lon)`.

---

## Full Example

```python
from model.data.load_data import LoadSquareSatelliteData
from model.data.datasets import (get_square_satellite_inputs, InputsDataset,
                                  FootprintDataset, make_dataloader)

# 1. Load
data = LoadSquareSatelliteData(year=2016, region="BRAZIL", month="01", size=10)

# 2. Build inputs
met_variables = {
    "x_wind": [15, 20], "y_wind": [15, 20],
    "specific_humidity": [15], "surface_temperature": [],
}
inputs = get_square_satellite_inputs(
    data, met_variables,
    time_deltas=[6],
    static_variables=["topog", "lat_coords"],
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
    batch_size=32, randomize=True,
)

train_features, train_labels = next(iter(dataloader))
```

Passing a dataset with multiple variables will stack them. For example, `scaled_fps` has variables `["fp_transformed", "fp_original"]`, so 
```python
# 5. DataLoader with multiple output labels
train_loader, fp_labels = make_dataloader(
    scaled_inputs, scaled_fps,
    batch_size=32, randomize=True,
)

train_features, train_labels = next(iter(dataloader))
```
will yield `train_labels` with shape `(batch_size, lat, lon, n_labels)`, in this case `n_labels=2` and `train_labels=["fp_transformed", "fp_original"]`.