# How-To: Data Pipeline

This guide covers how to load, process, and prepare data for the LPDM emulator using `load_data.py` and `datasets.py`.

The pipeline takes LPDM footprints + meteorology + static fields, cuts them to squares around each satellite measurement release point, scales them, and produces a PyTorch DataLoader for training.

For a hands-on walkthrough of each step on a small data subset, see the [data tutorial notebook](../notebooks/data_tutorial.ipynb).

---

## Required Data

| Data | Format | Required dimensions / variables |
|------|--------|--------------------------------|
| **Footprints** | NetCDF `.nc`, one file per month | dims: `time`, `lat`, `lon`; vars: `fp`, `release_lat`, `release_lon` |
| **Meteorology** | Zarr, one store per year | dims: `time`, `levels`, `lat`, `lon` |
| **Topography** | Single global NetCDF | dims: `lat`, `lon` |
| **Land cover** | Single global NetCDF | dims: `lat`, `lon`, `pseudo_level` |
| **Flux maps** *(optional)* | NetCDF, one file per species per year: `<flux_datadir>/<DOMAIN>/<species>_<DOMAIN>_<YYYY><flux_suffix>.nc` | dims: `time`, `lat`, `lon`; var: `flux`. Only used for evaluation — see [Flux / Mole-Fraction Metrics](HOW_TO_evaluation.md#flux--mole-fraction-metrics) |

### Where the data lives: `config.yml` paths

The loader builds every data path from the `data_paths` section of `config.yml` (see the [config guide](HOW_TO_CONFIG.md)). Each entry is joined onto `base_data_path`, so `base_data_path` is the single place to point at your data root. An example:

```yaml
data_paths:
  base_data_path:    /group/chem/acrg          # prepended to every path below
  fp_datadir:        /LPDM/fp_NAME_pre20210701/
  met_datadir:       /met_archive/zarr_store/
  topog_datadir:     /LPDM/topog_NAME/TopogUMG_Mk8_global.nc
  landcover_datadir: /LPDM/topog_NAME/land_cover.nc     # optional
  flux_datadir:      /LPDM/emissions/
```

`fp_datadir`, `met_datadir` and `flux_datadir` are **directories**; `topog_datadir` and `landcover_datadir` point at **single global files**.

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
    year=2018,
    region="BRAZIL",
    month="01",   # omit for a full year
    size=10,      # cut to 10×10 square around release point
    freq=3,       # load one in every three footprints, sampled regularly along the time axis
    crop_met=False,  # not needed when building inputs with get_square_satellite_inputs_v2
    met_args={       # met to load; must include every variable used in Step 2
        "met_variables": ["x_wind", "y_wind", "atmosphere_boundary_layer_thickness"],
        "met_levels": [15, 21],
    },
)
```

`met_args["met_variables"]` is required: if it is empty or missing, no met variables are loaded. `met_levels` is optional; if it is empty, all levels are kept. Training fills `met_args` automatically from the `variables` section of the parameter file.

What happens under the hood:
- `load_fps()` opens and concatenates footprint NetCDF files. Files listed under `bad_fp_files` in `config.yml` are known to be malformed and are opened with a workaround.
- `_process_footprints()` cuts each footprint to a `size × size` square centred on its `release_lat`/`release_lon` (via `cut_satellite_data()`). Out-of-domain areas are filled with NaNs by default.
- `load_meteorology()` opens the yearly met Zarr store(s) with dask, slices to the requested month if one was given, and selects the levels/variables in `met_args` (ones missing from the store are skipped). The result is stored, uncropped, as `data.met_file`.
- `load_topog()` loads topography and land cover, interpolates both to the footprint grid.
- If `crop_met=True` (the default), `_process_meteorology()` also calls `cut_satellite_met()` to cut the met to the same square grid as the footprints, using the nearest met timestamp (within 4h) to each footprint time, and stores it as `data.met`. The inputs function in Step 2 cuts the met itself from `data.met_file`, so training runs set `crop_met: false` to skip this step.

Key attributes after loading:

| Attribute | Contents |
|-----------|----------|
| `data.fp_data_full` | Full footprint Dataset (dims: `time`, `lat`, `lon`) |
| `data.fp_xr` | Footprints cut to square (dims: `time`, `lat`, `lon`, with artificial centred coordinates) |
| `data.met_file` | Met for the whole domain, not cropped |
| `data.met` | Met cut to square, aligned to footprint times (only when `crop_met=True`) |
| `data.topog` | Topography/landcover interpolated to footprint grid |

**Subsampling** — use `freq` to load every N-th timestep (e.g. `freq=2` loads half the data).

The xarrays have coordinates `[0,1,2,3 ..., size-1]`, with the release coordinate at `int(size/2)`

---

### Step 2 — Build the inputs array: `get_square_satellite_inputs_v2`

- `met_variables` is a **flat list** of variable names.
- `met_levels` is a **single shared list** of levels applied to all atmospheric variables — surface variables (those without a `levels` dimension in the met dataset) are detected automatically.
- All `time_deltas` are processed in a single dask compute call.
- By default, each target time (footprint time − Δh) snaps to the nearest met timestamp. Samples with no met timestamp within 4h are dropped.
- `interp_to`: pass a pandas offset string (e.g. `"1h"`) to round each target time to that resolution and linearly interpolate the met between the two met timestamps either side. Samples whose rounded time falls outside the met record are dropped.

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

> **TODO:** add a page listing the available met variables (and their levels) and static variables.

---

### Step 3 — Scale inputs: `InputsDataset` / `DefaultInputsScaler`

```python
from gates.data.datasets import InputsDataset

inp_ds = InputsDataset(inputs, fit_on_subsample=0.8)
inp_ds.fit()                          # fit on 80% of timesteps (faster)
scaled_inputs = inp_ds.transform(inputs)
```

`fit_on_subsample` (0–1) is an argument of `InputsDataset`. It controls what fraction of timesteps, chosen at random, is used when fitting — useful for large datasets.

By default, `InputsDataset` uses `DefaultInputsScaler`, which applies:
- **Standard scaler** (zero mean, unit variance) per variable and level for met variables.
- **MinMax scaler** (to [0, 1]) for `landcover`, `topog` and the coordinate/distance fields (`lat_coords`, `lon_coords`, `x_coords`, `y_coords`, their sin/cos versions, `xy_distance_centre`, `earth_distance_centre`).
- **No transform** for the disaggregated landcover fields (`landcover_type_0` … `landcover_type_9`), which are already fractions in [0, 1].

The default lists are `DEFAULT_MINMAX_VARIABLES` and `DEFAULT_IGNORE_VARIABLES` in `gates/data/datasets.py`. Two `scaler_params` keys control which variables get which treatment:
- `minmax_variables` — list of variable names to apply MinMax scaling to instead of standard scaling. Passing it replaces the default MinMax list above.
- `ignore_variables` — list of variable names to pass through unchanged. Passing it replaces the default ignore list, so include the `landcover_type_*` names if you still want them untransformed. Variables in both lists are ignored.

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

Two footprint scalers are available, chosen with the `scaler` argument:
- `LogAndShiftMeanFpScaler` — the default when `scaler` is not passed. Takes log₁₀ of non-zero values and shifts so the mean of the log is ~0. The shift is fitted from the data.
- `LogAndShiftFpScaler` — takes log₁₀ of non-zero values and adds a fixed order-of-magnitude offset (`minimum_oom`, default 5). With `non_negative=True` (default), values that end up below zero — footprints smaller than 10⁻⁵ — are set to 0. It needs no fitting, so it translates across domain sizes. This is the one set in most parameter-file templates (`"fp_scaler": {"scaler": "LogAndShiftFpScaler"}`).

```python
from gates.data.datasets import FootprintDataset, LogAndShiftFpScaler

fp_ds = FootprintDataset(data.fp_xr, scaler=LogAndShiftFpScaler)
```

To recover original values (e.g. for evaluation):
```python
fp_recovered = fp_ds.inverse_transform(scaled_fps)
```

#### NaNs and the `fp_nan_mask`

Footprints cut near the edge of the domain contain NaNs where the square extends past it (see Step 1). NaNs cannot go through the model or the loss, so `FootprintDataset` can replace them with zeros and record where they were:

```python
fp_ds = FootprintDataset(data.fp_xr, add_nan_mask=True)
scaled_fps = fp_ds.fit_transform()
# scaled_fps now also has:
#   "fp_nan_mask"  — int32, 1 where fp_original was NaN, 0 where it was valid
```

With `add_nan_mask=True`, `transform` builds the mask from `fp_original` and then fills every NaN in the Dataset with 0 (via `add_fp_nan_mask()`). The default is `False`, which leaves NaNs in place.

The mask is there so the zero-filled pixels can be left out later. Without it, the model would be trained to predict 0 outside the domain, and those pixels would count towards the metrics:
- **Loss** — losses in `gates/evaluation/loss_functions.py` take a `nan_mask_label` argument. When set, they read the mask from that variable in `fp_batch` and ignore pixels where it is 1.
- **Metrics** — `calculate_losses()` in `gates/training/training.py` passes `fp_nan_mask` as `ignore_mask` to the metric functions whenever it is present in the test outputs.

In training, both are controlled by `dataloader.nans_to_zeros` in the parameter file (default `true`): it sets `add_nan_mask` on the `FootprintDataset`, and sets `nan_mask_label="fp_nan_mask"` on the training and test losses. Keep it `true`.

**Extra variables** — if you pass a Dataset rather than a DataArray, any other variable with dims `(time, lat, lon)` (e.g. `flux`, added by `data.get_flux()`) is copied into the output unchanged (`keep_vars=True`, the default). This is how losses that need the flux get it in `fp_batch`.

---

### Step 5 — Build DataLoader: `make_dataloader` / `make_tensor_dataloader`

Two loaders are available. Both return a standard `torch.utils.data.DataLoader` yielding `(inputs_batch, fp_batch)` pairs, plus the list of footprint labels.

| | `make_dataloader` (xbatcher) | `make_tensor_dataloader` (in-memory tensors) |
|---|---|---|
| How it batches | `xbatcher` slices the xarray objects batch by batch | Converts the inputs and footprints to torch tensors once, wrapped in a `TensorDataset` |
| Shuffling | `randomize=True` permutes `fp_time` once; batches are fixed contiguous windows of that order | `shuffle=True` reshuffles which samples go in each batch every epoch |
| Workers | `dataloader_params` passed to torch (defaults: 4 workers, `forkserver`) | Always `num_workers=0` |
| `flatten` default | `False` | `True` |
| Used in training | Test loader always; train loader by default | Train loader when `"review_fixes": {"use_tensor_loader": true}` |

**Load the data into memory first.** `xbatcher` is designed to batch lazily, but reading lazy (dask/Zarr) data batch by batch is currently too slow to be usable. Call `.compute()` on the inputs and footprints after Step 2, before scaling. Training does this when `load_before_training` is `true` (the default). `make_tensor_dataloader` always needs the data in memory, since it builds tensors from it.

The rest of this section uses `make_dataloader`. `inputs_batch` has shape `(batch_size, lat, lon, variable_name)`.

With `flatten=True`, lat and lon are stacked into a single dimension, so `inputs_batch` has shape `(batch_size, lat*lon, variable_name)` and `fp_batch` loses its lat/lon dims in the same way. Training uses `flatten=True`, as the model expects flattened inputs.

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

train_features, train_labels = next(iter(train_loader))
```

#### `fp_labels`

`fp_labels` is the list of names for the last dimension of `fp_batch`, in order. The tensors have lost their variable names, so this list is how you know which slice holds which variable:

| `fps` passed | `fp_labels` | `fp_batch` shape (`flatten=True`) |
|---|---|---|
| `scaled_fps["fp_transformed"]` (DataArray) | `["fp_transformed"]` (the DataArray's name, or `"fp"` if it has none) | `(batch_size, lat*lon)` |
| `scaled_fps` (Dataset) | `list(scaled_fps.data_vars)`, e.g. `["fp_transformed", "fp_original", "fp_nan_mask", "flux"]` | `(batch_size, lat*lon, len(fp_labels))` |

To pick out a variable, look up its index:

```python
orig = train_labels[..., fp_labels.index("fp_original")]
```

In training:
- The whole Dataset is passed, so `fp_batch` carries the target and everything the loss needs.
- The training loop takes **index 0** as the target (`fp_batch[:, :, 0]` in `scripts/train_GATES_model.py`). This is `fp_transformed`, because `FootprintDataset.transform` puts it first. If you build the Dataset yourself, keep `fp_transformed` first.
- The loss is built with `fp_labels`, so it can find its extra variables by name (`nan_mask_label`, `weight_label`, `flux_label`, …) and raises a `ValueError` if one is missing.
- The train and test loaders must return the same `fp_labels`, otherwise `setup_GATES_dataloaders()` raises an error.

#### Inputs to the dataloader
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

The tensor loader takes the same inputs. `batch_size`, `shuffle` and `num_workers` are set by the function, so any of these in `dataloader_params` are dropped:

```python
from gates.data.datasets import make_tensor_dataloader

train_loader, fp_labels = make_tensor_dataloader(
    scaled_inputs, scaled_fps["fp_transformed"],
    batch_size=10, shuffle=True,   # flatten=True by default
)
```


---

## Full Example

```python
from gates.data.load_data import LoadSquareSatelliteData
from gates.data.datasets import (get_square_satellite_inputs_v2, InputsDataset,
                                  FootprintDataset, make_dataloader)

met_variables = ["x_wind", "y_wind", "upward_air_velocity", "atmosphere_boundary_layer_thickness"]
met_levels = [3, 15]

# 1. Load
data = LoadSquareSatelliteData(
    year=2018, region="BRAZIL", month="01", size=10, crop_met=False,
    met_args={"met_variables": met_variables, "met_levels": met_levels},
)

# 2. Build inputs (flat list of variables + shared met_levels)
inputs, data = get_square_satellite_inputs_v2(
    data,
    met_variables=met_variables,
    met_levels=met_levels,
    time_deltas=[6],
    static_variables=["topog", "lat_coords", "lon_coords"],
)

# load into memory before scaling and batching
inputs = inputs.compute()
data.fp_xr = data.fp_xr.compute()

# 3. Scale inputs
inp_ds = InputsDataset(inputs, fit_on_subsample=0.2)
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

train_features, train_labels = next(iter(train_loader))
```

