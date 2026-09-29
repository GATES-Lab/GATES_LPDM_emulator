# How-To: Parameter File Reference

## Overview

The parameter file is a JSON configuration that controls every aspect of a GATES training run — data loading, variable selection, model architecture, scaling, training schedule, and logging. It is passed to `scripts/train_GATES_model.py` at runtime:

```bash
python scripts/train_GATES_model.py parameter_file.json
```
If your parameter file is not in the `parameter_files_dir` in `config.yml`, you can pass the path as an argument `--file_path /path/to/folder/where/file/is`.

Each run writes its outputs to `<save_models_dir>/<model_name>_<timestamp>/`, where `save_models_dir` comes from `config.yml` (or from `model_save_dir` in the parameter file). A copy of the resolved parameters is saved in that folder as `training_outputs/training_settings_<model_name>_<timestamp>.json`, so each run is fully reproducible.

The tracked templates in `parameter_files/` are a good starting point:

| Template | Use |
|----------|-----|
| `NEW_parameter_template.json` | Single region. Start here. |
| `NEW_parameter_template_multiregion.json` | Two regions (see [HOW_TO_MULTIREGION.md](HOW_TO_MULTIREGION.md)). |
| `NEW_parameter_template_sweep.json` | The template plus a `__sweep__` block (see [Launching a Sweep](HOW_TO_LAUNCH.md#launching-a-sweep)). |
| `NEW_parameter_template_terminal.json`, `NEW_parameter_template_terminal_multiregion.json` | Tiny smoke tests that run in a terminal or on a single CPU node. |

**Required keys.** Keys marked *(required)* below are read without a default, so leaving them out raises a `KeyError`. The required keys are: `model_name`, `use_wandb`, `train_load_data`, `test_load_data`, `variables`, `input_scaler`, `fp_scaler`, `dataloader`, `model_parameters`, `learning_rate`, all four `epochs.*` keys, and `loss_functions.criterion` / `loss_functions.criterion_test`.

---

## Top-Level Fields

```json
{
    "model_name": "gates_template",
    "verbose": true,
    "notes": "...",
    "seed": 34
}
```

| Field | Description |
|-------|-------------|
| `model_name` | (required) Base name for this experiment. A timestamp is appended at runtime, e.g. `gates_template_20260929_101500`. |
| `verbose` | (optional) If `true`, enables verbose logging output during data loading and training. Defaults to `true`. |
| `notes` | (optional) Free-text field for experiment notes. Saved alongside model artefacts. |
| `seed` | (optional) Random seed for reproducibility. Defaults to 34. It also seeds the training loader's shuffle, unless `dataloader.seed` is set. |
| `model_save_dir` | (optional) Folder the run folder is created in. Overrides `save_models_dir` in `config.yml`. |
| `data_dirs` | (optional) Path overrides for the data. See [Custom data paths](#custom-data-paths). |
| `grid_reference_fp` | (optional) Index of the footprint whose lat/lon coordinates are used to build the model grid. Defaults to 0 (the first footprint). |
| `review_fixes` | (optional) Experimental fixes that are off by default. See [`review_fixes`](#review_fixes). |
| `__sweep__` | (optional) Turns the file into a sweep over several configurations. See [Launching a Sweep](HOW_TO_LAUNCH.md#launching-a-sweep). |
| `regions`, `shared_load_parameters` | (multiregion only) Replace `train_load_data` / `test_load_data` in multiregion files. See [HOW_TO_MULTIREGION.md](HOW_TO_MULTIREGION.md). |

> **Note:** Keys that the scripts don't read are ignored. Prefixing a key with `_ignore_` (e.g. `_ignore_model_save_dir`) is a convention for storing an alternative value without activating it.



---

## `train_load_data`

(required) Controls which data is loaded for training. All keys except `year(s)` and `month(s)` are passed to `LoadSquareSatelliteData`.

```json
"train_load_data": {
    "years": ["2014", "2015"],
    "freq": 3,
    "region": "SAHARA",
    "size": 50,
    "crop_met": false
}
```

| Field | Description |
|-------|-------------|
| `years` | (required, or `year`) List of years to include in the training dataset. Must be a list of strings or ints. |
| `year` | Alternative to `years` for a single year. |
| `months` | (optional) List of months to keep, as strings or ints (e.g. `["01", "02"]` or `[1, 2]`). By default, all months are loaded. The whole year is still loaded and then filtered, so `time_deltas` that reach into the previous month are resolved. The exception is a single year with a single month, which loads only that month. |
| `month` | Alternative to `months` for a single month. |
| `region` | (required) Geographic region name. Must match a known region in `config.yml`. The loader falls back to `"BRAZIL"` if it is left out. |
| `domain` | (optional) The footprint domain for `region`. By default, it is looked up from `region` in `config.yml`. |
| `size` | (required) Side length (in grid cells) of the square patch cut around each satellite release point. Must be a positive, *even* int. The loader falls back to 10 if it is left out. |
| `freq` | (recommended) Controls sampling frequency by loading every nth sample. Must be a positive int. If not present, `freq` defaults to 1, which might lead to very large data. Increase the frequency to run a smaller subset of the data. |
| `sampling_mode` | (optional) `"regular"` keeps one footprint in every `freq`, in order. `"random"` keeps N/`freq` footprints chosen at random. Defaults to `"regular"`. |
| `freq_offset` | (optional) With `sampling_mode: "regular"`, shifts the start of the sampling. For example, `freq: 2` with `freq_offset: 0` keeps the even footprints, and `freq_offset: 1` the odd ones. Defaults to 0. |
| `fill_outofdomain_with` | (optional) `"nans"` or `"zeros"`. How to fill the parts of the square that fall outside the domain. Defaults to `"nans"`. |
| `delete_outofdomain` | (optional) If `true`, drops every footprint whose square leaves the domain, instead of filling it. Defaults to `false`. |
| `crop_met` | (set to `false`) Whether `LoadSquareSatelliteData` crops the met around each footprint. The training pipeline crops the met itself, so this step is redundant during training. The code default is `true`, so set it to `false` explicitly. Pass `true` only when working with the data manually. |
| `load_fps_in_mem` | (optional) If `true`, loads the footprints into memory. Defaults to `true`. |

You don't need to set `met_args.met_variables` or `met_args.met_levels`: they are filled in from the [`variables`](#variables) section.

### Custom data paths

By default, the data paths come from `config.yml` (see [HOW_TO_CONFIG.md](HOW_TO_CONFIG.md)). To override them for one run, add a top-level `data_dirs` block. It uses the same key names as `config.yml`, and applies to both the train and test loads:

```json
"data_dirs": {
    "fp_datadir": "/path/to/footprints/prefix_",
    "met_datadir": "/path/to/meteorology/prefix_",
    "topog_datadir": "/path/to/topography.nc",
    "landcover_datadir": "/path/to/landcover.nc"
}
```

| Field | Description |
|-------|-------------|
| `fp_datadir` | Prefix of the monthly footprint files. The date is appended, e.g. `prefix_201601.nc`. |
| `met_datadir` | Prefix of the yearly met Zarr stores. The year is appended, e.g. `prefix_2016.zarr`. |
| `topog_datadir` | Path to the topography file. |
| `landcover_datadir` | Path to the land-cover file. |

Any key you leave out keeps its `config.yml` value.

Alternatively, you can set the paths inside `train_load_data` (or `test_load_data`) using the `LoadSquareSatelliteData` argument names: `"fp_datadir": "..."`, `"met_args": {"met_datadir": "..."}` and `"topog_args": {"topog_path": "...", "landcover_path": "..."}`. If a path is set in both places, the `data_dirs` value is used.

---

## `test_load_data`

(required) Controls which data is loaded for evaluation. It is merged on top of `train_load_data`, so it only needs the keys that differ, typically a held-out year and a larger `freq`. Every other key (`region`, `size`, `crop_met`, …) is taken from `train_load_data`. An empty block `{}` tests on the training data.

```json
"test_load_data": {
    "years": ["2016"],
    "months": ["01", "02", "03"],
    "freq": 100
}
```

| Field | Description |
|-------|-------------|
| `years` | List of years to use as the test/evaluation set. |
| `months` | As above. |
| `freq` | Same as in `train_load_data`. |

---

## `variables`

(required) Selects which variables are assembled into the model inputs. These keys are passed to `get_square_satellite_inputs_v2` (see [HOW_TO_DATA.md](HOW_TO_DATA.md)).

```json
"variables": {
    "met_variables": ["x_wind", "y_wind", ..., "wind_speed", "wind_angle"],
    "met_levels": [3, 9, 15, 21, 30, 42, 51],
    "static_variables": ["sin_lat_coords", ..., "topog", "landcover"],
    "time_deltas": [6, 12],
    "add_wind_direction": true,
    "load_into_memory": false,
    "chunk_size": 64
}
```

| Field | Description |
|-------|-------------|
| `met_variables` | (required) Meteorological variables included in the model input array. Can include derived variables such as `wind_speed` and `wind_angle` if `add_wind_direction` is enabled. |
| `met_levels` | (required) Vertical levels at which meteorological variables are extracted for the input array. Each variable × level combination becomes a separate input channel. Surface variables are detected automatically and have no levels. |
| `static_variables` | (required) Time-invariant input features appended to each node. Includes coordinate encodings, topography, and land cover. |
| `time_deltas` | (optional) List of time offsets (in hours) for which lagged meteorological fields are included (e.g. `[6, 12]` adds met at t−6h and t−12h alongside t). |
| `add_timedelta_zero` | (optional) If `true`, adds the met at t (a time delta of 0) to `time_deltas`. Defaults to `true`. |
| `add_wind_direction` | (optional) If `true`, computes `wind_speed` and `wind_angle` from `x_wind`/`y_wind`. Defaults to `false`. Set it to `true` whenever `met_variables` contains `wind_speed` or `wind_angle`, as all the templates do. |
| `interp_to` | (optional) How each footprint time is matched to the met. If unset, the nearest met timestamp is used. If set to a pandas offset string (e.g. `"1h"`), the time is rounded to that resolution and the met is linearly interpolated between the two surrounding met timestamps. Defaults to unset. |
| `load_into_memory` | Controls how the met is cropped (see note below). If `true`, the **full** meteorology for the **needed timestamps** (whole domain) is loaded into memory before cropping — fast, but high peak memory. If `false`, the met is kept lazy (dask) during cropping: set `chunk_size` (below) to crop in memory-bounded blocks — otherwise the crop is left as a single lazy chunk, which builds a large dask graph and triggers a "large chunk" warning (not recommended)!! |
| `chunk_size` | (recommended when `load_into_memory: false`) Number of footprint samples to crop and load per block. When set (and `load_into_memory` is `false`), the met is read and cropped in blocks of this many samples instead of all at once, keeping peak memory low and avoiding a large-graph warning. Ignored when `load_into_memory` is `true`. Omit to leave the crop fully lazy (currently produces a "large chunk" warning — not recommended). |

> **Memory management:** `load_into_memory` and `chunk_size` set the peak memory of a run, together with the SLURM `--mem` / `--cpus-per-task` that size the Dask workers. See [Dask cluster and per-worker memory](HOW_TO_LAUNCH.md#dask-cluster-and-per-worker-memory) for how they interact.

---

## `input_scaler`

(required) Configures normalisation of meteorological and static input features. Leaving the block out, or setting it to `{}`, currently raises a `NameError`.

```json
"input_scaler": {
    "scaler": "DefaultInputsScaler",
    "fit_on_subsample": 1,
    "scaler_params": {}
}
```

| Field | Description |
|-------|-------------|
| `scaler` | Class name of the input scaler, looked up in `gates.data.datasets`. The default scaler `DefaultInputsScaler` applies `StandardScaler` to met variables and `MinMaxScaler` to static fields. |
| `fit_on_subsample` | (optional) Fraction of the training data used to fit the scaler. Fraction is selected at random. Speeds up scaler fitting on large datasets. Defaults to 1 (use the whole dataset). |
| `scaler_params` | (optional) Additional keyword arguments passed to the scaler constructor. |

Within `DefaultInputsScaler`, the variables that are standardised vs min-max vs no transform can be specified by passing `scaler_params:{"ignore_variables": ["lat_coords", "lon_coords"], "minmax_variables":["landcover", "topog", "xy_distance_centre", ...]}`. By default, met variables are standardised; `landcover`, `topog` and the coordinate/distance fields are transformed through minmax; and the disaggregated landcover fractions (`landcover_type_0` … `landcover_type_9`) are not transformed. Passing either list replaces its default (`DEFAULT_MINMAX_VARIABLES` / `DEFAULT_IGNORE_VARIABLES` in `gates/data/datasets.py`).

> **Note:** `HandcraftedInputsScaler` (scaling from pre-computed statistics) is still in development and doesn't yet work from a parameter file. Use `DefaultInputsScaler`.

---

## `fp_scaler`

(required) Configures transformation of the footprint (target) values. Leaving the block out, or setting it to `{}`, currently raises a `NameError`.

```json
"fp_scaler": {
    "scaler": "LogAndShiftFpScaler",
    "scaler_params": {"minimum_oom": 5, "non_negative": true}
}
```

| Field | Description |
|-------|-------------|
| `scaler` | Class name of the footprint scaler, looked up in `gates.data.datasets`. Options are `LogAndShiftFpScaler` and `LogAndShiftMeanFpScaler` (below). If the block has no `scaler` key, `LogAndShiftMeanFpScaler` is used. |
| `scaler_params` | (optional) Additional keyword arguments passed to the footprint scaler constructor. |

Both scalers take the log₁₀ of the non-zero footprint values and leave zeros as zero. They differ in the shift:

- **`LogAndShiftFpScaler`** (used in the templates) adds a fixed order of magnitude, so it needs no fitting and gives the same transform for any domain or size. Its `scaler_params` are:
  - `minimum_oom` (default 5): the order of magnitude added after the log, e.g. a footprint of 1e-5 becomes 0.
  - `non_negative` (default `true`): clips transformed values below zero to zero.
- **`LogAndShiftMeanFpScaler`** shifts by the mean of the log, which it fits on the training footprints. It takes no `scaler_params`.

---

## `flux`

(optional) Controls whether gridded flux/emissions data is loaded and cropped alongside the footprints. When enabled, the cropped flux is appended as a `"flux"` variable on the footprint dataset, making it available to flux-weighted loss functions (e.g. `weight_label: "flux"`) and to flux-based evaluation metrics. Omit this section (or set `get_flux: false`) to skip flux loading entirely.

```json
"flux": {
    "get_flux": true,
    "convert_units": true,
    "convert_units_args": {"unit_multiplier": 1e9}
}
```

| Field | Description |
|-------|-------------|
| `get_flux` | If `true`, loads the flux/emissions data for the domain and years, crops it to match each footprint, and appends it as a `"flux"` variable on the footprint dataset. Defaults to `true` when the `flux` section is present. |
| `convert_units` | If `true`, rescales the flux field via `transform_flux` so its units match the footprint units. If `true` but no `convert_units_args` are passed, a warning is raised and no conversion occurs. Defaults to `false`. |
| `convert_units_args.unit_multiplier` | Constant multiplier applied to the flux field when `convert_units` is `true` (e.g. `1e9`). Defaults to `1` (no change). |
| `year` | (optional) Load the flux for this year instead of the footprint year. Useful when there is no flux file for a footprint year. |
| `append_to_fp` | (optional) If `false`, the flux is loaded but not added to the footprint dataset, so the flux losses and metrics don't see it. Defaults to `true`. |
| `search_others` | (optional) If `true`, looks for the flux in other directories when it isn't in the default one. Defaults to `true`. |

> **Note:** The same `flux` block is reused for both the train and test data loads. A `"flux"` variable on the test outputs is what triggers the extra flux-weighted evaluation metrics. Footprints with no matching flux are dropped.

---

## `dataloader`

(required) Configures the PyTorch `DataLoader` used during training and evaluation.

The templates use the in-memory tensor loader, turned on with [`review_fixes.use_tensor_loader: true`](#review_fixes):

```json
"dataloader": {
    "batch_size": 5,
    "test_batch_size": 5,
    "nans_to_zeros": true,
    "dataloader_params": {
        "pin_memory": true
    }
}
```

| Field | Description |
|-------|-------------|
| `batch_size` | Number of samples per training batch. Defaults to 5. |
| `test_batch_size` | Number of samples per evaluation/test batch. Defaults to 5. |
| `nans_to_zeros` | (do not change) If `true`, replaces NaN values in the input tensor with zeros before passing to the model. Defaults to `true`. |
| `seed` | (optional) Seed for the training loader's shuffle only, overriding the top-level `seed`. Useful for separating the effect of shuffling randomness from model randomness. Used by both the tensor and the xbatcher loader. |
| `dataloader_params` | (optional) Extra keyword arguments passed to the **training** `torch.utils.data.DataLoader`. |
| `dataloader_params.pin_memory` | If `true`, pins loaded tensors to page-locked memory for faster CPU→GPU transfer. |

The tensor loader shuffles the samples into new batches every epoch. It ignores the xbatcher-only keys below if they are present, and writes the params it actually used back to the saved `training_settings_*.json`.

The test loader ignores `dataloader_params`: it always runs with `num_workers: 0` and no prefetching, to avoid a deadlock with the training workers.

### xbatcher loader

Without `review_fixes.use_tensor_loader` (the default, and the only option for the multiregion trainer), the training loader batches the data with `xbatcher`, but as the data is already on memory, there is not a lot of benefit. These are valid keys, but note that the xbatcher dataloader might be deprecated:

```json
"dataloader_params": {
    "prefetch_factor": 3,
    "num_workers": 2,
    "pin_memory": true,
    "persistent_workers": true,
    "multiprocessing_context": "forkserver",
    "shuffle": true
}
```

| Field | Description |
|-------|-------------|
| `dataloader_params.prefetch_factor` | Number of batches loaded in advance by each worker. `0` is converted to `None`, as PyTorch requires with `num_workers: 0`. |
| `dataloader_params.num_workers` | Number of parallel worker processes for data loading. |
| `dataloader_params.persistent_workers` | If `true`, worker processes persist between epochs rather than being respawned. |
| `dataloader_params.multiprocessing_context` | How worker processes are started. Keep `"forkserver"`. Only used when `num_workers` > 0. |
| `dataloader_params.shuffle` | If `true`, shuffles the order of the training batches each epoch. The samples are shuffled in time once before batching, so each batch always holds the same samples. |

These worker settings are meant for cluster runs. When running without workers (e.g. in a terminal, or on a single CPU node), use:

```json
"dataloader_params": {
    "prefetch_factor": 0,
    "num_workers": 0
}
```

---

## `review_fixes`

(optional) Fixes that change the results, so they are off by default to keep older runs reproducible. The templates turn both on.

```json
"review_fixes": {
    "fix_grid_transpose": true,
    "use_tensor_loader": true
}
```

| Field | Description |
|-------|-------------|
| `fix_grid_transpose` | If `true`, builds the model grid in the same (lat, lon) order as the flattened inputs. Recommended for new runs. Defaults to `false`. See the `get_grid` docstring in `gates/data/load_data.py`. |
| `use_tensor_loader` | If `true`, the training loader is an in-memory `TensorDataset` (`make_tensor_dataloader`) that reshuffles the samples into new batches every epoch, instead of the xbatcher loader. Recommended for new runs. Defaults to `false`. Not supported by the multiregion trainer. |

---

## `dynamic_edges`

(optional) Adds features computed from the input data to the mesh edges. Without this key, each mesh edge only carries its fixed geometry: the distance, dlat and dlon between its two mesh nodes. Dynamic edges need the default encoder (`use_dynamic_encoder`, on by default), and can't be combined with `model_parameters.initial_enc` or `concat_enc_neighbours`.

```json
"dynamic_edges": {"dynamic_wind": true, "wind_tuples": [["x_wind", 3, 0],["y_wind", 3, 0]], "dynamic_latlon":true, "dynamic_earthdistance":true}
```

`"dynamic_edges": true` is shorthand for `{"dynamic_wind": true}` with the default wind features.

| Field | Description |
|-------|-------------|
| `dynamic_wind` | bool, whether to add wind features to the edges. Each feature is the mean of its values at the edge's two mesh nodes. Defaults to `true`, so a dict without this key still adds wind. |
| `wind_tuples` | list of lists, with the name of each wind-related feature to be added as features to the edges. Each feature has name in format `(variable_name, level, time_delta)`. By default, the x- and y- wind at level 3 and time delta 0 are appended if `dynamic_wind=True`. |
| `dynamic_latlon` | bool, whether to replace the fixed dlat/dlon edge attributes with ones calculated from the input lat/lon features. These are the scaled inputs, so dlat/dlon are in the scaled space. Defaults to `false`. |
| `latlon_tuples` | list of exactly two lists, `[lat, lon]`, naming the lat and lon input features in the same `(variable_name, level, time_delta)` format. Defaults to `[["lat_coords", 0, 0], ["lon_coords", 0, 0]]`, so `lat_coords` and `lon_coords` must be in `static_variables`. |
| `dynamic_earthdistance` | bool, only valid with `dynamic_latlon=True`. Replaces the fixed edge distance with a haversine distance calculated from the lat/lon features. The lat/lon features are then replaced with random noise in the node inputs, so the model only sees position through the edges. Defaults to `false`. |

---

## `model_parameters`

(required) Defines the GNN architecture. These keys are passed to `GraphSatelliteForecaster` (`gates/model/forecast.py`).

```json
"model_parameters": {
    "num_blocks": 4,
    "node_dim": 64,
    "edge_dim": 64,
    "hidden_layers_processor_node": 2,
    "hidden_layers_processor_edge": 2,
    "hidden_layers_decoder": 1,
    "hidden_dim_processor_node": 16,
    "hidden_dim_processor_edge": 16,
    "hidden_dim_decoder": 16,
    "resolution": 4,
    "output_dim": 1,
    "residuals": false,
    "attention": false,
    "release_edges": false
}
```

> **Note:** The code defaults are much larger than the values above, and `resolution` defaults to 2. Always set the architecture keys explicitly.

| Field | Description |
|-------|-------------|
| `num_blocks` | Number of message-passing rounds in the Processor stage. Code default: 9. |
| `node_dim` | Feature dimensionality of mesh nodes in the Processor. Code default: 256. |
| `edge_dim` | Feature dimensionality of mesh edges in the Processor. Code default: 256. |
| `hidden_layers_processor_node` | Number of hidden layers in the node-update MLP within each Processor block. Code default: 2. |
| `hidden_layers_processor_edge` | Number of hidden layers in the edge-update MLP within each Processor block. Code default: 2. |
| `hidden_layers_decoder` | Number of hidden layers in the Decoder MLP. Code default: 2. |
| `hidden_dim_processor_node` | Hidden dimension of the node-update MLP in the Processor. Code default: 256. |
| `hidden_dim_processor_edge` | Hidden dimension of the edge-update MLP in the Processor. Code default: 256. |
| `hidden_dim_decoder` | Hidden dimension of the Decoder MLP. Code default: 128. |
| `resolution` | H3 mesh resolution controlling hexagon granularity. Resolution 4 is standard (~1–3 grid nodes per hexagon). Lower = coarser. Code default: 2. |
| `output_dim` | Number of output values per grid node. `1` for a single footprint value. Code default: 1. |
| `residuals` | If `true`, each Processor block applies a residual connection: the mesh-node MLP output is added back to the node's own features (`out = node_mlp(...) + x`). Otherwise, the mesh features are completely replaced by the output of the next MLP. Default: `false`. |
| `attention` | If `true`, the Processor's node update becomes masked self-attention over mesh nodes instead of edge-message + scatter-mean aggregation. Each node's new state is a softmax-weighted sum of its graph neighbours' (and its own) value vectors, where the weights come from the node embeddings (single hard-coded head; `node_dim` is the attention embedding dim). The mesh adjacency only supplies the attention *mask* (which pairs may attend); **`edge_attr` is not used**, so wind/distance/release edge features do not reach the node update on this path — treat attention as an *alternative* to message-passing, not an addition. See the note below. Default: `false`. |
| `release_edges` | If `true`, adds a directed edge from the release mesh node (the h3 cell containing the satellite measurement point) to every non-adjacent mesh node. This lets all nodes receive a direct message from the release node's encoded met state (BLH, wind, stability) in the first Processor block, without waiting for multi-hop propagation. A binary `is_release_edge` flag is appended to edge features so the MLP can distinguish these long-range edges from regular k-ring-1 edges. When `dynamic_latlon=True` in `dynamic_edges`, release edge distances and lat/lon offsets are recomputed from actual input coordinates rather than fixed h3 grid geometry. Compatible with all `dynamic_edges` settings. Default: `false`. |
| `initial_enc` | If `true`, applies an initial encoding MLP to each grid node's raw input features *before* they are aggregated onto the mesh nodes, instead of encoding only after aggregation. Cannot be combined with `dynamic_edges`. Default: `false`. |
| `initial_enc_dim` | Output width of the initial encoding MLP (only used when `initial_enc=true`). When unset, it matches the encoder's node-encoder output (the mesh latent dimension), reproducing the previous behaviour. Setting it to a different value gives the two MLPs distinct roles: the initial MLP compresses/transforms each grid node's raw features (`input_dim → initial_enc_dim`) before spatial aggregation, and the node encoder then maps the aggregated mesh-node representation to the final latent space (`initial_enc_dim → node latent dim`). Default: `null` (i.e. matches the node-encoder output). |

> **Note on `attention`.** A few things to know before enabling it: `node_dim` doubles
> as the attention embedding dimension; the attention uses a single head (fixed in code,
> not configurable); `attention` requires `disaggregated: false`; and with
> `release_edges: true` the attention mask includes the release edges, so attention gains
> a long-range path from the release node. Because the node update does not use `edge_attr`,
> combining `attention` with the dynamic wind/distance edges will not help — those edge
> features are inert on the attention path. This is WIP

---

## Training Control

```json
"learning_rate": 5e-5,
"use_wandb": true
```

| Field | Description |
|-------|-------------|
| `learning_rate` | (required) Initial learning rate for the optimiser. |
| `use_wandb` | (required) If `true`, logs metrics and artefacts to Weights & Biases. Requires the `wandb` section below. See [HOW_TO_WandB.md](HOW_TO_WandB.md). |

---

## `wandb`

Weights & Biases run configuration. Only used if `use_wandb` is `true`.

```json
"wandb": {
    "project": "gates-tests-working",
    "entity": "gates-lab",
    "tags": ["SAHARA", "gpu", "PixelWeightedMSELoss", ...]
}
```

| Field | Description |
|-------|-------------|
| `project` | W&B project name under which this run is logged. |
| `entity` | W&B team or user account name. |
| `tags` | List of string tags attached to the run for filtering in the W&B dashboard. |

If `project` or `entity` is missing, a warning is printed and the run continues without W&B.

---

## `epochs`

(required, all four keys) Controls the training schedule and checkpointing.

```json
"epochs": {
    "training": 250,
    "visualize": 30,
    "patience": 100,
    "model_save": 50
}
```

| Field | Description |
|-------|-------------|
| `training` | Total number of training epochs. |
| `visualize` | Frequency (in epochs) at which diagnostic plots are generated and saved to `training_imgs/`. |
| `patience` | Early stopping patience: training stops if validation loss does not improve for this many consecutive epochs. |
| `model_save` | Frequency (in epochs) at which a model checkpoint is saved to disk. |

---

## `loss_functions`

Configures training and evaluation loss functions.

```json
"loss_functions": {
    "criterion": "gates_losses.PixelWeightedMSELoss",
    "criterion_params": {
        "weight_label": "fp_original",
        "transform_fn": "scale_and_shift",
        "w": 1000,
        "a": 1.0
    },
    "criterion_test": "gates_losses.MSELoss",
    "criterion_test_params": {}
}
```

| Field | Description |
|-------|-------------|
| `criterion` | (required) Loss function class used during training, specified as `gates_losses.ClassName` (`gates_losses` is `gates.evaluation.loss_functions`). |
| `criterion_params` | (optional) Keyword arguments for the train loss function. |
| `criterion_test` | (required) Loss function used for evaluation/test metrics. Typically a simpler unweighted MSE. |
| `criterion_test_params` | (optional) Additional keyword arguments for the test loss function. |

For `PixelWeightedMSELoss`:

| Field | Description |
|-------|-------------|
| `criterion_params.weight_label` | The data field used to derive per-pixel loss weights (e.g. the original unscaled footprint). |
| `criterion_params.transform_fn` | The transform function to apply to the `weight_label`. The function `scale_and_shift` scales the data by parameter `w` and shifts it (adds) parameter `a`. |

See [HOW_TO_evaluation.md](HOW_TO_evaluation.md) for the other loss functions.

---

## Other notes
### Written back automatically

The training script adds these keys to the saved `training_settings_*.json`. Don't set them yourself.

| Field | Description |
|-------|-------------|
| `start_time` | Timestamp appended to `model_name`. |
| `num_features` | Number of input channels. `scripts/predict_GATES_model.py` needs it to rebuild the model. |
| `plotted_dates` | The four test dates used for the diagnostic plots in `training_imgs/`. |
| `sweep_id`, `sweep_combination` | Added by `scripts/train_GATES_sweep.py` to each generated config. |

### Keys that are no longer read

Remove these from old parameter files:

- `load_data_monthly`, `imports` and `shortcut` are ignored.
- `load_before_training` (and the old top-level `load_into_memory`) should be left out. The data is always loaded into memory before the dataloaders are built.
- `model_parameters.release_concat` raises a `TypeError`.

### Experimental model parameters

These keys are also accepted into the model parameters section. Most are experimental; leave them at their defaults unless you are testing them.

| Field | Default | Description |
|-------|---------|-------------|
| `n_decoder_neighbours` | 3 | Number of nearest mesh nodes each grid node reads from in the decoder. |
| `decoder_final_layer` | `null` | Final activation of the decoder. |
| `decoder_append_latlon` | `false` | Appends the grid lat/lon to the decoder input. |
| `concat_decoder_neighbours`, `concat_decoder_neighbours_2` | `false` | Concatenate the neighbouring mesh-node features in the decoder instead of taking a weighted mean. |
| `norm_type` | `"LayerNorm"` | Normalisation in the MLPs: `"LayerNorm"`, `"GraphNorm"`, `"InstanceNorm"`, `"BatchNorm"`, `"MessageNorm"` or `null`. |
| `dropout` | 0 | Dropout probability. |
| `use_checkpointing` | `false` | Gradient checkpointing, to reduce GPU memory at the cost of speed. |
| `encode_edges` | `true` | If `false`, skips the edge encoder. |
| `encode_nodes` | `true` | If `false`, skips the node encoder. |
| `scatter` | `"mean"` | How the Processor aggregates the messages arriving at each node. |
| `disaggregated` | `false` | Experimental Processor variant that keeps each incoming edge separate. |
| `use_dynamic_encoder` | `true` | Uses `SatelliteDynamicEncoder`, which supports `dynamic_edges`. `false` uses the older `SatelliteEncoder`. |
| `concat_enc_neighbours` | `false` | Concatenates, rather than averages, the grid-node features arriving at each mesh node. Cannot be combined with `dynamic_edges`. |