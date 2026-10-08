# How-To: Parameter File Reference

## Overview

The parameter file is a JSON configuration that controls every aspect of a GATES training run — data loading, variable selection, model architecture, scaling, training schedule, and logging. It is passed to `train_GATES_model.py` at runtime:

```bash
python train_GATES_model.py parameter_file.json
```
If your parameter file is not in the `parameter_files_dir` in `config.yml`, you can pass the path as an argument `--file_path /path/to/folder/where/file/is`.

A copy of the resolved parameters is saved to the folder `save_models_dir` in `config.yml`, `save_models_dir/<model_name>_<timestamp>/training_settings_<model_name>_<timestamp>.json` alongside the checkpoint, so each run is fully reproducible.

---

## Top-Level Fields

```json
{
    "model_name": "gatesimports2_SAHARA_dynamic_wind_test",
    "verbose": true,
    "load_before_training": true,
    "notes": "...",
}
```

| Field | Description |
|-------|-------------|
| `model_name` | Unique identifier for this experiment. |
| `verbose` | If `true`, enables verbose logging output during data loading and training. |
| `load_before_training` | If `true` (default), materialises the inputs/footprints into memory before the dataloader is built (crop happens once; each epoch reads RAM). If `false`, data stays lazy and is re-cropped every epoch. The former name `load_into_memory` is still accepted at the top level and treated as `load_before_training`. Not to be confused with `variables.load_into_memory`. |
| `notes` | Free-text field for experiment notes. Saved alongside model artefacts. |
| `grid_node_order` | Order of the nodes in the grid that is handed to the model. `"legacy"` (default, also when the key is absent): the list of `get_grid`, which is longitude-major, while the batches are flattened latitude-major. The model then places the data of cell (lat i, lon j) at the position of cell (lat j, lon i): the window reflected in its diagonal. Every model trained before 2026-09-29 has this geometry, and its saved settings rebuild it. `"latitude_major"`: the same positions in the order of the batches, so every cell's data sits at its own position and the mesh distances and latitude / longitude offsets are the true ones. Applied by `train_dual_headlr_model.py` and `predict_dual_model.py`; `train_dual_model.py` and `train_dual_refit_model.py` stop with an error when it is set; the boundary, pretraining and footprint-only trainers do not read it. Details: `gates/data/grid.py`; tests: `tests/test_grid_node_order.py`. |


> **Note:** Fields prefixed with `_ignore_` (e.g. `_ignore_model_save_dir`) are read but not used by the training script. They can be used to store alternative values without activating them.

---

## `train_load_data`

Controls which data is loaded for training.

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
| `years` | (required) List of years to include in the training dataset. Must be a list of strings or ints |
| `months` | (optional) List of months to include in the training dataset. By default, it loads all months in a year |
| `freq` | (recommended) Controls sampling frequency by loading every nth sample. Must be a positive int. If not present, `freq` defaults to 1, which might lead to very large data. Increase the frequency to run a smaller subset of the data.|
| `region` | (required) Geographic domain name. Must match a known domain in `config.yml`. For the dual-head trainers it may also be a **list of regions of the same domain** that share no footprints: every month is then loaded once per region. Note that the GOSAT-BRAZIL series is part of the GOSAT-SOUTHAMERICA series (`"region": "SOUTHAMERICA"` contains every BRAZIL footprint, see [HOW_TO_LARGE_DATASETS.md](HOW_TO_LARGE_DATASETS.md)). `test_load_data` inherits the training region unless it sets its own `region`. |
| `size` | (required) Side length (in grid cells) of the square patch cut around each satellite release point. Must be a positive, *even* int. The window also sets the model's grid / mesh and the background head's final layer, so a trained model is tied to its `size` (`predict_dual_model.py --override train_load_data.size=... --use-saved-scalers --allow-grid-mismatch` applies a model to another window; only the footprint head transfers). Windows that leave the LPDM domain are padded (footprint NaN → zeroed and masked, met edge values, topography zeros): at 50 x 50 about 1.5 % of the SOUTHAMERICA/BRAZIL windows are affected by ≤ 2 cells, at 100 x 100 22–32 % by up to 27 cells. Host memory of the data pipeline scales with `size`² x samples (≈ 11.7 GiB per 1,000 samples at 50 x 50, i.e. ~4x that at 100 x 100 — see `smoke_tests_dataload/`), so larger windows need a thinner `freq`.|
| `met_args.met_path` | (recommended) Vertical model levels to load from the meteorological data files. |
| `crop_met` | Controls whether to `crop_met` during data loading, which is redundant when running the training pipeline (ie leave as false during training). If you want to crop the data when interacting with it manually, pass as true. |
| `input_domain` | (optional, dual-head trainers with `data_cache` only) Give the model the meteorology of a LARGER window than the footprint it predicts: `{"size": 100, "mode": "full"}`. `size` (even, > the footprint `size`) is the input window in grid cells, centred on the same release point. `mode` `"full"`: every cell of the input window is a node of the model (4x the nodes for 100 x 100; the footprint loss, metrics, plots and exports stay those of the central footprint window, the background head reads the whole input window); `"coarse"`: the input window averaged in (`size` / footprint `size`)² blocks down to the footprint grid replaces the usual inputs (same node count; the node `d` cells from the release point holds the block `2 d` cells away); `"native_and_coarse"`: the usual inputs plus the block-averaged window as extra `ctx_` channels; with `"coarse_sizes": [100, 150]` (this mode only) one block-mean grid per listed window instead — the central 100 x 100 window in 2 x 2 blocks as `ctx100_` channels and the 150 x 150 window in 3 x 3 blocks as `ctx150_` channels (`size` must be the largest entry; entries are multiples of the footprint `size`, strictly increasing). Block wind direction / speed are recomputed from the block-mean components. Inherited by `test_load_data`, part of the month-cache key. Not combinable with `augmentation.mirror_ew`; `"full"` not with `loss_functions.fp_mass_loss`. Details: `gates/data/input_domain.py`; tests `tests/test_input_domain.py`. |

> **Note:  Adding custom paths**: Add your own paths with `"met_args" :{"met_datadir": "/path/to/filename_"}`, `"fp_datadi":"/path/to/filename_"` and `"topog_args":{"topog_path": /path/to/file.nc, "landcover_path": /path/to/file.nc}`


---

## `test_load_data`

Controls which data is loaded for evaluation. Uses the same field names as `train_load_data` but typically a held-out year and different `freq`.

```json
"test_load_data": {
    "years": ["2016"],
    "freq": 100
}
```

| Field | Description |
|-------|-------------|
| `years` | List of years to use as the test/evaluation set. |
| `months` | As above |
| `freq` | Same as in `train_load_data`|

---

## `variables`

Selects which variables are assembled into the model inputs.

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
| `met_levels` | (required)  Vertical levels at which meteorological variables are extracted for the input array. Each variable × level combination becomes a separate input channel. |
| `static_variables` | (required) Time-invariant input features appended to each node. Includes coordinate encodings, topography, and land cover. |
| `time_deltas` | (optional) List of time offsets (in hours) for which lagged meteorological fields are included (e.g. `[6, 12]` adds met at t−6h and t−12h alongside t; `[6, 12, 24, 48]` was used in experiment summary section 37, `[6, 12, 24, 48, 72]` in section 38). Each lag adds one set of met channels (51 for the usual seven variables at seven levels). The month loader reads a margin of meteorology before the first of the month (largest lag + 6 h; December of the previous year's store for January), so the first footprints of a month keep their lagged inputs; before 2026-10-06 they were dropped (2 % of the data at t−18 h, section 24). The lags are part of the month-cache key. |
| `add_wind_direction` | (recommended) If `true`, computes `wind_speed` and `wind_angle` from `x_wind`/`y_wind` and adds them to the input array. |
| `load_into_memory` | Controls how the met is cropped (see note below). If `true`, the **full** meteorology for the **needed timestamps** (whole domain) is loaded into memory before cropping — fast, but high peak memory. If `false`, the met is kept lazy (dask) during cropping: set `chunk_size` (below) to crop in memory-bounded blocks — otherwise the crop is left as a single lazy chunk, which builds a large dask graph and triggers a "large chunk" warning (not recommended)!! |
| `chunk_size` | (recommended when `load_into_memory: false`) Number of footprint samples to crop and load per block. When set (and `load_into_memory` is `false`), the met is read and cropped in blocks of this many samples instead of all at once, keeping peak memory low and avoiding a large-graph warning. Ignored when `load_into_memory` is `true`. Omit to leave the crop fully lazy (currently produces a "large chunk" warning — not recommended). |
---

## `input_scaler`

Configures normalisation of meteorological and static input features.

```json
"input_scaler": {
    "scaler": "DefaultInputsScaler",
    "fit_on_subsample": 1,
    "scaler_params": {}
}
```

| Field | Description |
|-------|-------------|
| `scaler` | (optional) Class name of the input scaler to use. The default scaler `DefaultInputsScaler` applies `StandardScaler` to met variables and `MinMaxScaler` to static fields. |
| `fit_on_subsample` | Fraction of the training data used to fit the scaler. Fraction is selected at random. Speeds up scaler fitting on large datasets. Defaults to 1 (use the whole dataset) |
| `scaler_params` | Additional keyword arguments passed to the scaler constructor. |

Within `DefaultInputsScaler`, the variables that are standardised vs min-max vs no transform can be specified by passing `scaler_params:{"ignore_variables": ["lat_coords", "lon_coords"], "minmax_variables":["land_cover", "topog", "xy_distance_centre, ..."]}`. Currently, the default is that static variables are transformed through minmax, and met variables standardised.


---

## `fp_scaler`

Configures transformation of the footprint (target) values.

```json
"fp_scaler": {
    "scaler": "LogAndShiftFpScaler",
    "scaler_params": {}
}
```

| Field | Description |
|-------|-------------|
| `scaler` | Class name of the footprint scaler. Options are `LogAndShiftFpScaler` (applies a log₁₀ transform to non-zero values followed by a shift) and `LogAndShiftMeanFpScaler` (which applies the same log-transform and shifts by the mean of the log) |
| `scaler_params` | Additional keyword arguments passed to the footprint scaler constructor. |

**TODO**: add info on scalers
---

## `flux`

Controls whether gridded flux/emissions data is loaded and cropped alongside the footprints. When enabled, the cropped flux is appended as a `"flux"` variable on the footprint dataset, making it available to flux-weighted loss functions (e.g. `weight_label: "flux"`) and to flux-based evaluation metrics. Omit this section (or set `get_flux: false`) to skip flux loading entirely.

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

> **Note:** The same `flux` block is reused for both the train and test data loads. A `"flux"` variable on the test outputs is what triggers the extra flux-weighted evaluation metrics.

---

## `dataloader`

Configures the PyTorch `DataLoader` used during training and evaluation.

```json
"dataloader": {
    "batch_size": 5,
    "test_batch_size": 5,
    "nans_to_zeros": true,
    "dataloader_params": {
        "prefetch_factor": 3,
        "num_workers": 2,
        "pin_memory": true,
        "persistent_workers": true,
        "multiprocessing_context": "forkserver",
        "shuffle": true
    }
}
```

| Field | Description |
|-------|-------------|
| `batch_size` | Number of samples per training batch. |
| `test_batch_size` | Number of samples per evaluation/test batch. |
| `nans_to_zeros` | (do not change) If `true`, replaces NaN values in the input tensor with zeros before passing to the model. |
| `dataloader_params.prefetch_factor` | Number of batches loaded in advance by each worker. |
| `dataloader_params.num_workers` | Number of parallel worker processes for data loading. |
| `dataloader_params.pin_memory` | If `true`, pins loaded tensors to page-locked memory for faster CPU→GPU transfer. |
| `dataloader_params.persistent_workers` | If `true`, worker processes persist between epochs rather than being respawned. |
| `dataloader_params.multiprocessing_context` | Multiprocessing start method (e.g. `"forkserver"`). [PLACEHOLDER — describe when to change this] |
| `dataloader_params.shuffle` | If `true`, shuffles the training data each epoch. |

The dataloader params above are set up for cluster runs with multiple workers. When running without them (e.g. in terminal, or in a single CPU node), use
```
        "dataloader_params": {
            "prefetch_factor": 0,
            "num_workers": 0}
```

---

## `data_cache`

Optional, dual-head trainers only. Loads every month once, keeps it on disk and builds the
training batches month by month in shared memory, so large training sets fit in memory and
jobs start within minutes. The data are identical to those of the standard pipeline. Full
description: [HOW_TO_LARGE_DATASETS.md](HOW_TO_LARGE_DATASETS.md).

```json
"data_cache": { "enabled": true }
```

| Field | Description |
|-------|-------------|
| `enabled` | If `true`, use the month cache. Default `false` (standard pipeline, unchanged). |
| `dir` | Root directory of the cache. Default: `user_paths.data_cache_dir` of `config.yml`. |
| `rebuild` | If `true`, reload every month from the archive and overwrite the cache. Default `false`. |
| `threads` | Threads for reading / scaling. Default: min(8, CPUs of the job). |
| `fit_buffer_gib` | Working memory (GiB) of the scaler fit. Default 48. |

---

## `exit_target` / `exit_loss` (exit-curtain model)

Read by `train_exit_model.py` only (ignored by the dual trainers). The exit-curtain model predicts
the particles' exit distribution over the domain boundary instead of a footprint and a background
value; full description: [HOW_TO_EXIT_CURTAINS.md](HOW_TO_EXIT_CURTAINS.md).

```json
"background_setup": {"detrend": true, "use_auxiliary_bc": false},
"exit_target": {"height_factor": 2, "edge_factor": 10, "remainder_bin": true},
"exit_loss": {"bg_weight": 0.0},
"head_learning_rates": {"trunk": 5e-5, "exit": 5e-5}
```

| Field | Description |
|-------|-------------|
| `exit_target.height_factor` | Pool the 20 curtain height levels in blocks of this many levels (default 1). |
| `exit_target.edge_factor` | Pool the along-edge cells (190 north / south, 357 east / west) in blocks of this many cells (default 1); the last block may be short. |
| `exit_target.remainder_bin` | Must be `true`: one extra bin for the particles that never leave the domain, so the target is a probability vector. |
| `exit_loss.bg_weight` | Weight of the implied-background term `((bg_implied - bg_exact) / std(bg))^2` added to the cross-entropy. Default 0 = the training never uses CAMS. |
| `head_learning_rates` | For this trainer the groups are `trunk` and `exit` (not `fp` / `bg`). `head_weight_decay` likewise. |
| `background_setup.use_auxiliary_bc` | Must be `false` for the exit model (no CAMS input channels); part of the month-cache key. |

---

## `augmentation`

Optional; applied by `train_dual_headlr_model.py` only (the other dual trainers stop with an
error if the block is present, instead of ignoring it). Off when the block is absent.

```json
"augmentation": { "mirror_ew": { "probability": 0.5 } }
```

| Field | Description |
|-------|-------------|
| `mirror_ew.probability` | Probability (0–1) with which a training sample is shown **mirrored east–west** about the meridian of the release point. Drawn afresh every epoch from the run's `seed`, the GPU rank, the epoch and the batch number, so a run is reproducible. `0` = off. Test data are never mirrored. |

What is mirrored (details and assumptions: `gates/training/augmentation.py`):

- inputs **and** targets together: every met / static map and every footprint variable is
  reflected along longitude, so the pair is still a valid example;
- `x_wind` and `wind_angle` change sign in physical units (through the fitted scaler);
- `x_coords` / `y_coords` are kept (they describe the grid cell), the east and west auxiliary
  boundary channels are exchanged, the summed background is unchanged;
- the release point is at index `size // 2`, so the window has one more column to the west than
  to the east: the westernmost column of a mirrored sample has no partner, is filled from its
  neighbour and excluded from the footprint loss through `fp_nan_mask` (needs
  `dataloader.nans_to_zeros: true`).

Requirements: inputs **without** absolute longitude channels (`lon_coords`, `sin_lon_coords`,
`cos_lon_coords`), `DefaultInputsScaler`, an even window size. The mirror is exact for the
transport through a given wind field; it is *not* a symmetry of the atmosphere (the turning of
the wind with height, the sense of rotation of weather systems, the position of mountains and
coasts), so it can also take information away. The resolved channel positions and constants are
saved with the run (`augmentation_resolved` in `training_settings_*.json`). Tests:
`tests/test_mirror_augmentation.py`.

---
## `dynamic_edges`

**TODO**
Controls the features that get addded to the mesh graph edges. 

```json
"dynamic_edges": {"dynamic_wind": true, "wind_tuples": [["x_wind", 3, 0],["y_wind", 3, 0]], "dynamic_latlon":true, "dynamic_earthdistance":true}
```

| Field | Description |
|-------|-------------|
| `dynamic_wind` | bool, whether to incorporate wind features to the edges |
| `wind_tuples` | list of lists, with the name of each wind-related feature to be added as features to the edges. Each feature has name in format `(variable_name, level, time_delta)`. By default, the x- and y- wind at level three are appended if `dynamic_wind=True` |
| `dynamic_latlon` | bool, whether to replace the static mesh edge attributes with delta-lat and delta-lon features calculated from the data. Control the name of the lat/lon coords with `latlon_tuples`|
| `dynamic_earthdistance` | bool, only valid `dynamic_latlon=True`. Whether to append the distance between two mesh nodes as attribute. |

---


## `model_parameters`

Defines the GNN architecture. 

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

| Field | Description |
|-------|-------------|
| `num_blocks` | Number of message-passing rounds in the Processor stage. |
| `node_dim` | Feature dimensionality of mesh nodes in the Processor. |
| `edge_dim` | Feature dimensionality of mesh edges in the Processor. |
| `hidden_layers_processor_node` | Number of hidden layers in the node-update MLP within each Processor block. |
| `hidden_layers_processor_edge` | Number of hidden layers in the edge-update MLP within each Processor block. |
| `hidden_layers_decoder` | Number of hidden layers in the Decoder MLP. |
| `hidden_dim_processor_node` | Hidden dimension of the node-update MLP in the Processor. |
| `hidden_dim_processor_edge` | Hidden dimension of the edge-update MLP in the Processor. |
| `hidden_dim_decoder` | Hidden dimension of the Decoder MLP. |
| `resolution` | H3 mesh resolution controlling hexagon granularity. Resolution 4 is standard (~1–3 grid nodes per hexagon). Lower = coarser. |
| `output_dim` | Number of output values per grid node. `1` for a single footprint value. |
| `residuals` | If `true`, each Processor block applies a residual connection: the mesh-node MLP output is added back to the node's own features (`out = node_mlp(...) + x`). Otherwise, the mesh features are completely replaced by the output of the next MLP. Default: `false`. Available in both model packages since 2026-09-26: the runtime `model/` package used by `train_dual_*` / `train_boundary_model` got the same residual update (branch `dual_pred_model_changes`; before that its `residuals: true` enlarged the footprint decoder's expected input and failed with a shape mismatch). Recommended whenever `num_blocks` is raised above 4 (experiment summary section 17). |
| `attention` | If `true`, replaces scatter-mean message aggregation in the Processor with multi-head self-attention over mesh nodes. Default: `false`. NEEDS TESTING|
| `release_edges` | If `true`, adds a directed edge from the release mesh node (the h3 cell containing the satellite measurement point) to every non-adjacent mesh node. This lets all nodes receive a direct message from the release node's encoded met state (BLH, wind, stability) in the first Processor block, without waiting for multi-hop propagation. A binary `is_release_edge` flag is appended to edge features so the MLP can distinguish these long-range edges from regular k-ring-1 edges. When `dynamic_latlon=True` in `dynamic_edges`, release edge distances and lat/lon offsets are recomputed from actual input coordinates rather than fixed h3 grid geometry. Compatible with all `dynamic_edges` settings. Default: `false`. Runtime `model/` package (dual trainers): implemented in `SatelliteEncoder.create_mesh_graph` since 2026-09-26 with the same semantics (directed edges from the release mesh node to every non-adjacent mesh node, `[distance, dlat, dlon, is_release_edge]` attributes, so the mesh edge encoder input grows from 3 to 4); the earlier version pointed the edges into the release node with wrong attributes. |
| `num_mesh_levels` | Dual-head / runtime `model/` package (since 2026-10-04). `1` (default): the single H3 mesh as before. `k > 1` adds GraphCast-style long-range edges: for each of the `k-1` coarser H3 levels (resolution-1, resolution-2, ...), one mesh node represents each coarse cell (the fine node nearest its centre) and the representatives of adjacent coarse cells are connected, with the usual edge attributes. The node set, the grid-to-mesh edges and the parameter count are unchanged; a processor block can then pass information ~104 km (level 2) or ~274 km (level 3) per hop instead of ~39 km (resolution 4). Cannot be combined with `higher_mesh_res`. Experiment summary section 32. |
| `initial_enc` | If `true`, applies an initial encoding MLP to each grid node's raw input features *before* they are aggregated onto the mesh nodes, instead of encoding only after aggregation. Cannot be combined with `dynamic_edges`. Default: `false`. |
| `initial_enc_dim` | Output width of the initial encoding MLP (only used when `initial_enc=true`). When unset, it matches the encoder's node-encoder output (the mesh latent dimension), reproducing the previous behaviour. Setting it to a different value gives the two MLPs distinct roles: the initial MLP compresses/transforms each grid node's raw features (`input_dim → initial_enc_dim`) before spatial aggregation, and the node encoder then maps the aggregated mesh-node representation to the final latent space (`initial_enc_dim → node latent dim`). Default: `null` (i.e. matches the node-encoder output). |
| `decoder_skip` | Dual-head model only (runtime `model/` package, since 2026-09-29). Grid-level skip connection into the **footprint** decoder. `false` (default): every lat/lon cell is decoded from the blend of its three nearest mesh nodes alone, so the output cannot hold detail finer than the mesh (at resolution 4 there are about half as many mesh nodes as grid cells). `"encoded"` (or `true`): the cell's own met / static input features are passed through an MLP of the decoder's size and joined to that blend before the decoder MLP. `"inputs"`: the features are joined as they are. The auxiliary boundary channels are not part of the skip (they are the same for every cell of a sample); the background head is unchanged. Not available with `concat_decoder_neighbours`. Tests: `tests/test_decoder_skip.py`. |
| `exit_head_hidden_dim` | Exit-curtain model only (`train_exit_model.py`, since 2026-10-05). `0` (default): the head's flattened feature map is mapped to the output logits by one linear layer, as in the dual model's background head. `> 0`: a hidden layer of this width is inserted (`flattened -> exit_head_hidden_dim -> n_outputs`). See [HOW_TO_EXIT_CURTAINS.md](HOW_TO_EXIT_CURTAINS.md). |

---

## Training Control

```json
"learning_rate": 5e-5,
"use_wandb": true
```

| Field | Description |
|-------|-------------|
| `learning_rate` | Initial learning rate for the optimiser. |
| `use_wandb` | If `true`, logs metrics and artefacts to Weights & Biases. Requires the `wandb` section below. See `HOW_TO_WandB.md`. |

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

| `mode` | Optional. `"offline"` writes the run to `WANDB_DIR` without talking to api.wandb.ai (upload later with `wandb sync wandb/wandb/offline-run-*`); `"online"` forces online; absent = the `WANDB_MODE` environment variable / W&B default. Added 2026-10-05 after an online run's final upload blocked a 4-GPU job for > 1.5 h; all three dual trainers and the data-loading summary run of `run_dual_experiments_shared_data.py` honour it. |
---

## `epochs`

Controls the training schedule and checkpointing.

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
| `criterion` | Loss function class used during training, specified as `module.ClassName`. |
| `criterion_params` | Keyword arguments for the train loss function. |
| `criterion_test` | Loss function used for evaluation/test metrics. Typically a simpler unweighted MSE. |
| `criterion_test_params` | Additional keyword arguments for the test loss function. |

For a specific loss function:
| `criterion_params.weight_label` | The data field used to derive per-pixel loss weights (e.g. the original unscaled footprint). |
| `criterion_params.transform_fn` | The transform function to apply to the `weight_label`. function  `scale_and_shift` scales the data by parameter `w` and shifts it (adds) parameter `a`|


---

## See Also

- [HOW_TO_CONFIG.md](HOW_TO_CONFIG.md) — setting up `config.yml` with data paths
- [HOW_TO_DATA.md](HOW_TO_DATA.md) — data pipeline reference
- [HOW_TO_WandB.md](HOW_TO_WandB.md) — Weights & Biases logging
- [HOW_TO_evaluation.md](HOW_TO_evaluation.md) — evaluation workflow
- [HOW_TO_LARGE_DATASETS.md](HOW_TO_LARGE_DATASETS.md) — month cache for large training sets (`data_cache`)