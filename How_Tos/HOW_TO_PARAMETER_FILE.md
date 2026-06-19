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
    "parallel_loading": true,
    "load_into_memory": true,
    "notes": "...",
    "load_data_monthly": true,
}
```

| Field | Description |
|-------|-------------|
| `model_name` | Unique identifier for this experiment. |
| `verbose` | If `true`, enables verbose logging output during data loading and training. |
| `parallel_loading` | If `true`, loads data files in parallel. 
(Recommended for cluster jobs, not recommended for terminal jobs) |
| `load_into_memory` | If `true`, loads the dataset into memory. Currently training *will* crash if `false`. |
| `notes` | Free-text field for experiment notes. Saved alongside model artefacts. |
| `load_data_monthly` | Load data for each month separately into memory and concatenate, rather than loading the full dataset at once. Significantly more gentle on memory. Soon, `load_data_monthly:false` will be deprecated. |


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
    "met_args": {
        "met_levels": [3, 9, 15, 21, 30, 42, 51],
        "met_variables": ["x_wind", "y_wind", ...]
    },
    "crop_met": false
}
```

| Field | Description |
|-------|-------------|
| `years` | (required) List of years to include in the training dataset. Must be a list of strings or ints |
| `months` | (optional) List of months to include in the training dataset. By default, it loads all months in a year |
| `freq` | (recommended) Controls sampling frequency by loading every nth sample. Must be a positive int. If not present, `freq` defaults to 1, which might lead to very large data. Increase the frequency to run a smaller subset of the data.|
| `region` | (required) Geographic domain name. Must match a known domain in `config.yml`. |
| `size` | (required) Side length (in grid cells) of the square patch cut around each satellite release point. Must be a positive, *even* int.|
| `met_args.met_levels` | (recommended) Vertical model levels to load from the meteorological data files. |
| `met_args.met_variables` | (recommended) Meteorological variables to load at the data loading stage (before the `variables` section further selects from these). Selecting met levels and variables during data loading reduces computational load. |
| `crop_met` | (do not change) Controls whether to `crop_met` during data loading, which is redundant when running the training pipeline (but is useful when only loading data to examine it) |

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
    "load_into_memory": true
}
```

| Field | Description |
|-------|-------------|
| `met_variables` | (required) Meteorological variables included in the model input array. Can include derived variables such as `wind_speed` and `wind_angle` if `add_wind_direction` is enabled. |
| `met_levels` | (required)  Vertical levels at which meteorological variables are extracted for the input array. Each variable × level combination becomes a separate input channel. |
| `static_variables` | (required) Time-invariant input features appended to each node. Includes coordinate encodings, topography, and land cover. |
| `time_deltas` | (optional) List of time offsets (in hours) for which lagged meteorological fields are included (e.g. `[6, 12]` adds met at t−6h and t−12h alongside t). |
| `add_wind_direction` | (recommended) If `true`, computes `wind_speed` and `wind_angle` from `x_wind`/`y_wind` and adds them to the input array. |
| `load_into_memory` | (required) Determines whether the monthly data is loaded into memory before concantenation. Confusing wrt top-level `load_into_memory`!! |

---

## `input_scaler`

Configures normalisation of meteorological and static input features.

```json
"input_scaler": {
    "scaler": "DefaultInputsScaler",
    "fit_on_subsample": 0.7,
    "scaler_params": {}
}
```

| Field | Description |
|-------|-------------|
| `scaler` | (optional) Class name of the input scaler to use. The default scaler `DefaultInputsScaler` applies `StandardScaler` to met variables and `MinMaxScaler` to static fields. |
| `fit_on_subsample` | (recommended) Fraction of the training data used to fit the scaler. Fraction is selected at random. Speeds up scaler fitting on large datasets. |
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
| `scaler` | Class name of the footprint scaler. Options are `LogAndShiftFpScaler` (applies a log₁₀ transform to non-zero values followed by a shift) and `LogAndSiftMeanFpscaler` (which applies the same log-transform and shift by the mean) |
| `scaler_params` | Additional keyword arguments passed to the footprint scaler constructor. |

**TODO**: add info on scalers
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

## `shortcut`

Connects selected input features directly to the decoder, bypassing the encode–process stage. For each grid node, the chosen features are appended to the aggregated mesh-node representation just before the decoder MLP. This gives the decoder direct access to local information (e.g. wind) that may be diluted or lost during the grid→mesh→grid round-trip.


```json
"shortcut": {
    "shortcut_tuples": [["x_wind", 3, 0], ["y_wind", 3, 0]]
}
```

| Field | Description |
|-------|-------------|
| `shortcut_tuples` | List of `[variable_name, level, time_delta]` triples identifying which input features to pass directly to the decoder. Each triple must match an entry in the assembled input feature list (same format as `wind_tuples` in `dynamic_edges`). The corresponding indices are resolved at runtime from `input_names` and injected into `model_parameters` automatically. |

The resolved `shortcut_indices` are stored in the saved `training_settings` JSON so that inference with `predict_GATES_model.py` reconstructs the model correctly without any extra arguments.

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
| `initial_enc` | (optional) If `true`, applies a small MLP to each grid node's raw features before the scatter-aggregation onto mesh nodes. This gives the model per-node non-linear capacity before the irreversible spatial pooling step. Cannot be combined with `dynamic_edges`. Default: `false`. |
| `initial_enc_dim` | (optional) Output dimension of the initial encoding MLP when `initial_enc` is `true`. Setting this to a value different from `node_dim` gives the two encoder MLPs distinct roles: `initial_encoder` maps `input_dim → initial_enc_dim` per grid node before aggregation, and `node_encoder` maps `initial_enc_dim → node_dim` after aggregation. Defaults to `node_dim` if not set (preserves old behaviour). |

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

---

## `epochs`

Controls the training schedule and checkpointing.

```json
"epochs": {
    "training": 250,
    "visualize": 10,
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