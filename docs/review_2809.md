# Documentation Review — 2026-09-28

Review of the GATES documentation: the to-do lists that already exist, new pages to add, updates to existing pages, and code bugs that break the documented usage.

The to-do lists, README and docs index were read directly. The checks of each How_To against the code were done by reading the code, not by running it. These claims were re-checked by hand: the scaler argument name, the land-cover name, the checkpoint filenames and `__all__`. The other claims have not been run or re-checked.

## Status

Legend: ⬜ Not started · 🟨 In progress · ✅ Done · ⏸️ Blocked (waiting on a code fix)

| Section | Item | Status |
|---|---|---|
| **1. Existing to-do lists** | docs_todo.md: `.gitignore` for build dirs | ⬜ |
| | docs_todo.md: landing page (`index.rst`) | ⬜ |
| | docs_todo.md: check the API module summaries | ⬜ |
| | docs_todo.md: navbar polish | ⬜ |
| | docs_todo.md: GitHub Actions → `gh-pages` | ⬜ |
| | HANDOFF.md: "Scalers branch" (empty heading) | ⬜ |
| | TODO markers in the docs (README, PARAMETER_FILE, TESTS, main_functions, model_description) | ⬜ |
| **2. New pages** | 2.1 Model architecture | ⬜ |
| | 2.2 Dynamic edges | ⬜ |
| | 2.3 Main functions | ⬜ |
| | 2.4 Outputs & checkpoints | ⬜ |
| | 2.5 Scalers | ⏸️ (bug 4.1) |
| | 2.6 Post-processing, bias correction & inversion | ⬜ |
| | 2.7 Experimental flags | ⬜ |
| | 2.8 HPC data transfer / launchers | ⬜ |
| | 2.9 Landing page | ⬜ |
| | 2.10 Notebooks index | ⬜ |
| **3. Updates to existing pages** | 3.1 README.md | ⬜ |
| | 3.2 HOW_TO_PARAMETER_FILE.md | ⬜ |
| | 3.3 HOW_TO_DATA.md | ⬜ |
| | 3.4 HOW_TO_LAUNCH.md | ⬜ |
| | 3.5 HOW_TO_evaluation.md | ⬜ |
| | 3.6 HOW_TO_WandB.md | ⬜ |
| | 3.7 HOW_TO_CONFIG.md | ⬜ |
| | 3.8 tests/HOW_TO_TESTS.md | ⬜ |
| | 3.9 Sphinx / API reference | ⬜ |
| | 3.10 CLAUDE.md | ⬜ |
| **4. Code bugs** | 4.1 Handcrafted scaler `stats` vs `stats_file` | ⬜ |
| | 4.2 `land_cover` vs `landcover` in the MinMax list | ⬜ |
| | 4.3 `--checkpoint <int>` filename mismatch | ✅ |
| | 4.4 `gates.__all__` lists a removed function | ✅ |
| | 4.5 Other smaller bugs | ⬜ |

**Suggested order:** Start with bugs 4.1–4.4, which make the documented usage fail. Then fix README.md and CLAUDE.md (3.1, 3.10), because newcomers read them first. Then write the architecture page and the outputs & checkpoints page (2.1, 2.4).

---

## 1. Existing to-do lists

### [docs/docs_todo.md](docs_todo.md)

- [ ] Add `docs/build/` and `docs/source/_autosummary/` to `.gitignore`.
- [ ] Write real landing-page content for `index.rst`. It still has the sphinx-quickstart placeholder text.
- [x] Wire the `How_Tos/` guides into the docs navigation (via `myst_parser`).
- [ ] Check the one-line module summaries in `source/api/` against what each file does.
- [ ] Navbar polish: GitHub link/icon, logo (`html_theme_options` in `conf.py`).
- [ ] Set up a GitHub Actions workflow to build and publish to `gh-pages`.

### [HANDOFF.md](../HANDOFF.md)

- Pending tasks: "(none)". The rest of the file is a log of completed work: the v2 loader migration, the single-month fast path in `load_GATES_data_v2`, and the multiregion clean-up.
- "To think about in the future → Scalers branch" is an empty heading with no content.

### TODO markers inside the docs

- [README.md:136](../README.md): "Inversion - TO DO!" is an empty section.
- [README.md:133](../README.md): predicting on a different size or domain is "still not implemented".
- [HOW_TO_PARAMETER_FILE.md](../How_Tos/HOW_TO_PARAMETER_FILE.md):
  - :153 "TODO: add info on scalers"
  - :207 "[PLACEHOLDER]"
  - :220 "TODO" under `dynamic_edges`
  - :286 the attention how-to is "coming soon"
- [tests/HOW_TO_TESTS.md:38, :96](../tests/HOW_TO_TESTS.md): "TODO: Make these importable from config!"
- [docs/source/user_guide/main_functions.md](source/user_guide/main_functions.md): "To be written."
- [model_description.md](../model_description.md): "I have made some changes I will detail here at some point".

---

## 2. New pages to add

### 2.1 Model architecture

This replaces the 10-line `model_description.md`.

- **Encoder** (`gates/model/layers/encoder.py`):
  - Each grid point maps to its h3 cell via `geo_to_h3` (182–198).
  - The grid→mesh edge attribute is the haversine distance (245).
  - Mesh edges come from `k_ring(1)` (427–467).
  - `SatelliteEncoder` (72) vs `SatelliteDynamicEncoder` (472), switched by the `use_dynamic_encoder` flag (`forecast.py:57,156`). The flag is not mentioned anywhere in the docs.
  - Experimental options: `higher_res`, `better_meshnodes` (283, 397), `concat_enc_neighbours`, `initial_enc`, `v2_edges`, and the release-node edges (170, 631).
- **Processor** (`processor.py:18`, `graph_net_block.py`):
  - Block variants: `NodeSatelliteProcessor`, `...Disaggregated` (550, which assumes 7 incoming edges) and `...Attention` (750).
  - Scatter options: `scatter_cat` / `scatter_cat_v2`.
  - The `scatter` and `disaggregated` kwargs are undocumented.
- **Decoder** (`decoder.py:71`):
  - kNN to the nearest `n_neighbours` mesh cells, with distance weighting (197–208).
  - Options: `concat_neighbours` (and `_2`), `append_latlon`, `final_activation`.
  - `residuals` exists but is unused.
- **h3 version:** the code uses the h3 v3 API (`geo_to_h3`, `k_ring`), so it needs `h3-py=3.7`. The docs don't say this.
- **Default resolution:** `GraphSatelliteForecaster` defaults to `resolution=2` (`forecast.py:29`), but the docs say 4 is standard. Only the config value makes it 4.
- **Undocumented model parameters:** add a table for the `model_parameters` keys that HOW_TO_PARAMETER_FILE doesn't cover:
  - `n_decoder_neighbours`, `concat_decoder_neighbours(_2)`, `decoder_append_latlon`, `decoder_final_layer`
  - `scatter`, `disaggregated`, `norm_type`, `dropout`
  - `encode_edges`, `encode_nodes`, `higher_mesh_res`, `use_checkpointing`
  - `use_dynamic_encoder`, `concat_enc_neighbours`, `release_coords`, `batchsize`
- Fix the absolute `/readme_imgs/...` image paths.

### 2.2 Dynamic edges

This fills the TODO at HOW_TO_PARAMETER_FILE:218–235. At the moment the mechanics are only described in two docstrings: `setup_dynamic_edges` (`training/training.py:127`) and `SatelliteDynamicEncoder` (`encoder.py:472–555`). The page should cover:

- the mean wind at the two endpoints of each edge
- dlat/dlon computed in normalised space
- `latlon_tuples`
- `dynamic_earthdistance`, which swaps the lat/lon features for noise (752–761)
- the `"dynamic_edges": true` shorthand vs the dict form

### 2.3 Main functions

This fills the stub at [source/user_guide/main_functions.md](source/user_guide/main_functions.md) with a walkthrough of the functions used day to day.

- **Config:**
  - `gates.config.setup(platform="local")` (`config.py:45`)
  - `get_config()` (14)
  - `Config(filename="config.yml")` (95)
- **Data loading:**
  - `LoadSquareSatelliteData(year, region="BRAZIL", month=None, domain=None, size=10, freq=1, freq_offset=0, fill_outofdomain_with="nans", delete_outofdomain=False, sampling_mode="regular", fp_datadir=None, lazy_load=True, met_args={}, topog_args={}, cfg=None, crop_met=True, load_fps_in_mem=True, ...)` (`load_data.py:1044`)
  - `load_GATES_data_v2(data_parameters, input_variables, datapath_args={}, flux_args=None, verbose=True, load_into_memory=False, use_wandb=False, wandb_state=None)` (`training.py:248`). This is what the scripts actually call.
- **Inputs:** `get_square_satellite_inputs_v2(data, met_variables, met_levels, time_deltas=None, static_variables=None, add_timedelta_zero=True, add_wind_direction=False, load_into_memory=False, interp_to=None, chunk_size=None)` (`datasets.py:1735`)
- **Scaling:**
  - `InputsDataset(inputs, scaler=None, fit_on_subsample=1, scaler_params={}, compute=True, seed=34)` (354)
  - `DefaultInputsScaler(minmax_variables=[...], ignore_variables=[])` (471)
  - `HandcraftedInputsScaler(stats)` (644)
  - `FootprintDataset(fp, scaler=None, scaler_params={}, add_nan_mask=False, keep_vars=True)` (959), with `.inverse_transform`, `.save_scaler` and `.load_scaler`
- **Loaders:**
  - `make_dataloader(inputs, fps, batch_size=10, randomize=False, random_seed=42, dataloader_params=None, flatten=False)` (1292)
  - `make_tensor_dataloader(...)` (1413)
  - `setup_GATES_dataloaders(parameters, train_inputs, train_fps, test_inputs, test_fps)` (`training.py:524`)
- **Grid and model:**
  - `get_grid(fp_xr, reference_fp=0, fix_transpose=False)` (`load_data.py:2280`)
  - `GraphSatelliteForecaster(lat_lons, resolution, feature_dim, output_dim, node_dim, edge_dim, num_blocks, ...)` (`forecast.py:25`)
  - `setup_dynamic_edges(...)` (`training.py:127`)
  - `setup_GATES_model(parameters, training_ctx, paths_ctx)` (734)
- **Evaluation:**
  - `compute_footprint_metrics` (`metrics.py:408`)
  - `iou_at_quantiles` (78)
  - `compute_metrics_by_threshold` (455)
  - `calculate_mfs` (628)
  - `compute_static_mf_metrics` (775)
  - `compute_flux_metrics` (884)
  - `threshold_fps` (`post_processing.py:4`)
- **Plotting:** `plot_fp_predictions(prediction_ds, idxs_list, ..., which_dataspace="original")` (`plotting_predictions.py:145`)
- **Utilities:** `load_parameter_file(file_path)` and `set_reproducibility(seed=None)` (`training_helperfuns.py:36, 124`)

### 2.4 Outputs & checkpoints

**Training** writes to `<save_models_dir>/<model_name>_<YYYYmmdd_HHMMSS>/`:

- `<name>_updates.txt`: plain-text epoch log.
- `training_outputs/scalers_<name>.pickle`: a dict of the fitted scalers, including `fp_scaler` and `input_names`.
- `training_outputs/training_settings_<name>.json`: the resolved parameters, plus `start_time`, `plotted_dates` and `num_features` (predict needs `num_features`).
- `training_outputs/grid_<name>.pickle`: a list of (lat, lon) tuples.
- `<name>_<epoch>.pt`: a dict with `epoch`, `model_state_dict`, `optimizer_state_dict`, `loss` and `learning_rate`.
- `<name>_best.pt`: a bare `state_dict` saved by EarlyStopping, so its format differs from the epoch checkpoints.
- `training_imgs/<name>_<epoch>.png`: a 4×4 comparison grid.
- `sample_predictions_test.nc`: contains `fp_original`, `fp_transformed`, `fp_pred` and `fp_transformed_pred`, plus `fp_nan_mask` and `flux` when present. Multiregion runs write one `sample_predictions_test_<idx>-<region>.nc` per region.
- When W&B is on, each of these is also logged as a W&B artifact.

**Prediction** writes to `{save_path}/{model_save_name}[_REGION][_sizeN]/`:

- `predictions_{year}_{MM}.nc`, or `..._DRYRUN.nc` for a dry run. It contains `fp_original`, `fp_transformed`, `fp_pred`, `fp_transformed_pred`, `lat_coords`, `lon_coords` and optionally `fp_nan_mask`.
- `run_record_<ts>.json`: a record of the command-line arguments used.

**How to reload a model:** the page should also explain loading a checkpoint and the saved scalers.

### 2.5 Scalers — ⏸️ blocked on bug 4.1

The page should cover:

- `DefaultInputsScaler` (`minmax_variables`, `ignore_variables`), `HandcraftedInputsScaler` and `GhostScaler`
- the format of the stats JSON
- where the example JSONs actually live: `gates/data/{handcrafted_scaler,handcrafted_scaler_test,multiregion_scaler}.json` (not `scaler_files/`)
- footprint scalers: `LogAndShiftMeanFpScaler` (the default) and `LogAndShiftFpScaler` (`minimum_oom`, `non_negative`)

It should also fill the "Scalers branch" note in HANDOFF.md.

### 2.6 Post-processing, bias correction & inversion

- `gates/evaluation/post_processing.py` contains only `threshold_fps`, and no user doc mentions it.
- Bias correction and integration back to the full domain exist only in the untracked legacy script `integrate_fps_NEW.py`, which:
  - imports `model.evaluation2005.quantile_mapping_interp`
  - has hard-coded paths
  - writes `{emulated_footprints_path}{year}{month}.nc` with a `bias_corrected` attribute
- This could start as a "current status" page and then fill the README "Inversion" section.

### 2.7 Experimental flags

- The `review_fixes` block:
  - `fix_grid_transpose` (`train_GATES_model.py:459`; see the `get_grid` docstring)
  - `use_tensor_loader`, which switches to `make_tensor_dataloader`
- `grid_reference_fp`, which has only one line in README:78.
- For each flag, say whether it is recommended.

### 2.8 HPC data transfer / launchers

- `scripts/transfer_met_data.sh`: transfers Jasmin → Isambard zarr, configured through env vars (lines 20–25).
- `launch/launch_experiment.sh`: a parametrised `sbatch <JOB> <PARAM_JSON>` launcher.
- The Dask cluster: `make_cluster` (`training.py:802`) sizes it from SLURM env vars.

### 2.9 Landing page (`index.rst`)

- What GATES is. The scientific context section of CLAUDE.md is a good starting point.
- An architecture diagram.
- A quickstart.
- A citation (GMD 2026, `CITATION.cff`).

### 2.10 Notebooks index

- [notebooks/README.md](../notebooks/README.md) is actually a note on the NIWA/MethaneSAT data, and it is untracked. Move that note elsewhere.
- Replace it with a description of `data_tutorial.ipynb` and `plotting_results.ipynb`. Both notebooks are thin at the moment: `plotting_results` describes itself as "a bit hacky".

---

## 3. Updates to existing pages

### 3.1 [README.md](../README.md)

- [ ] :8–26: the file tree is wrong.
  - The scripts are in `scripts/`.
  - `HOW_TO_DATA.md` is in `How_Tos/`, not `gates/data/`.
  - `gates/model`, `launch/` and `tests/` are missing.
- [ ] :34: the link to `HOW_TO_BOUNDARIES.md` is broken; the file doesn't exist.
- [ ] :40: the env link `.env_gates_pytorch.yml` should be `env_gates_pytorch.yml`.
- [ ] :65: the platform is spelled `isambard_ai` here but `isambard-ai` in HOW_TO_CONFIG.
- [ ] :78: a stray code line (`grid, _ = get_grid(...)`) is pasted into the prose.
- [ ] :82: the named template `NEW_parameter_template_gpu_new3b.json` is untracked and crashes (bug 4.1).
- [ ] :102–106: the commands should be `python scripts/train_GATES_model.py` and `sbatch launch/launch_train.sh`.
- [ ] :109–124: the output tree is wrong.
  - The folder is timestamped.
  - `grid_*`, `training_settings_*` and `scalers_*.pickle` are in `training_outputs/`. The scalers file is not called `transform_parameters_*`.
  - `_best.pt` and `sample_predictions_test.nc` are missing.
- [ ] :129: the predict command line is out of date. It is now `--test_year Y --reference_model NAME ...`.
- [ ] :136: the "Inversion" section is empty (see 2.6).
- [ ] HOW_TO_LAUNCH.md and tests/HOW_TO_TESTS.md are not linked from the list or from "See Also".

### 3.2 [HOW_TO_PARAMETER_FILE.md](../How_Tos/HOW_TO_PARAMETER_FILE.md)

- [ ] :7: should be `scripts/train_GATES_model.py`.
- [ ] :12: the settings file is saved in `training_outputs/`.
- [ ] :24: a trailing comma makes the JSON invalid.
- [ ] :60: `met_args.met_path` is wrong.
  - The keys are `met_datadir`, `met_levels` and `met_variables`.
  - v2 fills levels and variables from `variables` automatically.
- [ ] :63: typo `fp_datadi`. The top-level `data_dirs` override is not documented.
- [ ] :61: `crop_met` defaults to `True` in the code.
- [ ] :70: say that `test_load_data` is merged on top of `train_load_data`.
- [ ] :109: `add_wind_direction` is described as "recommended", but it defaults to `False`.
- [ ] :132: the land-cover MinMax claim is wrong (bug 4.2).
- [ ] :150: document the default fp scaler and the `LogAndShiftFpScaler` params.
- [ ] :174: flux also accepts `year`, `append_to_fp` and `search_others`.
- [ ] :210–215: the test loader ignores `dataloader_params` and hardcodes `num_workers=0`.
- [ ] :241–272: the code defaults differ from the example: `num_blocks` 9, dims 256, `resolution` 2. `output_dim` must be set to 1.
- [ ] Mark the required keys:
  - `verbose`, `use_wandb`, `dataloader`, `test_load_data`, `learning_rate`
  - all four `epochs.*` keys
  - `loss_functions.criterion` / `criterion_test`
- [ ] Add the keys that are missing:
  - **Top level:** `seed` (default 34), `model_save_dir`, `data_dirs.*`, `grid_reference_fp`, `review_fixes.*`, `__sweep__`, `regions`, `shared_load_parameters`.
  - **`train_load_data`:** `year` (single value or glob, e.g. `"201[4-5]"`), `month`, `domain`, `freq_offset`, `sampling_mode`, `fill_outofdomain_with`, `delete_outofdomain`, `load_fps_in_mem`, `topog_args`, `fp_datadir`.
  - **`variables`:** `interp_to`, `add_timedelta_zero`.
  - **`dynamic_edges`:** `latlon_tuples`.
  - **`model_parameters`:** see 2.1.
  - **Written back automatically:** `start_time`, `num_features`, `plotted_dates`, `sweep_id`, `sweep_combination`.
- [ ] Remove the keys that are no longer read, or that cause errors:
  - `parallel_loading`, `load_data_monthly`, `imports` and `shortcut` are no longer read.
  - `model_parameters.release_concat` causes a TypeError.
- [ ] Fill the TODOs at :153, :207, :220 and :286.

### 3.3 [HOW_TO_DATA.md](../How_Tos/HOW_TO_DATA.md)

- [ ] :24–28: the example `bp` paths don't match `config_defaults.yml`.
  - The defaults use `met_datadir: /met_archive/UM/`, not `/zarr_store/`.
  - The defaults include a `flux_datadir`, which the example lacks.
- [ ] :70–77: the table says INDIA → SOUTHASIA and CHINA → EASTASIA, but `config_defaults.yml` has `"INDIA"` / `"CHINA"`. The defaults are probably the ones that are wrong.
- [ ] :99: the "hardcoded" list of bad files now lives in `config.yml` under `bad_fp_files`.
- [ ] :102, :111: `data.met` exists only when `crop_met=True`, and training sets `crop_met: false`.
- [ ] :120–122: the v1 inputs function has been removed, not just deprecated.
- [ ] :159, :294: `scaler_params={"fit_on_subsample": ...}` raises a TypeError. `fit_on_subsample` is an argument of `InputsDataset`.
- [ ] :166, :171: the land-cover MinMax claim is wrong (bug 4.2).
- [ ] :185–199: `stats_file`, the `"ghost"` type and the fallback don't exist in the class that's imported, and `scaler_files/` doesn't exist.
- [ ] :231: the batch shape is `(B, HW, V)` when `flatten=True`, which is what training uses.
- [ ] :254, :308: the variable is called `train_loader`.
- [ ] Delete `How_Tos/HOW_TO_DATA copy.md`. It is an older, untracked duplicate.

### 3.4 [HOW_TO_LAUNCH.md](../How_Tos/HOW_TO_LAUNCH.md)

- [ ] :17–43: the env name is `gates_env` in the doc but `new_gates_env` in `launch/launch_train.sh`. There is also a stale root `launch_train.sh` (tracked) that still runs `python train_GATES_model.py`.
- [ ] :55: says training "can be resumed", but there is no resume path: `epoch_so_far` is always 0.
- [ ] :94, :134: the sweep paths are wrong.
  - The default sbatch script path and the working directory are wrong.
  - Logs go to `slurms/`, not `launch/logs/`.
- [ ] :152–164: the `--size` flag is missing from the table. It is also only a placeholder and is ignored.
- [ ] :164: document the `_DRYRUN` suffix and `run_record_<ts>.json`.
- [ ] :169: the output variables are `lat_coords` / `lon_coords` and `fp_nan_mask`. Also say that predict always uses `freq=1` and loads no flux.
- [ ] :171: the region suffix is added whenever `--region` is passed, even if it matches training. The size suffix is `_size{N}`.
- [ ] :236: the per-region file is named `sample_predictions_test_<idx>-<region>.nc`.
- [ ] Multiregion:
  - it uses a different W&B key scheme (`metrics_original/<region>/...`, `/aggregate/`)
  - it logs images every 3×`visualize` epochs
  - it does not log `metrics_fluxes`
  - `shared_load_parameters` are merged only into `train_load_data`

### 3.5 [HOW_TO_evaluation.md](../How_Tos/HOW_TO_evaluation.md)

- [ ] :11: typo "datatasets".
- [ ] :88: `cut_flux_data` returns `(cropped_flux, nan_idxs)`.
- [ ] :104: `calculate_mfs` is called without its module prefix.
- [ ] :122: `checkerboard_50` only appears when H and W are over 100.
- [ ] :148: the flux advice is inaccurate; the `flux` block appends the flux automatically.
- [ ] :162–163: the variable names don't match (`loss_fn` vs `criterion`).
- [ ] :167: the link `../data/datasets.py` is broken; it should be `../gates/data/datasets.py`.
- [ ] :180–186: add `GradientMSELoss` (`loss_functions.py:763`), `StructuralLoss` (:902) and their pixel weighting (`weight_label` / `weight_transform_fn`).
- [ ] Add `iou_at_quantiles`, which is logged as `iou_q25` … `iou_q99`.
- [ ] Explain how `criterion` strings are evaluated as `gates_losses.X`, and how `nans_to_zeros` adds `nan_mask_label`.
- [ ] Document the exact set of metrics that `calculate_losses` logs:
  - original-space IoU at 1e-5
  - transformed-space metrics
  - static mole-fraction metrics
  - flux metrics

### 3.6 [HOW_TO_WandB.md](../How_Tos/HOW_TO_WandB.md)

The page is accurate for single-region runs. Add:

- [ ] the `loading/*` metrics
- [ ] the `training_plots` image key
- [ ] the multiregion key scheme
- [ ] that runs use `save_code=True`
- [ ] that `wandb` is missing from `env_gates_pytorch.yml`, although every training script imports it

### 3.7 [HOW_TO_CONFIG.md](../How_Tos/HOW_TO_CONFIG.md)

- [ ] :22: the platform name `isambard-ai` vs `isambard_ai` is inconsistent.
- [ ] :51: `flux_datadir` is required: `config.py:185` raises a KeyError if it is missing.

### 3.8 [tests/HOW_TO_TESTS.md](../tests/HOW_TO_TESTS.md)

- [ ] All relative links break, because they use `tests/...` from inside `tests/`. Affected lines: 24, 25, 34, 52, 75, 87, 102, 138, 146.
- [ ] :31, :49–54: the sample met is described as monthly NetCDF with `model_level_number`, but the loader now only reads yearly Zarr. The `--sample-files` mode and `generate_sample_data.py` are probably stale.
- [ ] TODOs at :38 and :96.

### 3.9 Sphinx / API reference

- [ ] No API pages exist for `gates.data.handcrafted`, `gates.data.load_receptor_data` (an empty file) or `gates.evaluation.loss_functions_static`. All three are untracked, so either commit and document them or delete them.
- [ ] The top-level `automodule` directives for `gates`, `gates.data` and `gates.plotting` are commented out.
- [ ] The scripts in `scripts/` are not in the Sphinx docs.
- [ ] The docs `{include}` the How_Tos, so the How_Tos' relative links also break in the built site.
- [ ] `gates/utils/config_defaults.yml` is not described anywhere.
- [ ] Worst docstrings:
  - [ ] `training/training_dataclasses.py`: every field of `PathContext`, `TrainingContext` and `ModelContext` is `<FILL IN>`.
  - [ ] One-line class docstrings: `GraphSatelliteForecaster` (`forecast.py:23`), `SatelliteDecoder` (`decoder.py:71`), `SatelliteProcessor` (`processor.py:18`), `MLP` (`graph_net_block.py:153`).
  - [ ] Many forecaster kwargs are documented only as "Forwarded to the encoder" (`forecast.py:112–135`).
  - [ ] `<FILL IN>` placeholders:
    - `processor.py:54–56, 82`
    - `decoder.py:118, 133`
    - `encoder.py:126, 130, 136`
    - `forecast.py:122`
    - `datasets.py:166`
    - `load_data.py:2169`
    - `graph_net_block.py:287, 307, 420, 440, 773, 781`
    - `plotting_predictions.py:53, 167`
    - `training_helperfuns.py:327`
  - [ ] Short or missing docstrings in `predict_GATES_model.py`: `load_checkpoint`, `find_model_dir`, `run_inference`.

### 3.10 [CLAUDE.md](../CLAUDE.md)

- [ ] The predict command line is out of date. It is now `--test_year --reference_model`, not a `training_settings_*.json` argument with a `reference_model` block.
- [ ] `integrate_fps.py` doesn't exist. Only the untracked `integrate_fps_NEW.py` and `integrate_fps_old.py` do.
- [ ] The loss names `MSE_weighted_by_truth_nans` / `MSE_nans` are gone. The current names are `gates_losses.PixelWeightedMSELoss` / `MSELoss`.
- [ ] The output variables are `fp_pred` / `fp_transformed_pred`, not `predictions` / `trans_predictions`.
- [ ] The artefact is `training_outputs/scalers_*.pickle`, not `transform_parameters_*.pickle`.
- [ ] `scaler_files/` doesn't exist.
- [ ] The environment file is inconsistent: CLAUDE.md says `environment_short.yml`, README says `env_gates_pytorch.yml`.
- [ ] The v1 inputs function has been removed, not just deprecated.
- [ ] The repository layout is missing `scripts/`, `graph_net_block.py`, HOW_TO_LAUNCH and HOW_TO_PARAMETER_FILE.

---

## 4. Code bugs that break the documented usage

Fix these before documenting the features they affect.

### 4.1 Handcrafted scaler `stats` vs `stats_file`

- The class that's imported, `HandcraftedInputsScaler.__init__(self, stats)` ([datasets.py:667](../gates/data/datasets.py)), takes `stats`.
- The templates pass `scaler_params={"stats_file": ...}`:
  - `NEW_parameter_template_gpu_new3b.json:37`
  - `_terminal.json:81`
  - `_terminal_new.json:86`
  - This should raise a TypeError.
- The class docstring says "IN DEVELOPMENT - NOT READY FOR USE" (:645).
- `fit()` handles only `minmax`; everything else, including `"ghost"`, gets standard scaling. `minmax` uses only the stats for the first level.
- It has no fallback to data-driven scaling for missing variables.
- The `stats_file` version, `gates/data/handcrafted.py`, is untracked, never imported, and broken: it doesn't import `xr`, and `stats` is undefined when a dict is passed.

### 4.2 `land_cover` vs `landcover` in the MinMax list

- The default `minmax_variables` list has `"land_cover"` ([datasets.py:492](../gates/data/datasets.py)).
- The variable is actually called `"landcover"` (`load_data_helper_funs.py:151`).
- As a result, land cover gets standard scaling instead of MinMax.

### 4.3 `--checkpoint <int>` filename mismatch

- Training saves `{parameters['model_name']}_{epoch}.pt`, without the timestamp (`train_GATES_model.py:269`).
- Predict looks for `{model_dir.name}_{epoch}.pt`, where the folder name includes the timestamp ([predict_GATES_model.py:152](../scripts/predict_GATES_model.py)).
- `--checkpoint best` works.

✅ **Fixed (2026-09-28):** `load_checkpoint` tries `<timestamped_name>_<epoch>.pt` first, then `<base_name>_<epoch>.pt`, and raises a `FileNotFoundError` that lists both names if neither exists. The training output format is unchanged, so existing trained models still load. A new helper, `strip_timestamp`, is shared with `determine_save_name`.

### 4.4 `gates.__all__` lists a removed function

- [gates/\_\_init\_\_.py:15](../gates/__init__.py) lists `get_square_satellite_inputs`, which no longer exists.
- As a result, `from gates import *` fails.

✅ **Fixed (2026-09-28):** `get_square_satellite_inputs_v2` is now imported in `gates/__init__.py` and exported in its place. `from gates import *` works.

### 4.5 Other smaller bugs

- [ ] **`data_dirs` overrides crash:** `data_dirs.topog_datadir` / `landcover_datadir` raise a TypeError. `training_dataclasses.py:70–73` writes `topog_datadir`, but the loader expects `topog_path`.
- [ ] **Omitting a scaler section crashes:** leaving out `input_scaler` / `fp_scaler`, or setting it to `{}`, raises a NameError (`training.py:482–491, 508–519`).
- [ ] **Sweep launcher is broken from the repo root:**
  - The default `--sbatch-script` path is resolved relative to the current directory, but the file is in `launch/`.
  - The job is submitted with `cwd=launch/`, so `scripts/train_GATES_model.py` can't be found.
  - `--output=slurms/` also resolves relative to `launch/`.
  - The comments still refer to the old name `launch_sweep.py`.
- [ ] **`_get_normalize_fn` raises wrongly:** it raises even when evaluating the string succeeds (`loss_functions.py:147–157`).
- [ ] **`Config` ignores `filename`:** `Config(filename)` always opens `config.yml` (`config.py:161`).
- [ ] **`loss_functions_static.py` doesn't parse:** it has an IndentationError at line 307. The file is untracked and nothing imports it.
- [ ] **Stale templates:**
  - `NEW_parameter_template.json` (tracked) has `test_load_data.year: "201601"` and no `met_levels`, both of which break with v2.
  - `edge_exps_2.json` uses `release_concat`, which causes a TypeError.
