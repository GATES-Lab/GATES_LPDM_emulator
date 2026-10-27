# Flux handling: missing-flux warnings, fixed flux times, per-region multiregion metrics

Branch / worktree: `worktree-multiregion_flux` (`.claude/worktrees/multiregion_flux`)

## Goals

1. A missing flux file (or missing flux month) warns and training continues; it does not crash.
2. Flux can be taken from a fixed time (`"2016"` annual mean, or `"2016-01"` single month) and applied to every footprint, so flux metrics are comparable across months and years.
3. Footprints without matching flux are **kept** (flux is NaN), never removed from the dataset.
4. Multiregion training computes real-flux metrics **per region only** (no aggregate), and only for regions that have flux.

Out of scope for now: tests, aggregate flux metrics, making CHINA / INDIA flux load (see "Known data issues").

## Current behaviour (for reference)

- `load_flux_data` ([gates/data/load_data.py](../gates/data/load_data.py)) raises `ValueError` if the file is missing. Inside `load_GATES_data_v2` this is recorded as a failed year and becomes a `RuntimeError` after the year loop, so the run dies.
- `cut_flux_data` matches each footprint to the nearest flux snapshot within 32 days. `get_flux` then calls `remove_indeces(nan_idxs)`, which **drops** unmatched footprints; if all are unmatched it raises "All data has been removed".
- Multiregion: `transform_test_region` rebuilds the test dataset with only `fp_transformed` / `fp_original`, so `flux` never reaches `calculate_losses`. Real-flux metrics are never computed in multiregion.
- If one region has `flux` and another doesn't, the `xr.concat` of train fps and of the aggregate test set fails.
- `calculate_mfs` uses `np.nansum`, so NaN flux silently behaves like zero flux. Any NaN flux must be masked out before computing flux metrics.

## Config: new `flux_time` key

In the parameter file `"flux"` block:

```json
"flux": {
    "get_flux": true,
    "flux_time": "2016",
    "convert_units": true,
    "convert_units_args": {"unit_multiplier": 1e9}
}
```

| `flux_time` | Meaning |
|---|---|
| `null` / absent (default) | Current behaviour: load the footprint year's file, match each footprint to the nearest snapshot by date (32-day tolerance). |
| `"2016"` | Load the 2016 file. If it has a `time` dimension, take the mean over `time`. Apply that single field to every footprint, any year. |
| `"2016-01"` | Load the 2016 file and select January 2016. Apply that single field to every footprint. If that month is not in the file: **warn and continue** with NaN flux for that load. |

The year in `flux_time` decides which file is loaded and overrides `get_flux(year=...)`.

### Multiregion: per-region overrides

The top-level `"flux"` block is the default for every region. A region can override it with its own `"flux"` block, placed **next to** `train_load_data` (not inside it: every `train_load_data` key is passed through to `LoadSquareSatelliteData`):

```json
"flux": {"get_flux": true, "flux_time": "2016",
         "convert_units": true, "convert_units_args": {"unit_multiplier": 1e9}},

"regions": {
    "0-SAHARA": {"train_load_data": {...}, "test_load_data": {...}},
    "1-BRAZIL": {"train_load_data": {...}, "test_load_data": {...},
                 "flux": {"flux_time": "2016-01"}},
    "2-CHINA":  {"train_load_data": {...}, "test_load_data": {...},
                 "flux": {"get_flux": false}}
}
```

- Merge is one level deep: region keys replace global keys. A region that sets `convert_units_args` replaces the whole sub-dict.
- `"get_flux": false` for a region means no flux is loaded and no warning is raised; its flux is NaN (see Step 5), so it gets no flux metrics.
- The single-region script keeps using only the top-level block.

### Multiregion: flux is loaded for test data only

Flux metrics only use the test set. Training data loads flux **only if the loss uses it** (`weight_label: "flux"` in `loss_functions.criterion_params` or `criterion_test_params`). This saves load time and avoids NaN-flux warnings for training years.

## Step by step

### Step 1: `load_flux_data` raises `FileNotFoundError`

File: `gates/data/load_data.py`, `load_flux_data`.

- Replace the `ValueError`s for a missing file (the `flux_path` check and both "Flux file not found" branches) with `FileNotFoundError`, keeping the existing messages.
- Leave the other `ValueError`s (bad `cfg`, unknown domain) as they are: those are config mistakes and should still crash.

### Step 2: helper to select a fixed flux time

File: `gates/data/load_data.py`, new function next to `load_flux_data`, e.g. `select_flux_time(flux, flux_time)`.

- Input: the lazy `flux` DataArray from `load_flux_data`, and `flux_time` (str or None).
- `None` → return `flux` unchanged.
- `"YYYY"` → if `"time"` in `flux.dims`, return `flux.mean("time")`; otherwise return `flux` as is.
- `"YYYY-MM"` → if `"time"` in `flux.dims`, select that month (`flux.sel(time=flux_time)`, then drop the length-1 `time` dim). If the month is not present, `warnings.warn(...)` and return `None` to signal "no flux".
- Parse `flux_time` with `pd.Timestamp` / a regex so that anything other than `YYYY` or `YYYY-MM` raises `ValueError` (config mistake).
- A file with no `time` dimension (e.g. EASTASIA EDGAR) plus a `"YYYY-MM"` request → warn that the file has no time dimension and use the single field (confirmed).
- Fix found in testing: a month missing from *inside* the file's time range gives an empty selection, not a `KeyError`; that case now also warns and returns `None`.

### Step 3: `cut_flux_data` supports a time-less flux field

File: `gates/data/load_data.py`, `cut_flux_data`, step "1. Time matching".

- If `"time"` not in `flux.dims` (fixed-time case): broadcast the field to `fp.time` (`flux.expand_dims(time=fp.time.values)`), and set `nan_idxs` to an empty `DatetimeIndex`. Skip the tolerance matching.
- Otherwise match each footprint to the snapshot from its own calendar month (year + month). **Bug fix:** the old code matched to the *nearest* snapshot within 32 days; with snapshots dated the 1st of each month, footprints from ~the 16th onwards got the next month's flux. The `tolerance` argument is removed (no caller passed it). Unmatched footprints become NaN.
- Spatial cropping (steps 2–5) is unchanged.
- Update the docstring: `flux` may be `(time, lat, lon)` or `(lat, lon)`.

### Step 4: `get_flux` keeps all footprints and takes `flux_time`

File: `gates/data/load_data.py`, `LoadSquareSatelliteData.get_flux`.

- New argument `flux_time=None`. If set, the file year is the year part of `flux_time` (ignore / override `year`).
- Order: `load_flux_data` → `select_flux_time` → `transform_flux` (if `convert_units`) → `cut_flux_data`.
- If `select_flux_time` returned `None` (month not found): build an all-NaN flux (see Step 5 helper) and return it, rather than raising.
- **Remove** the `self.remove_indeces(nan_idxs)` call. Instead, if `len(nan_idxs) > 0`, `warnings.warn` with the count and note that flux is NaN for those footprints.
- Docstring: document `flux_time` and the NaN behaviour.

`flux_args` is already forwarded as `**kwargs` to `get_flux` by `load_GATES_data_v2`, so `flux_time` in the parameter file reaches `get_flux` with no extra plumbing.

### Step 5: missing flux file warns and fills with NaN

File: `gates/training/training.py`, `load_GATES_data_v2` (the `if get_flux: data.get_flux(**flux_kwargs)` block).

- Wrap `data.get_flux(**flux_kwargs)` in `try / except FileNotFoundError as e`.
- On failure: `warnings.warn(f"No flux for {region/domain} {year}: {e}. Continuing with NaN flux.")` and add an all-NaN flux variable: `data.fp_xr["flux"] = xr.full_like(data.fp_xr["fp"], np.nan, dtype="float32")`.
- Rule that falls out of this: **when `get_flux` is on, every loaded dataset has a `flux` variable**, NaN where no flux exists. All the `xr.concat` calls (across years, across regions, aggregate) then work without special cases.
- Optional (performance only): when `flux_time` is set, the same file/field is loaded for every year. It can be loaded once per call, but correctness doesn't depend on it.

### Step 6: flux metrics ignore NaN flux

File: `gates/training/training.py`, `calculate_losses`.

- Before `compute_flux_metrics`, find the timestamps with valid flux, e.g. `valid = test_outputs_xr.flux.notnull().any(["lat", "lon"])`.
- If no timestamps are valid: skip flux metrics (no `flux_eval_metrics` key, nothing appended to `losses["metrics_fluxes"]`).
- Otherwise compute on `.isel(time=valid)` / `.sel(time=valid)` for `fp_original`, `fp_pred`, `flux`.
- This also benefits single-region training (`scripts/train_GATES_model.py`), which already reads `flux_eval_metrics` when present.
- Optionally pass `ignore_mask=fp_nan_mask` to `compute_flux_metrics`, as other metrics do.

### Step 7: helper to detect a flux-weighted loss, plus a guard

File: `gates/training/training.py`.

- New helper, e.g. `loss_uses_flux(parameters)`: returns True if `parameters["loss_functions"]["criterion_params"]` or `["criterion_test_params"]` has `weight_label == "flux"`. Used by Step 9 and by the guard below.
- Guard, where the criterion is built (`setup_GATES_model`) or right after the train dataloader is built: if `loss_uses_flux(parameters)` and the training `flux` is missing or contains NaNs → `raise ValueError(...)` explaining that flux-weighted losses need flux for every training sample.
- A missing flux is only a warning for metrics, never for the loss.

### Step 8: per-region flux config in multiregion

File: `scripts/train_GATES_model_multiregion.py`, `build_region_configs` and its call in `train_and_save_model_multiregion`.

- Pass the global `parameters.get("flux")` into `build_region_configs` (new argument, e.g. `global_flux_args`).
- For each region: `region_flux = {**(global_flux_args or {}), **region_entry.get("flux", {})}`; store it as `"flux"` on the region config dict (next to `train_load_data` / `test_load_data`).
- Update the `build_region_configs` docstring with the new key and merge rule.
- `parameters["regions"]` is already saved in `training_settings`, so each region's override is recorded. Optionally also save the resolved per-region flux dicts (e.g. `parameters["resolved_region_flux"]`) so the effective settings are explicit.

### Step 9: load flux for test only (unless the loss needs it)

File: `scripts/train_GATES_model_multiregion.py`, `load_multiregion_data` (and its call).

- Remove the single `flux_args` argument; use `region_config["flux"]` per region instead.
- New argument `train_flux` (bool), set by the caller to `gates_training.loss_uses_flux(parameters)`.
- Train load: `flux_args=region_config["flux"] if train_flux else None`.
- Test load: `flux_args=region_config["flux"]` always (subject to its `get_flux`).
- Consequence: training fps normally have **no** `flux` variable, so the training loader labels don't include `flux`. Test sets must match (Step 10).
- If `train_flux` is True, every region's training data has a `flux` variable (NaN where missing, Step 5), so the train concat works; the Step 7 guard then catches any NaNs.
- When `get_flux` is false for a region's test load but other regions have flux, `flux` is missing from that region's test set. Fine for metrics (Step 6 skips it); Step 11 must handle a region with no `flux` variable at all.

### Step 10: multiregion test sets carry `flux` for metrics, not in the loader

File: `scripts/train_GATES_model_multiregion.py`, `transform_test_region`.

- New argument `fp_labels` (the training loader labels).
- Build `test_fp_ds` and the test loader as now, so the loader labels match training. Only include `flux` in the loader dataset if `"flux"` is in `fp_labels` (flux-weighted loss case).
- After `trim_to_batch_size`, if `test_fps` is a Dataset with `flux`, attach `test_fp_ds["flux"] = test_fps["flux"]` aligned on `time` (trimmed to the same timestamps), so it's available to `calculate_losses` but not fed to the model.
- Watch: the loader may hold a reference to `test_fp_ds`, and the training loop already adds `fp_pred` etc. to it in place. Check that adding variables after loader creation doesn't change what the loader yields (e.g. by passing the loader a `[labels]` subset/copy).
- Optional: compare the test loader labels with `fp_labels` and raise if they differ, as `setup_GATES_dataloaders` does for single-region.

### Step 11: per-region flux metrics, no aggregate

File: `scripts/train_GATES_model_multiregion.py`, `run_full_training_multiregion`.

- In the per-region loop: `region_log[region_name]["flux_eval"] = computed_metrics_r.get("flux_eval_metrics")` (None for regions without valid flux, via Step 6).
- In W&B logging: for regions where `flux_eval` is not None, log `metrics_fluxes/{region_name}/{metric}`. Covered by the existing `define_metric("metrics_*", ...)`.
- Aggregate: drop `flux` before the aggregate concat (`xr.concat([ds.drop_vars("flux", errors="ignore") for ds in all_test_fp_datasets], dim="time")`), so `calculate_losses` on the aggregate never produces flux metrics.
- Optionally add the per-region flux metrics to the `write_to_file` per-region line.

### Step 12: multiregion loading summary

File: `scripts/train_GATES_model_multiregion.py`, `load_multiregion_data`.

- After loading, print (and `write_to_file`) each region's resolved flux settings and whether its test set has any non-NaN flux, so it's obvious from the logs which regions will get flux metrics.

### Step 13: parameter templates and docs

- Add `"flux_time"` to `parameter_files/NEW_parameter_template*.json` `"flux"` blocks. Suggested: `"2016"` in the multiregion templates (comparable metrics), `null` elsewhere (unchanged behaviour).
- In the multiregion templates, show one example per-region `"flux"` override.
- Document `flux_time`, per-region overrides, test-only loading and the NaN / warning behaviour in the README or How_Tos where the `"flux"` block is described.

## Known data issues (not fixed by this branch)

Checked in `/projects/b5bn/data/fluxes_archive`:

- `NORTHAFRICA`: `ch4_NORTHAFRICA_{2010..2017}.nc`, 12 monthly snapshots each.
- `SOUTHAMERICA`: `ch4_SOUTHAMERICA_{2010..2016}_SWAMPS-v32-5_Saunois-Annual-Mean.nc` (12 monthly snapshots despite the name), plus 2017–2018 `_copy-from-2016` files.
- `EASTASIA`: only `ch4_EASTASIA_2016_EDGARv6.nc` (renamed from `ch4-…`), with **no time dimension** and dims named `latitude` / `longitude` (renamed to `lat` / `lon` by `select_flux_time`). EASTASIA met stops at 2015.
- `SOUTHASIA`: empty.
- `config.yml` (local) now maps CHINA to `domain_name: EASTASIA`, `flux_suffix: _EDGARv6`, so CHINA 2016 flux loads; other CHINA years warn and run with NaN flux. INDIA still has `domain_name: SOUTHASIA` with no flux files, so it warns and runs with NaN flux.

## Checklist

- [x] Step 1: `FileNotFoundError` in `load_flux_data`
- [x] Step 2: `select_flux_time` helper
- [x] Step 3: time-less flux in `cut_flux_data`
- [x] Step 4: `get_flux` keeps footprints, takes `flux_time`
- [x] Step 5: missing file → warning + NaN flux in `load_GATES_data_v2`
- [x] Step 6: NaN-aware flux metrics in `calculate_losses`
- [x] Step 7: `loss_uses_flux` helper + flux-weighted loss guard
- [x] Step 8: per-region flux config in `build_region_configs`
- [x] Step 9: test-only flux loading in `load_multiregion_data`
- [x] Step 10: `transform_test_region` attaches `flux` for metrics, not the loader
- [ ] Step 11: per-region flux metrics, no aggregate
- [ ] Step 12: loading summary of regions with flux
- [ ] Step 13: templates and docs
