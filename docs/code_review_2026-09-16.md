# GATES code review — 2026-09-16

**Scope.** Branch `zarr_stores_metonly2` at commit `e881b01`, the `gates/` package,
`scripts/`, `tests/`, and the legacy directories. Reviewed for (a) correctness and
(b) where wall-clock time goes and how to remove it. Reviewed by reading the code
and reproducing index arithmetic with numpy; nothing was run on a GPU or against
real data. Requested by Matt Rigby.

**Audience.** An agent working in `~/code/gates`. No GPU and no real data are
assumed on the machine; steps that need Isambard AI are marked **[GPU]**.

## How to use this document

- Findings are numbered **C** (correctness), **P** (performance), **H** (code health).
  Each states severity, location, what is wrong, why it matters, the fix, and how
  to verify the fix.
- Work through the phases in §5 in order. Phase 0 builds the regression fixtures
  that every later change is checked against. Do not skip it.
- Line numbers refer to the commit above. If they have drifted, search for the
  quoted code.
- Findings marked **changes outputs** alter the numbers a trained model produces.
  Existing checkpoints remain loadable but behave differently, so those changes
  need retraining and must be called out explicitly in the PR description.
- Keep each fix in its own small commit with the finding ID in the message.

## 1. Summary

The GATES network is tiny (about 0.7 GFLOP per footprint, §3.1) and nothing in
the model explains the published cost of roughly one second per footprint. The
time is in the pipeline around the model: batch size 5, an xarray/dask/xbatcher
loader wrapped around data that is already in memory, a met crop that is redone
for every run, per-epoch evaluation in xarray, and a paper-era prediction script
that runs the model on the CPU. Fixing the pipeline (§3.2) should give two to
four orders of magnitude on inference and one to two on training epochs.

Separately, the review found one high-severity geometry bug that has been present
in every trained model (C1: the grid node list is the transpose of the flattened
data), one clear index typo in the newest encoder feature (C2), a training loop
that shuffles once with a fixed seed so every epoch sees identical batches (C3),
and a metric-logging bug that corrupts the saved loss history (C4).

| Severity | Count | IDs |
| --- | --- | --- |
| High | 4 | C1, C2, C3, C4 |
| Medium | 11 | C5–C15 |
| Low | 10 | C16–C25 |
| Performance | 7 | P0–P6 |
| Code health | 9 | H1–H9 |

## 2. Correctness findings

### C1 (High) The grid node list is the transpose of the flattened data — changes outputs

**Location.** `gates/data/load_data.py:2300-2305` (`get_grid`) and
`gates/data/datasets.py:1194` (`inputs.stack(flat_lat_lon=["lat", "lon"])`), also
`datasets.py:1239` and `:1253` for the footprints.

**What.** `get_grid` builds the node list with `np.meshgrid(lat_coords, lon_coords)`
(default `indexing="xy"`) and iterates `for i in range(lon.size) for j in
range(lat.size)`, so node `k` is `(lat[k % n], lon[k // n])`. The data are
flattened with `stack(["lat", "lon"])`, so node `k` is `(lat[k // n], lon[k % n])`.
Reproduced on a 3×3 grid: nodes 0, 4 and 8 (the diagonal) agree, every other node
is swapped. Because the window is square the shapes match and nothing raises.

**Why it matters.** Every grid node is assigned to an H3 mesh cell, and every mesh
edge carries `[distance, dlat, dlon]`, using the coordinates of the transposed
position. The model therefore sees a valid regular grid but with lat and lon
roles swapped relative to the meteorology and footprint it is given. Static
features such as `lat_coords`, `earth_distance_centre` and `topog` come from the
data and are correct, so each node carries two inconsistent descriptions of where
it is. Training still works because the relabelling is the same for every sample,
which is why the paper results are fine, but the physically meaningful geometry
that newer features rely on (`latlon_mesh_edges`, `dynamic_earthdistance`,
`release_edges`, wind on edges) is mirrored. It would also raise a shape error the
moment a non-square window is used.

**Fix.** Build the node list in the data's order:

```python
lat_c = fp_xr.lat_coords.isel(time=reference_fp).values
lon_c = fp_xr.lon_coords.isel(time=reference_fp).values
latlons = [(la, lo) for la in lat_c for lo in lon_c]          # lat outer, lon inner
idx_latlons = [(i, j) for i in range(len(lat_c)) for j in range(len(lon_c))]
```

Check every consumer of `idx_latlon` (`better_meshnodes`, `higher_res`,
`append_latlon`) still gets what it expects; they are all off by default.

**Verify.** Add `tests/data/test_get_grid.py`: for a synthetic `fp_xr` with
distinct lat and lon values, assert `get_grid(fp_xr)[0][k] ==
(lat_c[k // n_lon], lon_c[k % n_lon])` for all `k`, and assert that the
`earth_distance_centre` feature at node `k` equals the haversine distance from
`latlons[k]` to the release point to within 1 m. Retrain before comparing to the
paper model.

### C2 (High) `dynamic_earthdistance` uses the longitude feature as the source latitude — changes outputs

**Location.** `gates/model/layers/encoder.py:886`.

**What.** `src_lat` is read from `features[:, self.latlon_indices[1], :]`; index
`[1]` is longitude. Lines 887–889 are correct. The edge distance is therefore
`haversine(src_lon, src_lon, dst_lat, dst_lon)`.

**Fix.** Change `[1]` to `[0]` on line 886.

**Verify.** Unit test on a two-node mesh with known coordinates: the dynamic
distance must equal `mesh_edge_attr_static[:, 0] / mean_edge_length` when the
lat/lon features are the raw degrees (build the encoder with a `GhostScaler` on
the coordinate features for the test).

### C3 (High) Training data are shuffled once, with a fixed seed — changes outputs

**Location.** `gates/data/datasets.py:1348-1351`.

**What.** `make_dataloader` seeds numpy with `random_seed=42`, permutes `fp_time`
once, and hands the permuted arrays to xbatcher. Every epoch then iterates the
same batches in the same order. With batch size 5 the batch composition matters.
The global `np.random.seed` call also resets the process RNG for everything that
runs afterwards (C24).

**Fix.** Do not permute the arrays. Give the `DataLoader` a
`torch.utils.data.RandomSampler` (or `shuffle=True` with a per-epoch generator)
over the xbatcher `MapDataset`, or, better, move to the tensor dataset of P2 where
this is a one-line `torch.randperm` per epoch.

**Verify.** Log the first five `fp_time` values of epoch 0 and epoch 1 and assert
they differ. Retrain.

### C4 (High) Metric history lists are shared between keys

**Location.** `gates/training/training.py:576-590` (`initialise_losses`).

**What.** `metrics_dict.copy()` and `flux_metrics_dict.copy()` are shallow copies,
so `losses["metrics_transformed"]["mse"]` and `losses["metrics_original"]["mse"]`
are the same list object, as are the three `metrics_fluxes_static` entries and
`metrics_fluxes`. `calculate_losses` (line 615) then appends to both through
either name. The text log written by `write_to_file`, the `loss` entry in every
checkpoint, and anything plotted from `losses` are wrong (each list holds both
series interleaved). Weights & Biases logging uses `computed_metrics` directly and
is unaffected.

**Fix.** Build fresh lists per key, for example
`{k: [] for k in ("nmae", "mse", "bias", "mae", "iou")}` inside a small factory
function called once per key, or `copy.deepcopy`.

**Verify.** `losses = initialise_losses(); losses["metrics_transformed"]["mse"].append(1); assert losses["metrics_original"]["mse"] == []`.

### C5 (Medium) Data-loading failures are swallowed per year

**Location.** `gates/training/training.py:339-385` (`load_GATES_data_v2`).

**What.** A failure for one year is printed, `loaded_samples` is set to 0, and the
loop continues. If every year fails the later `xr.concat([])` raises a cryptic
error; if some fail the run silently trains on fewer years.

**Fix.** Re-raise after printing, or collect the exceptions and raise at the end
of the loop with the list of failed years.

**Verify.** Point `met_datadir` at a nonexistent path in a test and assert the
call raises.

### C6 (Medium) `load_fps` falls through to a `NameError`

**Location.** `gates/data/load_data.py:143-176`.

**What.** In the `except` branch, when no file matches the bad-files list the code
prints two messages and, because the following `if` is false, reaches the `else`
that prints "there was a problem". `fp_data_full` is never assigned, so line 176
raises `NameError` and the original exception is lost.

**Fix.** `raise` the original exception (keep it as `e`) in that `else` branch,
and in the "couldn't find a match" branch.

### C7 (Medium) Duplicate `fp_time` removal is dead code

**Location.** `gates/data/datasets.py:1900-1907`.

**What.** `len_inputs_before` is set and immediately compared with the same
length, so the warning and the `fp_xr` filtering never run. If duplicate
timestamps survive (`drop_duplicates` runs on the full footprint set at
`load_data.py:724`, but `xr.concat` across years at `training.py:413-414` can
reintroduce them), `inputs.sel(fp_time=permuted_time)` in the dataloader
misbehaves.

**Fix.** Either implement the intent (`concatenated_inputs =
concatenated_inputs.drop_duplicates("fp_time")` before measuring, then filter
`data.fp_xr` to match) or delete the block. Add a hard assertion that
`concatenated_inputs.fp_time` is unique.

### C8 (Medium) Met is sliced to the calendar month, so early-month soundings are dropped

**Location.** `gates/data/load_data.py:603-604`; `gates/training/training.py:297-301`.

**What.** For a single-month load the met is filtered to that month, so a
sounding at 03:00 on the 1st has no met at t−6 h and t−12 h and is dropped with a
warning. In `predict_GATES_model.py` every month is predicted this way, so each
month loses its first `max(time_deltas)` hours of soundings. Whole-year training
loads lose the first hours of January the same way. The docstring documents this
but a prediction product with systematically missing soundings is a defect.

**Fix.** Load the met for `[month_start - max(time_deltas) h, month_end]`. The
Zarr store is per year, so this is a `sel(time=slice(...))` with the year boundary
handled by opening the previous year's store when the margin crosses it.

**Verify.** Predict January with `time_deltas=[6, 12]` and assert the number of
predictions equals the number of soundings in the footprint file.

### C9 (Medium) Batch trimming discards soundings at inference

**Location.** `gates/data/datasets.py:1272-1290`; used at
`scripts/predict_GATES_model.py:379` and `gates/training/training.py:539-540`.

**What.** `trim_to_batch_size` drops the last `n % batch_size` samples. At
inference this loses up to `batch_size − 1` soundings per month. In training it
always drops the same latest samples because trimming happens before the shuffle.

**Fix.** At inference, run the final partial batch (the model has no fixed batch
size) or pad it and discard the padding after the forward pass. In training,
trim after shuffling, or drop the trimming once P2 replaces the loader.

### C10 (Medium) Scaler names are popped from the parameters before they are saved

**Location.** `gates/training/training.py:456-462` and `:482-488`.

**What.** `input_scaler_params.pop("scaler")` mutates `parameters["input_scaler"]`
in place. `training_settings_<model>.json` is written afterwards
(`train_GATES_model.py`, `save_object(parameters, "training_settings", ...)`), so
the record of which scaler class was used is lost. A second call on the same
`parameters` dict (multiregion, or test then train) gets the default scaler
instead of the configured one.

**Fix.** `input_scaler_params = dict(parameters.get("input_scaler", {}))` before
popping, in both functions.

**Verify.** After `setup_GATES_dataloaders`, assert
`"scaler" in parameters["input_scaler"]`.

### C11 (Medium) `_pad_domain` assumes ascending coordinates

**Location.** `gates/data/load_data.py:1968-1972` and `:1987-1991`.

**What.** The padding coordinates are built as
`sorted([lat[0] - (i+1)*delta ...]) + list(lat) + [lat[-1] + (i+1)*delta ...]`.
If the coordinate is descending (`delta < 0`, as in ARCO ERA5 and the GLIDE met
stores) the result is non-monotonic and later `.sel(method="nearest")` picks
wrong cells. Index arithmetic in `_get_release_idxs` is direction-agnostic, so the
failure is silent.

**Fix.** Either assert monotonic ascending lat and lon at load time and sort the
datasets once (`sortby`), or build the padding as
`lat[0] + delta * np.arange(-pad_S, 0)` and `lat[-1] + delta * np.arange(1, pad_N+1)`
without `sorted`.

**Verify.** Extend `TestPadDomain` with a descending-latitude dataset and assert
the padded coordinate is monotonic and the release index is unchanged.

### C12 (Medium) An empty `wind_indices` is accepted silently

**Location.** `gates/training/training.py:177-184` (`setup_dynamic_edges`).

**What.** `wind_indices` is built by matching `(name, level, delta)` tuples
against `input_names`. If the level type or delta does not match (for example the
JSON round-trip turned the tuple into a list of a different level) the list is
empty, `n_wind` is 0, and the model runs with "dynamic wind edges" that carry no
wind. Same for `latlon_indices`, which is at least length-checked.

**Fix.** Raise if `wind_indices` is empty, and print the resolved indices and
names once at startup.

### C13 (Medium) The min–max scaler patches over fill values

**Location.** `gates/data/datasets.py:305-323`.

**What.** After computing `da.min()` the scaler checks `abs(min) > 1e25 or min <
-50000` and recomputes from `.values`. This is a workaround for undecoded fill
values (for example −1e30 in topography or land cover) reaching the scaler. The
underlying data still contain those values, so wherever the check does not trip
the scaled inputs are wrong.

**Fix.** Decode `_FillValue`/`missing_value` at load time (`mask_and_scale=True`,
or an explicit `where(abs(x) < 1e20)` in `load_topog`) and delete the workaround.
Add an assertion in `DefaultInputsScaler.fit` that every fitted min, max, mean and
std is finite and within a plausible range.

### C14 (Medium) The attention mask excludes self-attention — verify intent

**Location.** `gates/model/layers/encoder.py:717-719` (and `:318-319` in the
obsolete encoder); consumed at `gates/model/layers/graph_net_block.py:842`.

**What.** `attn_mask = 1 - to_dense_adj(edge_index)` then `fill_diagonal_(1)`.
For `nn.MultiheadAttention` a boolean mask value of `True` means "not allowed to
attend". The k-ring already includes the cell itself, so the diagonal was 0
(allowed) and is then set to 1 (masked). Each node therefore attends to its
neighbours but not to itself. If that is intended, document it; if not, delete
the `fill_diagonal_` line. The attention path is marked "needs testing" so this is
Medium rather than High.

### C15 (Medium) The training target is selected by position 0

**Location.** `scripts/train_GATES_model.py:67-69` and `:121-123`;
`scripts/train_GATES_model_multiregion.py:46-48` and `:85-87`.

**What.** `true_values = fp_batch[:, :, 0]` assumes `fp_transformed` is the first
stacked variable. It is today, because `FootprintDataset.transform` builds the
Dataset in that order and `keep_vars` appends `flux` afterwards, but nothing
enforces it and `fp_labels` is available.

**Fix.** `target_idx = fp_labels.index("fp_transformed")` once, stored in
`ModelContext`, and index with it.

### C16 (Low) The obsolete encoder re-creates a Parameter every forward

**Location.** `gates/model/layers/encoder.py:371`.

**What.** `self.h3_nodes = torch.nn.Parameter(self.h3_nodes.to(device))` replaces
the parameter object each call, so the optimizer's reference is stale. Still
reachable through `use_dynamic_encoder=False`.

**Fix.** Delete `SatelliteEncoder` (the dynamic encoder with all flags off is the
same model), or register `h3_nodes` once as a buffer.

### C17 (Low) `encode_edges=False` / `encode_nodes=False` produce inconsistent dimensions

**Location.** `gates/model/forecast.py:200-203`.

**What.** The processor is built with `edge_dim=2` or `node_dim=feature_dim`, but
the encoder still emits `edge_dim`/`node_dim`-wide tensors, so the first
processor block raises a shape error.

**Fix.** Remove the flags or implement the bypass in the encoder.

### C18 (Low) Disaggregated processor assumes exactly seven incoming edges

**Location.** `gates/model/layers/graph_net_block.py:484` (CPU allocation in
`scatter_cat`), `:590` (`7*in_dim_node`), `:621-631`.

**What.** `scatter_cat_v2` pads to the largest in-degree present; with
`release_edges=True` that exceeds 7 and `node_mlp_2` raises. `scatter_cat`
allocates on the CPU regardless of input device. The index bookkeeping is Python
lists rebuilt when the edge count changes.

**Fix.** Compute the max in-degree at construction from the mesh and size the MLP
from it; keep `idx_positions` as a registered `LongTensor` buffer; delete the
unused `scatter_cat`.

### C19 (Low) Zero padding is allocated at float32 regardless of input dtype

**Location.** `gates/model/layers/encoder.py:39`, `gates/model/layers/decoder.py:39`.

**Fix.** `torch.zeros(..., device=x.device, dtype=x.dtype)`. Needed before P3
introduces bf16 autocast.

### C20 (Low) wandb checkpoint artifacts are named by the patience counter

**Location.** `gates/training/training_helperfuns.py:232`.

**What.** `checkpoint_epoch_{self.counter}` uses the early-stopping counter, which
is 0 whenever a checkpoint is saved, not the epoch.

**Fix.** Pass the epoch into `EarlyStopping.__call__` and use it in the name.

### C21 (Low) `domain_*` static variables are incompatible with the square format

**Location.** `gates/data/datasets.py:1846`;
`gates/data/load_data_helper_funs.py:258-332`.

**What.** They are called with `data.fp_data_full` (full domain grid) but assign
into `static_ds`, which is `(fp_time, size, size)`, so a shape error follows.

**Fix.** Either compute them from `fp_xr` with `lat_coords`/`lon_coords`
(as `_earth_distance_centre` does) or remove them from the registry.

### C22 (Low) `eval()` on configuration strings

**Location.** `gates/training/training.py:695` and `:705`;
`gates/data/datasets.py:419`.

**Fix.** Replace with a name→class registry
(`{"MSELoss": gates_losses.MSELoss, ...}`) and a small argument parser. Also
removes the need for the parameter file to know module aliases.

### C23 (Low) `DefaultInputsScaler` standardises `wind_angle`

**Location.** `gates/data/datasets.py:560-575`; angle created at `:1503-1504`.

**What.** `wind_angle` is circular in (−π, π]; z-scoring it leaves a
discontinuity at ±π that the model has to learn around.

**Fix.** Feed `sin(angle)` and `cos(angle)` (or just `x_wind`, `y_wind`, which
already carry the information) and drop the raw angle.

### C24 (Low) Global numpy RNG is reseeded inside `make_dataloader`

**Location.** `gates/data/datasets.py:1348`.

**Fix.** Use a local `np.random.default_rng(random_seed)`; goes away with C3.

### C25 (Low) Test doubles are out of date with the Zarr refactor

**Location.** `tests/data/test_load_data.py:278-284`; `tests/test_helper_funs.py:76-82`;
`tests/conftest.py:46-49`.

**What.** The tests patch `_get_meteorology_file` with a stand-in whose signature
is `(met_datadir, lazy_load, met_time_chunk, parallel)`, but the current method is
called with `met_levels=` and `met_variables=` keywords, so those tests should
fail with `TypeError`. The synthetic met uses `model_level_number` and relies on
the stand-in to rename it to `levels`. There is no CI job that runs the tests
(H4), so this may have gone unnoticed.

**Fix.** Run `pytest -q`, update the stand-ins to the current signature, and add
the test job (H4).

## 3. Performance findings and acceleration plan

### 3.1 Where the time goes

Model cost for the paper configuration (50×50 window, 160 input features, node
and edge dim 64, hidden dim 16, 4 blocks, H3 resolution 4). The mesh is about
1,400 nodes and 10,000 edges for a South American window of this size.

| Stage | Approx. FLOPs per footprint |
| --- | --- |
| Encoder node MLP (160→16→16→64, per mesh node) | 11 M |
| Encoder edge MLP (3→16→16→64, per edge) | 27 M |
| Processor block: edge MLP (192→16→16→64, per edge) | 85 M |
| Processor block: node MLP 1 (128→16→16→64, per edge) | 65 M |
| Processor block: node MLP 2 (128→16→16→64, per node) | 9 M |
| Processor, 4 blocks | 640 M |
| Decoder (64→16→1, per grid node) | 7 M |
| **Total** | **≈ 0.7 GFLOP** |

That is tens of microseconds per footprint on a GH200 even allowing for the
memory-bound shape of these skinny matmuls. The observed costs come from
elsewhere:

1. **Batch size 5** (`gates/training/training.py:533-534`, default in the
   parameter files). The forward pass is roughly 150 kernel launches whatever
   the batch size, so at batch 5 the GPU is launch-bound at perhaps 150 µs per
   footprint. Batch 256–512 fits in memory easily (activations are about 10 MB
   per sample per block).
2. **The loader re-daskifies in-memory data.** `make_inputs_batcher`
   (`datasets.py:1190`) calls `.chunk(fp_time=batch_size)` on the materialised
   inputs, turning a numpy array into a dask array in five-sample chunks.
   `FootprintDataset.transform` (`datasets.py:1099`) chunks the targets with
   `{"time": 1}`. Each batch is then an xarray `isel` plus a dask compute inside a
   forkserver worker, and the whole array is pickled to every worker at startup.
3. **The met crop is redone every run.** `_cut_satellite_met_multi_delta`
   (`datasets.py:1388`) does per-sample vectorised indexing in xarray/pandas,
   logged at minutes per year, for every training run and every predicted month.
   There is no model-ready cache on disk.
4. **Per-epoch evaluation.** `run_full_training` inverse-transforms and runs the
   full xarray metric suite (IoU, correlation, static flux patterns) on the whole
   test set every epoch, and `wandb.watch(model, log="all")`
   (`train_GATES_model.py:493`) logs gradient histograms every 100 steps.
   `validate_and_predict` keeps every output tensor on the GPU until the end.
5. **The paper-era prediction script runs on the CPU.**
   `v0.1.0_files/general_make_prediction.py:243-255` loads the checkpoint with
   `map_location='cpu'`, never moves the model, and pushes a whole month through
   `model(...)` on the host. The current `predict_GATES_model.py` uses the GPU but
   with batch 5 and `num_workers=0`.
6. **Dynamic shapes in the loss.** `ThresholdedMSELoss`
   (`loss_functions.py:392-393`) uses `mask.sum() != 0` (a host sync per bin per
   batch) and boolean indexing `pred[mask]` (data-dependent shapes), which blocks
   CUDA-graph capture.
7. `set_reproducibility` (`training_helperfuns.py:149-151`) sets
   `cudnn.deterministic=True` and `benchmark=False` globally.

### 3.2 Plan

**P0. Measure first. [GPU]** Add `scripts/profile_training.py` (skeleton in
Appendix B). It times one epoch split into (a) `next(loader)`, (b) host→device
copy, (c) forward+backward+step with `torch.cuda.synchronize()` around each, and
runs `torch.profiler` for 20 steps. Report a table of seconds per epoch by
category and footprints per second. Everything below is prioritised on the
assumption that (a) dominates; confirm before refactoring.

**P1. Split preprocessing from training.** New `scripts/preprocess_GATES_data.py`
that runs the existing loading and scaling once and writes a model-ready store:

- `inputs`: `(N, H*W, F)` float16, flattened in `stack(["lat","lon"])` order
- `targets`: `(N, H*W)` float32 transformed footprint, `nan_mask` `(N, H*W)` uint8,
  and any `keep_vars` (e.g. `flux`) as separate arrays
- `fp_time`, `lat_coords (N,H)`, `lon_coords (N,W)`, `release_lat/lon`
- the fitted scaler objects and `input_names`, plus the parameter file used

Zarr with chunks of about 256 samples, or one `.npy` per array with a JSON
sidecar. Training reads this and never touches xarray. Keep the xbatcher path
only behind a `--legacy-loader` flag until parity is confirmed. Scaler fitting
keeps the current `fit_on_subsample` logic and seed so numbers are unchanged.

**P2. Tensor dataset, large batches, per-epoch shuffle.** With P1, the paper
training set is about 9 GB at float16 and fits on a GH200, so load it to the
device once and index with `torch.randperm(N)[i:i+B]`. For larger sets use a
memory-mapped array with a one-batch-ahead prefetch thread and pinned memory.
Batch size 128–512 with the learning rate scaled and a short warmup; sweep
`{64, 128, 256, 512}` × `{1e-4, 3e-4, 1e-3}` on the paper config. This also
resolves C3, C9 and C24.

**P3. Model changes** (all output-preserving except where noted; verify each
against the Phase 0 golden predictions):

- Replace `torch_scatter.scatter_sum/mean` with `torch.zeros(...).index_add_`
  (sum) and a precomputed in-degree divisor (mean), or `Tensor.scatter_reduce_`
  (`graph_net_block.py:387, 737-739`; `encoder.py:866`; `decoder.py:...scatter_mean`).
- Drop `torch_geometric.nn.MetaLayer`; call the edge and node modules directly
  (`graph_net_block.py:931-958`).
- Register the decoder's `edge_index`, `edge_weights`, `norm_distances` as
  buffers (`decoder.py`, the `.to(device)` calls at the top of `forward`).
- Pass the batch size explicitly instead of via `self.encoder.batch_size`
  (`forecast.py:forward`).
- Factor the first linear layer of the edge MLP and of node MLP 1 so the node
  contribution is computed per node and gathered, rather than gathered then
  multiplied per edge (`graph_net_block.py:426-427, 699-700`). Exact in exact
  arithmetic; expect float-level differences only.
- `torch.compile(model, mode="reduce-overhead")` once shapes are static, and
  `torch.autocast("cuda", dtype=torch.bfloat16)` around forward/backward with
  fp32 master weights. **Changes outputs at the bf16 tolerance level.**

**P4. Loss functions with static shapes.** Rewrite the masked means as
`(se * mask).sum() / mask.sum().clamp_min(1)` and drop the `if mask.sum() != 0`
branches (`loss_functions.py:389-395` and the other classes). Same values, no
host syncs, no data-dependent shapes.

**P5. Cheaper evaluation.** Run the xarray metrics every `epochs.visualize`
epochs rather than every epoch; move the transformed-space MSE to a torch
computation on the GPU; move each output batch to the CPU as it is produced;
default `wandb.watch` to off; make the cuDNN determinism flags opt-in.

**P6. Inference at scale. [GPU]** For satellite-scale prediction the model is no
longer the ceiling once P1–P3 are done; building input tensors is. Keep the
domain met for a window on the device and gather the `size×size` windows with
index tensors (the same operation `_cut_satellite_met_multi_delta` does in
xarray, in torch), run the compiled model at batch 1,024, and write predictions
to a Zarr store with sensible chunks instead of one NetCDF per month. Remove the
`trim_to_batch_size` call (C9).

**Architecture headroom.** Because the current network is launch- and
memory-bound, widening hidden dims to 128–256, adding blocks, or adding the
planned encoder–decoder skip connections will cost almost nothing in wall-clock
time once P1–P3 are in. Do not let the current per-footprint figure constrain
architecture choices.

**Expected gains** (estimates to be replaced by P0 measurements):

| Change | Expected effect |
| --- | --- |
| P1 + P2 (no xarray in the loop, batch 256) | 10–50× on epoch time |
| P3 compile + bf16 | 2–4× on model time |
| P3 MLP factoring | ~1.5–2× on processor time |
| P6 GPU-side crop and batching | inference from ~1 s to ~10⁻⁴ s per footprint |

## 4. Code health

- **H1. Legacy duplicates.** `model/` is an older copy of `gates/model` (files
  differ), `v0.1.0_files/`, `general_train_nawid.py` at the root, and the 956 KB
  `examples.ipynb`. Move to a `legacy/` tag or delete; they confuse imports and
  reviewers.
- **H2. `pyproject.toml` dependencies are incomplete.** Only numpy, xarray and
  torch are listed; the package imports torch_geometric, torch_scatter, h3,
  einops, xbatcher, dask, pandas, scikit-learn, joblib, cartopy and wandb. Either
  list them or state that the conda file is the only supported install.
- **H3. README drift.** It refers to `train_GATES_model.py` at the root (now in
  `scripts/`), to a parameter template that does not exist, and the "Inversion —
  TO DO" and "different size prediction not implemented" notes are stale.
- **H4. No test CI.** `.github/workflows/docs.yml` only builds docs. Add a
  workflow that installs the conda env (or a CPU-only subset) and runs `pytest -q`.
- **H5. Copy-paste between training scripts.** `train_one_epoch` and
  `validate_and_predict` exist in both `train_GATES_model.py` and
  `train_GATES_model_multiregion.py`. Move them into `gates/training/` and import.
- **H6. `print` used for logging throughout**, with large commented-out blocks
  in `encoder.py`, `decoder.py`, `graph_net_block.py`, `predict_GATES_model.py`.
  Use `logging` and delete dead code.
- **H7. Docstrings with `<FILL IN>`** in `forecast.py`, `graph_net_block.py`,
  `training_dataclasses.py`, `training_helperfuns.py`. Fill or remove.
- **H8. `config.yml` in `.gitignore` but required to import.** `get_config()`
  raises if it is missing; the tests need it. Ship a `config.example.yml` and
  have tests build a `Config` from a fixture.
- **H9. Mutable default arguments** (`met_args={}`, `topog_args={}`,
  `scaler_params={}`, `kwargs={}`) across `load_data.py` and `datasets.py`.
  Use `None` and create inside.

## 5. Order of work

**Phase 0 — regression fixtures and profiling.**
1. `pytest -q`; record what passes. Fix C25 so the data tests run.
2. **[GPU]** With the paper checkpoint (`trained_models/satellite_Brazil_50x50`)
   and 200 test inputs, save `golden_inputs.npy` and `golden_preds.npy`
   (transformed space) under `tests/golden/`. Every output-preserving change must
   reproduce `golden_preds` to `atol=1e-5`; bf16/compile changes to `atol=1e-2`.
3. **[GPU]** Run the P0 profiler on the paper configuration and record the table
   in this file.

**Phase 1 — fixes that do not change outputs.** C4, C5, C6, C7, C9, C10, C12,
C15, C16, C17, C18, C19, C20, C21, C22, C24, C25, H1–H9. One commit each.

**Phase 2 — fixes that change outputs.** C1, C2, C3, C11, C13, C14, C23. Each
gets a test, a note in the PR that retraining is required, and a retrained paper
configuration whose metrics are compared with the published ones.

**Phase 3 — performance.** P1, P2, P4, P5 first (they need no GPU to write, only
to measure), then P3, then P6. After each step rerun the profiler and update the
gains table.

## Appendix A — how the FLOP estimate was made

Per-sample sizes: 2,500 grid nodes; H3 resolution 4 cells average 1,770 km²; a
50×0.352° by 50×0.234° window near 10°S is roughly 1,900 km × 1,300 km, giving
about 1,400 mesh nodes and 7 edges per node (self plus 6 neighbours). An MLP
`a→h→h→b` costs `2(ah + h² + hb)` FLOPs per row. Multiply per-row cost by the row
count (nodes or edges) and sum. The numbers are order-of-magnitude, not exact.

## Appendix B — profiler skeleton for P0

```python
"""scripts/profile_training.py — time one epoch by phase. Run on a GPU node."""
import time, torch
from torch.profiler import profile, ProfilerActivity

def timed_epoch(model, loader, criterion, optimizer, device, n_profile_steps=20):
    t_load = t_copy = t_step = 0.0
    n = 0
    model.train()
    it = iter(loader)
    while True:
        t0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            break
        t1 = time.perf_counter()
        x = batch[0].to(device, non_blocking=True)
        y = batch[1].to(device, non_blocking=True)
        torch.cuda.synchronize(); t2 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y[..., 0:1], y)
        loss.backward(); optimizer.step()
        torch.cuda.synchronize(); t3 = time.perf_counter()
        t_load += t1 - t0; t_copy += t2 - t1; t_step += t3 - t2; n += x.shape[0]
    total = t_load + t_copy + t_step
    print(f"load {t_load:8.1f}s ({100*t_load/total:4.1f}%)  copy {t_copy:8.1f}s "
          f"({100*t_copy/total:4.1f}%)  step {t_step:8.1f}s ({100*t_step/total:4.1f}%)  "
          f"{n/total:8.1f} footprints/s")

def profile_steps(model, loader, criterion, optimizer, device, steps=20):
    it = iter(loader)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(steps):
            batch = next(it)
            x, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y[..., 0:1], y)
            loss.backward(); optimizer.step()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
```

Wire it to the existing `setup_GATES_dataloaders` / `setup_GATES_model` so it
uses the real loader, and run it once with the current loader and once with the
P1/P2 tensor dataset.
