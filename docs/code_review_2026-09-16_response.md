# Response to GATES code review — 2026-09-16

**Status of this document.** Response to [`docs/code_review_2026-09-16.md`](code_review_2026-09-16.md)
(Fable review requested by Matt Rigby, against branch `zarr_stores_metonly2` at commit
`e881b01`). This response was produced by re-reading the code at the same commit and
checking each claim against the source. Nothing was run on a GPU or against real data;
verdicts marked **[needs run]** require execution to close out.

**Verdict key.**

- ✅ **Confirmed** — reproduced in the code exactly as described.
- ⚠️ **Confirmed, caveat** — the bug is real but a detail (line number, scope, or trigger condition) differs from the writeup.
- 🔵 **Needs testing** — plausible from a read but only a run/retrain settles it.
- ❌ **Not reproduced** — could not confirm the claim.

## 1. Bottom line

The review is accurate. I checked all 25 correctness findings and the premises behind the
performance and health items; every one I could verify statically was **confirmed**. The four
High-severity findings (C1–C4) are genuine and two of them (C1, C4) have been silently
corrupting geometry and the saved loss history in every run to date.

Nothing here invalidates the published paper results — the review's own reasoning explains why
(C1's relabelling is identical across every sample, so a trained model learns around it; the
paper did not depend on the newer physical-geometry edge features). But C1 and C2 do block the
newer `dynamic_edges` / `latlon_mesh_edges` work from being physically correct. C3 is real but
milder than written: our templates set `shuffle: true`, so batch *order* is reshuffled each epoch;
what stays frozen for the whole run is batch *composition* (see C3 below — proposed downgrade to
Medium).

**Recommended immediate actions (before any new training runs):**

| Priority | Findings | Why now |
| --- | --- | --- |
| Do first | C4, C10, C24, C25 | Output-neutral, cheap, and they make everything downstream trustworthy (loss history, saved config, RNG, tests). |
| Do before next paper run | C1, C2, C3 | Change outputs; must be fixed + retrained for the geometry/dynamic-edge work to mean anything. |
| Do before next prediction product | C8, C9 | Systematically drop soundings from every predicted month. |

## 2. Correctness findings

### High severity

**C1 — grid node list is the transpose of the flattened data. ✅ Confirmed.**
[`load_data.py:2300-2305`](../gates/data/load_data.py#L2300-L2305).
`np.meshgrid(lat, lon)` with the default `indexing="xy"` returns arrays of shape
`(n_lon, n_lat)`, and the comprehension iterates `for i in range(lon.size) for j in
range(lat.size)`, so node `k` = `(lat[k % n_lat], lon[k // n_lat])`. The data are flattened with
`stack(["lat","lon"])`, i.e. node `k` = `(lat[k // n_lon], lon[k % n_lon])`. For a square window
the diagonal agrees and the off-diagonal nodes are swapped; shapes match so nothing raises. The
review's numpy reproduction matches mine. **Action:** adopt the review's `latlons`/`idx_latlons`
construction (lat outer, lon inner), audit `better_meshnodes` / `higher_res` / `append_latlon`
consumers, add `tests/data/test_get_grid.py`, **retrain**.

**C2 — `dynamic_earthdistance` uses the longitude feature as source latitude. ✅ Confirmed.**
[`encoder.py:886`](../gates/model/layers/encoder.py#L886) reads `src_lat` from
`self.latlon_indices[1]` (longitude). Lines 887–889 are correct. One-character fix (`[1]`→`[0]`).
This only affects runs with `dynamic_earthdistance=True`, which is off by default, so no existing
default-config checkpoint is wrong because of C2 — but the dynamic-edge experiments are. **Action:**
fix line 886, add the two-node unit test the review describes, retrain any dynamic-edge model.

**C3 — batch *composition* is frozen for the whole run. ⚠️ Confirmed, but the mechanism in the
writeup is wrong when `shuffle: true` is set (which our templates do).**
[`datasets.py:1346-1352`](../gates/data/datasets.py#L1346-L1352). The review says every epoch
"iterates the same batches in the same order". The order part is only true when `dataloader_params`
does **not** set `shuffle`. Our parameter templates set
`"dataloader_params": {..., "shuffle": true}` (e.g.
[`NEW_parameter_template_gpu_new.json:47`](../parameter_files/NEW_parameter_template_gpu_new.json#L47)),
this is read at [`training.py:535,542`](../gates/training/training.py#L535) and forwarded to the
DataLoader, which is built with `batch_size=None` at
[`datasets.py:1374-1378`](../gates/data/datasets.py#L1374-L1378). With `batch_size=None` +
`shuffle=True`, PyTorch attaches a `RandomSampler` over the MapDataset, so the **order of batches is
reshuffled each epoch**.

What is *not* fixed by `shuffle: true`: batch **composition**. The single
`np.random.permutation` runs once at dataloader construction, and `make_inputs_batcher` slices
`inputs.chunk(fp_time=batch_size)` into **fixed contiguous windows**
([`datasets.py:1362`, `make_inputs_batcher`](../gates/data/datasets.py#L1167)). `shuffle` reorders
those windows but never regroups the samples inside them. So for the entire run, the same 5 soundings
are always batched together; only the sequence in which those fixed batches are presented varies.

**Net:** the defect is real but milder than the review implies for our `shuffle: true` configs — we
get per-epoch *ordering* stochasticity but no per-epoch *composition* stochasticity (which is the
part that matters most for SGD). For any config that omits `shuffle`, the review's full "same batches,
same order" scenario holds exactly. **Action:** still worth fixing — shuffle the underlying array
per-epoch (RandomSampler over samples + per-epoch xbatcher rebuild, or the tensor-dataset
`torch.randperm` from P2). Fixes C24 at the same time. Because it changes batch composition, a fixed
version **changes outputs → retrain**. Suggest **downgrading from High to Medium** given the partial
mitigation already in place.

**C4 — metric history lists shared between keys. ✅ Confirmed.**
[`training.py:576-590`](../gates/training/training.py#L576-L590). `metrics_dict.copy()` is a shallow
copy: each `.copy()` produces a new dict but the value *lists* are the same objects. So
`losses["metrics_transformed"]["mse"] is losses["metrics_original"]["mse"]` is `True`, and likewise
the three `metrics_fluxes_static` blocks share `flux_metrics_dict`'s lists. `calculate_losses`
appends through both names, interleaving the two series. The saved `loss` in every checkpoint and
the text log are affected; W&B logging uses `computed_metrics` directly and is fine. **Action:**
build fresh lists per key (`copy.deepcopy` or a small factory). Output-neutral for the *model*, so
no retrain needed — but historical saved loss curves are unreliable. Cheap, do first.

### Medium severity

**C5 — per-year load failures are swallowed. ✅ Confirmed.**
[`training.py:382-385`](../gates/training/training.py#L382-L385). `except Exception` prints,
`traceback.print_exc()`, sets `loaded_samples = 0`, continues. A single bad path silently trains on
fewer years; all-bad raises a cryptic empty-`concat`. **Action:** re-raise, or collect failed years
and raise at end of loop.

**C6 — `load_fps` falls through to a `NameError`. ✅ Confirmed.**
[`load_data.py:143-176`](../gates/data/load_data.py#L143-L176). In the except path, the
"couldn't find a match" branch (143) and the `else: print("there was a problem", e)` (173-174)
both leave `fp_data_full` unassigned, so line 176 raises `NameError` and loses the original `e`.
**Action:** `raise` (keep `e`) in both branches.

**C7 — duplicate `fp_time` removal is dead code. ✅ Confirmed (line drift → now
[`datasets.py:1913-1922`](../gates/data/datasets.py#L1913-L1922)).** `len_inputs_before =
len(...fp_time)` then `if len(...fp_time) < len_inputs_before:` compares a value to itself → always
`False`; there is no `drop_duplicates` between them, so the block never runs. Note the *fp* side is
already deduped at [`load_data.py:724`](../gates/data/load_data.py#L724), but `xr.concat` across
years can reintroduce input-side duplicates. **Action:** implement the intent (`drop_duplicates`
before measuring, then filter `fp_xr`) or delete the block and add a uniqueness assertion.

**C8 — met sliced to the calendar month, dropping early-month soundings. ✅ Confirmed.**
[`load_data.py:~603`](../gates/data/load_data.py#L603) does
`met_file.sel(time=met_file.time.dt.month == int(self.month))` for single-month loads, so a
sounding early on the 1st has no met at `t − max(time_deltas)` and is dropped. Every predicted month
loses its first `max(time_deltas)` hours. **Action:** extend the met slice to
`[month_start − max(time_deltas), month_end]`, opening the previous year's Zarr when the margin
crosses the year boundary. Verify predicted-count == sounding-count.

**C9 — batch trimming discards soundings at inference. ✅ Confirmed.**
[`datasets.py:1272-1289`](../gates/data/datasets.py#L1272-L1289) drops the last `n % batch_size`
samples. Called at inference ([`predict_GATES_model.py:379`](../scripts/predict_GATES_model.py#L379))
and in training ([`training.py:539-540`](../gates/training/training.py#L539-L540)) *before* the
shuffle, so training always drops the same latest samples. **Action:** at inference run/pad the
final partial batch; in training trim after shuffle (or drop it entirely under P2).

**C10 — scaler names popped before they're saved. ✅ Confirmed.**
[`training.py:456-461`](../gates/training/training.py#L456-L461) and
[`:482-489`](../gates/training/training.py#L482-L489). `input_scaler_params =
parameters.get("input_scaler", {})` returns the *same* dict object, and `.pop("scaler")` mutates
`parameters["input_scaler"]` in place, so the written `training_settings_<model>.json` loses the
scaler class, and a second call (multiregion / test-then-train) silently uses the default scaler.
**Action:** `dict(parameters.get(...))` before popping, in both functions. Cheap, do early.

**C11 — `_pad_domain` assumes ascending coordinates. ✅ Confirmed.**
[`load_data.py:1968-1972`](../gates/data/load_data.py#L1968-L1972) and
[`:1987-1991`](../gates/data/load_data.py#L1987-L1991) wrap the padded coords in `sorted(...)`;
with a descending coordinate (`delta < 0`, as in ARCO ERA5 / GLIDE stores) the result is
non-monotonic and later `.sel(method="nearest")` picks wrong cells silently. **Action:** either
assert/`sortby` ascending at load, or build padding as `coord[0] + delta*np.arange(-pad,0)` etc.
without `sorted`. Extend `TestPadDomain` with a descending case.

**C12 — empty `wind_indices` accepted silently. ✅ Confirmed.**
[`training.py:182`](../gates/training/training.py#L182):
`wind_indices = [i for i, name in enumerate(input_names) if name in wind_tuples]` — if the
`(name, level, delta)` tuples don't match (e.g. JSON round-trip turned a tuple into a list), the
list is empty, `n_wind = 0`, and the model runs "dynamic wind edges" carrying no wind. `latlon_indices`
*is* length-checked ([`encoder.py:607-608`](../gates/model/layers/encoder.py#L607-L608)). **Action:**
raise on empty `wind_indices`; print resolved indices/names once at startup.

**C13 — min–max scaler patches over fill values. ✅ Confirmed.**
[`datasets.py:305-323`](../gates/data/datasets.py#L305-L323) recomputes min/max from `.values` when
`abs(min) > 1e25 or min < -50000` (and `abs(max) > 1e25`). This masks undecoded `_FillValue`s
(e.g. −1e30) that still live in the data, so wherever the heuristic doesn't trip the scaled inputs
are wrong. **Action:** decode fills at load (`mask_and_scale=True` or explicit `where` in
`load_topog`), delete the workaround, assert finite/plausible fitted stats.

**C14 — attention mask excludes self-attention. ✅ Confirmed (intent to confirm).**
[`encoder.py:717-718`](../gates/model/layers/encoder.py#L717-L718): `attn_mask = 1 -
to_dense_adj(mesh_edge_index)` then `.fill_diagonal_(1)`. For `nn.MultiheadAttention`, `True`/`1`
means *not allowed to attend*; the k-ring already includes the cell, so the diagonal was 0 (allowed)
and is forced to 1 (masked) — each node attends to neighbours but not itself. The attention path is
flagged "needs testing", so this is Medium. **Decision needed from us:** is masking self intended? If
not, delete the `fill_diagonal_` line; if yes, document it.

**C15 — training target selected by position 0. ✅ Confirmed.**
`fp_batch[:, :, 0]` at [`train_GATES_model.py:67`](../scripts/train_GATES_model.py#L67) and
[`:121`](../scripts/train_GATES_model.py#L121), and multiregion `:46`/`:85`. Works today only
because `fp_transformed` happens to be stacked first. **Action:** resolve
`target_idx = fp_labels.index("fp_transformed")` once and index with it.

### Low severity — all ✅ Confirmed

- **C16** [`encoder.py:371`](../gates/model/layers/encoder.py#L371): the obsolete `SatelliteEncoder`
  rebuilds `self.h3_nodes = nn.Parameter(...)` every forward → stale optimizer reference. Reachable via
  `use_dynamic_encoder=False`. Fix: delete `SatelliteEncoder` or register once as a buffer.
- **C17** [`forecast.py:202-205`](../gates/model/forecast.py#L202-L205): `encode_edges=False` /
  `encode_nodes=False` set `edge_dim=2` / `node_dim=feature_dim` for the processor while the encoder
  still emits full-width tensors → shape error in the first block. Fix: remove the flags or implement
  the bypass in the encoder.
- **C18** [`graph_net_block.py:590`](../gates/model/layers/graph_net_block.py#L590) (`7*in_dim_node`),
  docstring at `:553`. The disaggregated processor hard-codes 7 incoming edges; `release_edges=True`
  exceeds 7 → `node_mlp_2` raises. `scatter_cat` also allocates on CPU regardless of device. Fix:
  compute max in-degree at construction; buffer `idx_positions`; delete the unused `scatter_cat`.
- **C19** [`encoder.py:39`](../gates/model/layers/encoder.py#L39),
  [`decoder.py:39`](../gates/model/layers/decoder.py#L39): `torch.zeros(..., device=x.device)` with no
  `dtype` → float32 padding, will break under bf16 autocast (P3). Fix: add `dtype=x.dtype`.
- **C20** [`training_helperfuns.py:232`](../gates/training/training_helperfuns.py#L232): W&B artifact
  named `checkpoint_epoch_{self.counter}` — the early-stopping counter, which is 0 at save time, not the
  epoch. Fix: pass the epoch into `EarlyStopping.__call__`.
- **C21** [`datasets.py:1846+`](../gates/data/datasets.py#L1846): `domain_*` static variables come from
  the full-domain grid but assign into `static_ds` shaped `(fp_time, size, size)` → shape error. Fix:
  compute from `fp_xr` `lat_coords`/`lon_coords` or drop from the registry. **[needs run to trigger]**
- **C22** `eval()` on config strings at [`training.py:695`](../gates/training/training.py#L695),
  [`:705`](../gates/training/training.py#L705), [`datasets.py:419`](../gates/data/datasets.py#L419).
  Fix: name→class registry.
- **C23** [`datasets.py:1627`](../gates/data/datasets.py#L1627) builds `wind_angle` via `arctan2` (circular
  in (−π, π]); `DefaultInputsScaler` z-scores it, leaving a discontinuity at ±π. Fix: feed `sin`/`cos`
  (or just `x_wind`/`y_wind`) and drop the raw angle.
- **C24** [`datasets.py:1348`](../gates/data/datasets.py#L1348): global `np.random.seed` reseeds the
  process RNG for everything downstream. Fix: local `np.random.default_rng`; disappears with C3.
- **C25** test doubles out of date with the Zarr refactor
  ([`tests/data/test_load_data.py:294`](../tests/data/test_load_data.py#L294) and the conftest/helper
  stubs). Fix: run `pytest -q`, update the `_get_meteorology_file` stand-in signature, add the CI job (H4).
  **[needs run]** to confirm the current failure mode.

## 3. Performance findings

The premises I could check are all correct: batch size **5** is the default in every parameter file
(e.g. [`NEW_parameter_template_gpu_new3b.json:47`](../parameter_files/NEW_parameter_template_gpu_new3b.json#L47));
`wandb.watch(model, log="all", log_freq=100)` is at
[`train_GATES_model.py:493`](../scripts/train_GATES_model.py#L493); inference runs with
`num_workers=0` ([`predict_GATES_model.py:388`](../scripts/predict_GATES_model.py#L388)). The FLOP
estimate (§3.1) is order-of-magnitude and consistent with a launch-/memory-bound tiny network.

I have **not** measured where wall-clock time actually goes — that is exactly what **P0** exists for,
and the review is right to gate everything on it. **Recommendation:** accept the P0→P6 plan as
written, but do P0 before committing to the P1/P2 rewrite so the 10–50× claim is grounded in a
measured profile rather than an estimate. P4 (static-shape losses) and P5 (cheaper eval, `wandb.watch`
off by default) are low-risk and can land alongside the Phase-1 correctness fixes.

One coupling to flag: **P2 supersedes C3, C9 and C24** (tensor dataset + per-epoch `randperm` + no
trimming). If we're doing P2 soon, fix C3/C9/C24 minimally now rather than investing in the xbatcher
path we're about to delete.

## 4. Code health — ✅ Confirmed

- **H1** legacy duplicates present: `model/`, `v0.1.0_files/`, `v1_files/`, `examples.ipynb`,
  `examples_old.ipynb`. Move to a `legacy/` tag or delete.
- **H2** [`pyproject.toml`](../pyproject.toml) lists only `numpy`, `xarray`, `torch`; the package imports
  torch_geometric, torch_scatter, h3, einops, xbatcher, dask, pandas, scikit-learn, joblib, cartopy,
  wandb. List them or state the conda file is authoritative.
- **H3** README drift (root-vs-`scripts/` entry points, missing template, stale TODO notes). **[needs
  read of README to itemise]**
- **H4** only [`.github/workflows/docs.yml`](../.github/workflows/docs.yml) exists — no test CI. Add a
  `pytest -q` job. This is why C25 went unnoticed.
- **H5** `train_one_epoch` / `validate_and_predict` duplicated across the two training scripts → move to
  `gates/training/`.
- **H6** `print`-logging + large commented-out blocks. **H7** `<FILL IN>` docstrings. **H8** required
  `config.yml` is gitignored (ship `config.example.yml`). **H9** mutable default args (`={}`) across
  `load_data.py`/`datasets.py`.

## 5. Proposed order of work (our response to §5)

We accept the review's phasing. Concretely:

1. **Now, output-neutral:** C4, C10, C24, C25, plus H4 (CI) so regressions are caught. Land C5/C6
   (fail-loud loading) and C7 in the same batch.
2. **Before the next paper/geometry run, output-changing:** C1, C2, C3 — each with the test the review
   specifies, retrain the paper config, and compare metrics to the published numbers in the PR.
   Add C11, C13, C14, C23 to this phase.
3. **Before the next prediction product:** C8 and C9 (stop dropping soundings).
4. **Performance:** run **P0** first and paste the profile into the review doc, then P4/P5, then P1/P2,
   then P3/P6.

**Open decisions for Matt:**
- C14 — is masking self-attention intended?
- Legacy directories (H1) — tag-and-delete, or keep for reproducibility of v0.1.0 outputs?
- Do we commit to P1/P2 (preprocess-to-store + tensor dataset) this cycle, or fix C3/C9/C24 minimally
  and defer the loader rewrite?

## 6. Implementation log

Fixes are implemented on branch `code-review-fixes` (worktree), one commit per finding.

| Finding | Status | Changes outputs? | Commit | Verification |
| --- | --- | --- | --- | --- |
| C4 | Committed | No | C4+C5+C6 (one commit) | `new_metrics()`/`new_flux_metrics()` factories return fresh lists per key; isolated test confirms `metrics_original["mse"]` stays `[]` after appending to `metrics_transformed["mse"]`. Terminal smoke test (SAHARA size-10, 1 epoch) ran clean. |
| C5 | Committed | No | C4+C5+C6 (one commit) | Per-year load errors collected in `failed_years` and raised after the loop with a summary, instead of silently continuing with `loaded_samples=0`. Happy path verified by terminal smoke test. |
| C6 | Committed | No | C4+C5+C6 (one commit) | `load_fps` except-branch now re-raises the original error when no file matches the bad-files list, instead of falling through to a `NameError`. Workaround branch (unchanged) exercised by the smoke test, which loads a known bad SAHARA file. |
