# HOW TO: train the dual-head model on large data sets (month cache)

The standard data pipeline of the dual-head trainers holds about seven copies of the training
inputs at its peak (11.7 GiB of host memory per 1,000 samples at a 50 x 50 window), so a 460 GB
node takes at most ~36,000 training samples, and every job spends 25–45 minutes loading before
the first epoch. The **month cache** removes both limits:

- every month of footprints + met + background is loaded from the archive **once** and kept on
  disk; later jobs read it back in seconds;
- the training batches are built **month by month straight in shared memory**, with one copy of
  the data instead of seven.

It is **off by default**: a parameter file without a `data_cache` block runs the standard
pipeline exactly as before.

> **What does not change.** The month cache builds the *same* data as the standard pipeline:
> same samples, same batches in the same order, same fitted scalers, same values. This is
> checked bit for bit on real data (`smoke_tests_lean/compare_old_vs_lean.py`, 2 training months)
> and on synthetic data (`tests/test_lean_dual_data.py`). Training on it gives the same losses
> as training on the standard pipeline (section 6).

## 1. Parameter file

```json
"data_cache": { "enabled": true }
```

| key | values | meaning |
|-----|--------|---------|
| `data_cache.enabled` | bool (default `false`) | Use the month cache and the memory-lean pipeline. |
| `data_cache.dir` | path (default: `user_paths.data_cache_dir` of `config.yml`) | Root directory of the cache. |
| `data_cache.rebuild` | bool (default `false`) | Reload every month from the archive and overwrite the cache. Use it after a change to the loader that alters what is loaded (see section 5). |
| `data_cache.threads` | int (default: min(8, CPUs of the job)) | Threads for reading and scaling. Each holds up to one month of inputs. |
| `data_cache.fit_buffer_gib` | number (default `48`) | Working memory of the scaler fit. The fit works through the variables and levels a few at a time; more memory = more of them at once = faster. |

`data_cache` is a data-loading key: like `train_load_data` it cannot be overridden per experiment
in `run_dual_experiments_shared_data.py` (all arms of a job share one data set).

Supported: the three dual trainers (`train_dual_model.py`, `train_dual_refit_model.py`,
`train_dual_headlr_model.py`), the shared-data driver, any number of GPUs, and
`predict_dual_model.py`. Limits: `DefaultInputsScaler` only (anything else raises an error).

### Which footprints are there? (SOUTHAMERICA domain)

`fp_archive/SOUTHAMERICA` holds two GOSAT series. **The BRAZIL series is part of the SOUTHAMERICA
series**: every BRAZIL footprint is in the SOUTHAMERICA files too, with the same time, release
point and footprint (checked on all months of 2013, 2014, 2015 and 2017,
`smoke_tests_lean/series_overlap.py`). The SOUTHAMERICA series adds the soundings of the rest of
the continent (release points 56°S–14°N, 82°W–33°W; BRAZIL: 36°S–7°N, 76°W–33°W).

| footprints at `freq` 1 | 2013 | 2014 | 2015 | 2016 | 2017 | 2013+14+15+17 |
|---|---:|---:|---:|---:|---:|---:|
| `"region": "BRAZIL"` | 11,045 | 16,530 | 17,390 | 21,671 | 22,345 | **67,310** |
| `"region": "SOUTHAMERICA"` | 15,242 | 22,663 | 24,122 | 33,629 | 36,585 | **98,612** |

(2012 has no CAMS backgrounds and cannot be used by the dual model. The 2016 numbers are the
file counts; 2016 is the test year of the programme.)

So the largest training set that keeps 2016 for testing is `"region": "SOUTHAMERICA"` with the
four years at `freq` 1: 98,612 footprints, of which 31,302 are outside the BRAZIL region.
`test_load_data` inherits every key it does not set from `train_load_data`, **including the
region**: give the test set `"region": "BRAZIL"` to keep it as in all earlier runs.

```json
"train_load_data": { "years": ["2013", "2014", "2015", "2017"], "freq": 1, "region": "SOUTHAMERICA", ... },
"test_load_data":  { "years": ["2016"], "freq": 10, "region": "BRAZIL" }
```

### SOUTHAMERICA at freq 2 with lagged met (experiment summary section 42)

The first training run on the SOUTHAMERICA series (2026-10-12) uses the section-38 inputs
(`time_deltas` [6, 12, 24, 48, 72]) at `freq` 2:
`parameter_files/parameter_dual_southamerica4yr_f2_nolatlon_td24_48_72_2013-14-15-17_test2016.json`
(about 49,300 training footprints; the BRAZIL test months of that file have the *same* cache key as
the BRAZIL run's, so they are reused and the two runs are tested on identical cached data). The CAMS
backgrounds are chosen by the domain, so both regions read the same
`ch4_SOUTHAMERICA_YYYYMM_CAMS-inversion.nc` files. Before production,
`smoke_tests_zarr/check_southamerica.sbatch` checks the caches
(`smoke_tests_zarr/check_southamerica_cache.py`: every second footprint of the freq-1 cache per
month, shared footprints identical to the BRAZIL cache, windows leaving the LPDM domain) and smokes
the arm for 3 epochs. `smoke_tests_zarr/run_time_horizon_production.sbatch` takes the base-file name
from `PARAM_FILE_PATTERN` (with `TAG` for the arm). To score a run on the whole continent instead of
the BRAZIL test set, build the test months of `..._test2016sa.json` (`SPLIT=test`) and predict with
`--override 'test_load_data.region="SOUTHAMERICA"' --use-saved-scalers`;
`smoke_tests_zarr/southamerica_test_split.py` splits the result into footprints inside / outside the
BRAZIL series.

### More than one region

`train_load_data.region` may also be a **list of regions of the same domain**; every month is
then loaded once per region (this works with the standard pipeline too). The regions must not
share footprints: a data set in which a time stamp occurs twice is refused, so
`["BRAZIL", "SOUTHAMERICA"]` raises an error.

### Several training sources, also of different domains (`extra_sources`)

`train_load_data.extra_sources` (month cache only; added 2026-10-14 for the cross-domain study,
experiment summary section 45) adds further regions to the training set, each with its own
years / months / `freq` and, if its domain's meteorology is stored elsewhere, its own
`data_dirs`; every other loading key (`size`, `met_args`, ...) is inherited from
`train_load_data`:

```json
"data_dirs": {"met_datadir": "/projects/b5bn/data/met_archive_zarr/NORTHAFRICA/NORTHAFRICA_Met_"},
"train_load_data": {
  "years": ["2012", "2013", "2014"], "freq": 2, "region": "SAHARA", "size": 50, "met_args": {...},
  "extra_sources": [
    {"region": "SOUTHAMERICA", "years": ["2013", "2014", "2015", "2017"],
     "data_dirs": {"met_datadir": "/projects/b5bn/data/met_archive_zarr/SOUTHAMERICA/SOUTHAMERICA_Met_"}}
  ]
},
"test_load_data": {"years": ["2015"], "freq": 10, "region": "SAHARA"}
```

- A source is cached under the key of a single-region run with that configuration, so the
  months of earlier single-region runs are reused (no cache build).
- The sources may belong to **different domains**: the auxiliary CAMS channels (the 16 boundary
  values appended to the inputs) are per domain and month, so each sample gets those of its own
  domain, exactly what a single-domain run of that domain would give it; the background target
  of every sample is its own domain's. The background normalisation is fitted on the union, so
  the normalised bg loss is not comparable with single-domain runs (compare in ppb).
- The samples are identified by their time stamp: the sources must not share one (the GOSAT
  overpass times of SOUTHAMERICA and NORTHAFRICA are 15-18 and 11-14 UTC).
- One graph serves all samples: the grid is built from the reference footprint
  (`grid_reference_fp`, default the first training footprint in time), as in any run.
- The test set inherits `train_load_data` without the extra sources and must be of one domain.
- `data_dirs` inside `train_load_data`, `test_load_data` or a source overrides the top-level
  `data_dirs` for that block only (month cache only).

## 2. `config.yml`

```yaml
user_paths:
  data_cache_dir: /projects/b5bn/public/Nawid/gates_data_cache
```

The cache needs about 1.6 GB per 1,000 footprints (50 x 50 window, 162 met / static channels).
It holds derived data only and can be deleted at any time; the next job rebuilds what it needs.

## 3. Building the cache

A training job builds the months it does not find in the cache by itself, one after another.
For a large data set it is faster to build the cache first, with several jobs at once:

```bash
PARAM_FILE=my_params.json sbatch --array=0-5 launch_prepare_data_cache.sh
```

Each array task loads every 6th month; months that are already cached are skipped, so the
command can be repeated to complete a cache. Without `--array` one job builds everything.

Measured (2026-09-29, job 6936348, six tasks): the BRAZIL and the SOUTHAMERICA series, 2013 +
2014 + 2015 + 2017, every footprint (`freq` 1) = 96 region-months, 165,922 footprints loaded
(67,310 + 98,612; each series has its own cache), 256 GB on disk, in 11–16 minutes of wall time
(75 minutes of loading in total, i.e. ~0.3 min per month plus ~0.2 min per 1,000 footprints).

**Memory of the build.** While it cuts the windows, the loader holds the full-domain met of
every footprint of the month: about 65 MiB per footprint, i.e. ~160 GiB for a month of 2,300
footprints. The launcher therefore asks for the memory of a whole node (no GPU needed).

## 4. What a job does, and what it costs

| stage | 67,310 samples (BRAZIL, 4 years) | 98,612 samples (SOUTHAMERICA, 4 years) |
|-------|---------------:|---------------:|
| read the cached months | 8 s | 13 s |
| fit the input scaler (same code, same subsample as the standard pipeline) | 44 s | 80 s |
| scale the inputs in place | 55 s | 86 s |
| footprints, backgrounds, test set, start of the GPU workers | 50 s | 58 s |
| **job start to first epoch** | **2.9 min** | **4.3 min** |
| training batches in shared memory | 115 GiB | 169 GiB |
| **peak host memory** (during the scaler fit) | **164 GiB** | **222 GiB** |
| host memory while training | 136 GiB | 190 GiB |
| one epoch (4 GPUs, per-GPU batch 5) | 114–125 s | 175–180 s |

(Jobs 6937642 and 6937574; epoch times also from 6937051 and 6937256. Enlarged trunk with two-way
release edges, test set 2,173 samples, every footprint of 2013 + 2014 + 2015 + 2017. Host memory =
anonymous + shared memory + kernel memory of the job, sampled every 2 s. For comparison, the
standard pipeline on 18,765 samples: 35 minutes to the first epoch, 227 GiB peak; on 98,612
samples it would need about 1,160 GiB.)

Rules of thumb for a 50 x 50 window:

- host memory = 1.75 GiB per 1,000 training samples (the tensors) + 50–55 GiB while the scaler
  is fitted (`fit_buffer_gib`, default 48); request `--mem` accordingly (the 300G of the
  launchers is enough for 98,612 samples). `/dev/shm` of an
  Isambard-AI node holds 334 GiB, i.e. ~190,000 samples;
- an epoch takes 1.8 ms per training sample on 4 GPUs at a per-GPU batch of 5, as with the
  standard pipeline (98,612 samples: 3 min; 300 epochs: 15 h, so raise `--time`). The step time,
  not the data, sets the pace: see [HOW_TO_MULTI_GPU.md](HOW_TO_MULTI_GPU.md) for the effect of
  the batch size. A larger data set gives more optimizer steps per epoch, so fewer epochs are
  needed for the same number of steps: 300 epochs of 18,765 samples are 281,400 steps, which is
  84 epochs of 67,310 samples (2.8 h) or 57 epochs of 98,612 (2.8 h). How many epochs a larger
  data set needs is NOT measured here.

Arms of one job that prepare the data in the same way (same scaler settings, batch size, seed,
background classes) share the training tensors: the second arm starts training at once.

## 5. When is a cached month reused?

A cached month is identified by everything that decides what the loader returns for it:

- the region, year and month;
- every other entry of `train_load_data` / `test_load_data` (`freq`, `size`, `met_args`, ...);
- the `variables` block, the background settings (`detrend`, auxiliary CAMS levels);
- the data directories of `config.yml` and any `data_dirs` override.

These are hashed into the directory name (`<cache>/<REGION>/<key>/<YYYY>-<MM>/`, with the full
configuration in `<key>/config.json`), so a changed setting never picks up a stale month. In
addition:

- the footprint files of the month are recorded (name, size, modification time); if they
  changed, the month is rebuilt;
- the **source code of the loader** is fingerprinted. A change does *not* invalidate the cache
  (unrelated edits would force a rebuild every time); instead the job prints

  `FLAG: month cache ...: the loader source changed since (some of) it was built ...`

  naming the files. If the change affects what is loaded, run once with
  `"data_cache": {"rebuild": true}` (or `REBUILD=1 ... launch_prepare_data_cache.sh`).

Scaling is *not* part of the cache: the months are stored raw, and the scalers are fitted in
every job on the training set of that job. The resolved cache directories are saved with the run
(`data_cache_resolved` in `training_settings_*.json`).

A larger input domain (`train_load_data.input_domain`, see
[HOW_TO_PARAMETER_FILE.md](HOW_TO_PARAMETER_FILE.md)) is applied here, between the loader and the
cache: the month is loaded on the input window and reduced to what the model trains on (the
footprint window's inputs plus block-averaged channels, or the full window with the footprints
masked outside their window) before it is written. Such months have their own key (the option is
an entry of `train_load_data`); at a 100 x 100 window a month holds 4x the data of a 50 x 50 month
in mode `"full"`, 2x in mode `"native_and_coarse"`, the same in mode `"coarse"`. The standard
pipeline refuses the option.

What the option has been found to do (experiment summary sections 30 and 31, BRAZIL 4 yr, 201 and 300
epochs): no footprint or mole-fraction gain in any mode, a consistent background gain of about −0.3 ppb
in modes `"native_and_coarse"` (34 s/epoch, 134 GiB) and `"full"` (118 s/epoch, 281 GiB). The
ready-made arm definitions are `parameter_files/parameter_dual_brazil4yr_f2_nolatlon_in{50native,100both,100full}_*.json`
with `parameter_files/experiments_dual_input_resolution_300ep_gpu4_brazil4yr_f2.json` (one arm per base
file via `--index`; `smoke_tests_zarr/run_input_resolution_production.sbatch` runs them in turn).
`scripts/compare_epoch_series.py` checks that a rerun reproduces an earlier run epoch by epoch.

Several coarse grids at once (section 36): `"input_domain": {"size": 150, "mode": "native_and_coarse",
"coarse_sizes": [100, 150]}` loads every month on the 150 x 150 window and caches the native 50 x 50 inputs
plus one block-mean grid per listed window (`ctx100_` = the central 100 x 100 window in 2 x 2 blocks,
the `ctx_` channels of a `{"size": 100}` run bit for bit except the topography block means, which differ
by one float32 ulp of the source heights because the 150 x 150 window pads the footprint domain in
latitude and the padded coordinate is float64; `ctx150_` = the whole window in 3 x 3 blocks):
464 channels per node, 3x the data of a 50 x 50 month. Arm definition
`parameter_files/parameter_dual_brazil4yr_f2_nolatlon_in150multi_*.json` with
`parameter_files/experiments_dual_input_resolution_multi_gpu4_brazil4yr_f2.json` (300 and 450 epochs;
`smoke_tests_zarr/run_input_resolution_multi_production.sbatch`); the cache is checked against the earlier
caches and a direct 150 x 150 load by `smoke_tests_zarr/check_input_domain_multi_cache.py`. No BRAZIL window
of 150 x 150 cells leaves the met domain (checked 2026-10-06 on the cached window coordinates: 0 of 33,666 +
2,173), so the coarse blocks are real meteorology; 43 % of them leave the footprint domain, which only pads the
footprint with NaN outside the central window that is kept.

The whole domain as an input grid (section 50, `gates/data/whole_domain.py`): `"input_domain": {"mode": "domain"}`
(the whole footprint domain in 50 x 50 bins with the release point marked REPLACES the window's met / topography;
160 channels) or `{"mode": "native_and_domain"}` (the window's 156 channels plus the whole domain as `dom_*` channels
and the four release channels; 314 channels), or `"whole_domain": true` next to `"native_and_coarse"`. These channels
need the month's loader object (the met archive on the domain grid, the domain topography, the release positions), so
they are built INSIDE the loader: `ensure_months_cached` hands `load_GATES_month_with_bg` an `inputs_hook`
(`DomainGridHook`) that derives them from the loader object; `apply_input_domain(..., domain=...)` then composes them
with the window's channels before the month is written. The window itself is loaded exactly as before (the window
channels of such a cache are the plain cache's bit for bit). Cost: the archive is read a second time per month (binned
lazily, one task per chunk); a 50 x 50 month holds the same (`domain`) or twice (`native_and_domain`) the data of a
plain month. Arm definitions `parameter_files/parameter_dual_brazil4yr_f2_nolatlon_{domonly,dom50both}_*.json` with
`experiments_dual_whole_domain_300ep_gpu4_brazil4yr_f2.json` (`smoke_tests_zarr/run_whole_domain_production.sbatch`,
trainer `train_whole_domain_model`); the caches are checked against the plain cache, a direct load with the hook and an
independent recomputation from the archive by `smoke_tests_zarr/check_whole_domain_cache.py`. Adding the hook argument
to `load_GATES_month_with_bg` (a fingerprinted loader file) makes every cache built before 2026-10-29 print the
"loader source changed" FLAG; nothing in their loading changed (the hook is None without the option).

The exit-curtain model (`train_exit_model.py`, [HOW_TO_EXIT_CURTAINS.md](HOW_TO_EXIT_CURTAINS.md)) keeps
the boundary exit curtains of the footprints in a **side file** of each cached month
(`<YYYY>-<MM>/exit_curtains.npz` + `exit_curtains_meta.json`, `gates/data/exit_curtains.py`): built on
first use from the footprint files for exactly the month's cached footprint times, checked against the
month's cached background, and gone with the month when the month is rebuilt. The month cache itself
does not know about these files.

Months the loader cannot build are skipped, as in the standard pipeline, and listed in a
`FLAG:` line; they are not cached, so the next job tries again.

## 6. Verification

| check | result |
|-------|--------|
| standard pipeline after the refactor vs. tensors saved before it | identical, bit for bit |
| month cache vs. standard pipeline, 2 real training months (841 samples): training and test tensors, scaler statistics, normalisation values, footprint labels, NumPy / torch RNG state | identical, bit for bit (jobs 6936186, 6936236) |
| same, synthetic months: 4 batch sizes / subsample fractions, months in any order, threads | identical, bit for bit (`tests/test_lean_dual_data.py`) |
| 4-GPU training, one epoch, 5 launches x 3 arms: standard pipeline, month cache, month cache rebuilt by every arm | weights after the epoch identical, bit for bit, in all 15 runs (job 6937105) |
| training tensors before / after training | unchanged (checksums) |
| 2 epochs on 67,310 and on 98,612 samples, 4 GPUs | run to the end, no out-of-memory event (jobs 6937051, 6937256) |
| `predict_dual_model.py` on a run trained from the month cache and on a standard run | both reproduce the test losses of the training log (job 6937815) |

The bit-for-bit agreement of the trained weights needs the fix of the encoder's mesh graph of
2026-09-29 (`model/layers/encoder.py`: neighbours in sorted order). Before it, the edge order of
the mesh graph changed from process to process, and any two launches of the same run, with either
pipeline, differed in the last digits.

## 7. Prediction

`predict_dual_model.py` works on runs trained from the month cache. Such a run's training inputs
are not loaded into memory, so the input scaler cannot be refitted: the script applies the run's
**saved** input scaler and background normalisation (as `--use-saved-scalers` does) and says so.

To predict a run on a test region of **another domain** (cross-domain test, section 45), override
the test block only, with its own `data_dirs`; a top-level `data_dirs` override would change the
training months' cache key and reload them with the wrong meteorology:

```
python predict_dual_model.py --run_dir <SOUTHAMERICA run> --checkpoint best_fp \
    --override 'test_load_data.region="SAHARA"' --override 'test_load_data.years=["2015"]' \
    --override 'test_load_data.data_dirs={"met_datadir": "/projects/b5bn/data/met_archive_zarr/NORTHAFRICA/NORTHAFRICA_Met_"}' \
    --out sample_predictions_sahara2015_best_fp.nc
```

The model keeps its own graph (built from its training footprints); the window of the other
domain is fed as the same 50 x 50 index grid (both domains have 0.234 x 0.352 deg cells). The
background head is applied with the run's saved normalisation, so its output is the model as
trained; its bias on another domain is part of the result.

## 8. Files

| file | content |
|------|---------|
| `gates/data/month_cache.py` | the on-disk cache of one region + loading configuration |
| `gates/training/lean_dual_data.py` | building the training tensors from cached months |
| `prepare_dual_data_cache.py`, `launch_prepare_data_cache.sh` | building a cache ahead of training |
| `tests/test_lean_dual_data.py` | unit tests (CPU, seconds) |
| `smoke_tests_lean/` | throwaway: comparisons on real data, capacity probes |

## Choosing which samples of the cached pool are used (`sample_selection`)

The loader's `freq` keeps every k-th sounding of a month. A top-level `sample_selection` block
chooses a subset of the loaded pool by a rule instead — `random` (seeded draw), `stride` (evenly
spaced in time), `coverage` (k-means on footprint / season / position descriptors, budget per
cluster proportional to `n_cluster ** temper`, evenly spaced in time inside a cluster) or `month`
(the same number of samples from every calendar month, evenly spaced in time inside a month;
section 54) or `direction` (the same number from every transport direction of the footprint; section 56) — of a given size (`n_samples` or `fraction`); `offset` repeats a rule on other samples. It is applied inside the lean pipeline, on the small data of the
cached months before any input array is read, so the shared-memory tensor holds the selected samples
only (reading the selected rows of each month's `inputs.npy` through a memory map), and the pool's
cache is reused: the block is not part of the cache key and not a data-loading key of the shared-data
driver, so the arms of one job may select differently. Recorded in the settings
(`sample_selection_resolved`) and in `sample_selection_<model>.csv`. Details and the key table:
[HOW_TO_PARAMETER_FILE.md](HOW_TO_PARAMETER_FILE.md#sample_selection); module
`gates/training/sample_selection.py`; tests `tests/test_sample_selection.py`; first use: experiment
summary section 53 (random vs coverage at budgets of 2,048 and 8,192 of the 33,666-footprint BRAZIL
freq-2 pool).
