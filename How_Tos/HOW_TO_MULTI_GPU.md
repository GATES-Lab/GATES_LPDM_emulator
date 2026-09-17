# HOW TO: multi-GPU training (dual-head trainers)

The three dual-head trainers (`train_dual_model.py`, `train_dual_refit_model.py`,
`train_dual_headlr_model.py`) and the drivers built on them
(`run_dual_experiments_shared_data.py`, `run_dual_sweep_wandb.py`) can train on several GPUs of
**one node**. The implementation lives in
[gates/training/distributed.py](../gates/training/distributed.py). With the default settings
nothing changes: a parameter file without a `distributed` block runs the single-GPU code path
exactly as before.

## 1. Read this first: it changes the effective batch size

The dual model is small (~270k parameters) and its training step is limited by kernel-launch
overhead, not by GPU compute. Measured on one GH200 (synthetic data, deterministic mode,
`smoke_tests_multigpu/bench_step.py`, job 6626917):

| batch size | ms / step | ms / sample |
|-----------:|----------:|------------:|
| 1          | 26.4      | 26.4        |
| 5          | 27.5      | 5.5         |
| 10         | 30.6      | 3.1         |
| 20         | 52.6      | 2.6         |
| 40         | 99.2      | 2.5         |

A step costs the same for any batch size up to ~10, so splitting the usual batch of 5 across
GPUs would not be faster at all. Multi-GPU training therefore gives **every GPU its own full
batch**:

- `dataloader.batch_size` is the **per-GPU** batch size;
- each optimizer step averages the gradients of `num_gpus` consecutive batches, i.e. the
  **effective batch size is `batch_size × num_gpus`** and an epoch has `num_gpus`× fewer
  optimizer steps.

This is a change to the optimisation, exactly like raising `batch_size` on one GPU:

- a multi-GPU run is **not directly comparable** with single-GPU runs of the same parameter
  file (including all sweeps run before this feature existed);
- per epoch the model takes `num_gpus`× fewer steps, so expect to need more epochs and/or
  re-tuned learning rates for the same result. Nothing is rescaled automatically — learning
  rates stay exactly as written in the parameter file;
- the resolved values are saved with the run (`distributed_resolved` in
  `training_settings_*.json` and the W&B config): `num_gpus`, `per_gpu_batch_size`,
  `effective_batch_size`.

If you need results comparable with earlier single-GPU runs, keep `num_gpus: 1`.

## 2. Parameter file

```json
"distributed": { "num_gpus": 4 }
```

| key | values | meaning |
|-----|--------|---------|
| `distributed.num_gpus` | int ≥ 1 (default `1`) or `"auto"` | GPUs to train on. `"auto"` = every GPU visible to the job. Asking for more GPUs than the job has is an error (raised before the data load). If the job has more GPUs than are used, the log prints a `FLAG:` line. |
| `dataloader.in_memory_batches` | bool (default `false`) | Single-GPU opt-in to the in-memory batch loader (section 4). Always on when `num_gpus > 1`. |

Both can be set per experiment in an experiments file (they are not data-loading keys), e.g.
`{"name": "gpu4", "overrides": {"distributed": {"num_gpus": 4}}}`.

## 3. Launching on Isambard-AI

Request the GPUs on the `sbatch` command line **and** enable them in the parameter file:

```bash
PARAM_FILE=my_params.json sbatch --gres=gpu:4 launch_dual_zarr_isambard.sh
```

- No `srun`/`torchrun` is needed: the job stays a single task. The training process spawns one
  worker process per extra GPU itself, after the data has been loaded.
- **Keep the launcher's normal CPU request; do not ask for the whole node's 288 CPUs.** With
  288 CPUs the data load was ~5x slower (2.2 vs 0.45 min per month, jobs 6629378 vs 6627279)
  and training was no faster, even with every process pinned next to its GPU. (`make_cluster`
  now also caps the Dask cluster at 64 workers — `GATES_DASK_MAX_WORKERS` — because a
  286-worker cluster failed to start at all, job 6628781.)

Measured speed-up (smoke configuration: 1 year at freq 10 = 331 training batches of 5, 4
epochs of the head-LR trainer incl. freeze and refit, deterministic mode; seconds per epoch
including validation, metrics and checkpoints; jobs 6627279 / 6629378):

| setup | s / epoch | optimizer steps / epoch |
|-------|----------:|------------------------:|
| 1 GPU, default loader        | ~14 | 331 |
| 1 GPU, `in_memory_batches`   | ~12 | 331 |
| 2 GPUs                       | ~9.5 | 165 |
| 4 GPUs                       | ~4-5 | 82 |

These are short smoke runs; per-epoch fixed costs (validation, metrics, plots) are a larger
share than in a production run.

### Full-length comparison (one run each, 2026-09-17)

Head-LR trainer, train 2014+2015 / test 2016, continuous joint 150 epochs, `bg_loss_weight`
0.1, trunk/fp LR 5e-5, bg LR 5e-6, deterministic; identical parameters except
`distributed.num_gpus` (learning rates NOT rescaled). 1 GPU = W&B run `kyqkwfbu` (job
6500993); 4 GPUs = job 6632597, W&B group `isambard_multigpu`. Plot:
`model_runs/multigpu_vs_1gpu_2016test.png`, made with `scripts/compare_multigpu_run.py`.

| | 1 GPU (batch 5) | 4 GPUs (effective batch 20) |
|---|---:|---:|
| s / epoch | 127.7 | 23.2 (5.5x: 4 GPUs + in-memory loader) |
| 150 epochs | 5.4 h | 1.1 h |
| optimizer steps / epoch | 2262 | 565 |
| best test fp loss | 0.4286 @ epoch 146 | 0.4206 @ 149 |
| best test bg loss | 0.0990 @ 144 | 0.1138 @ 149 (still falling) |
| bg test-train gap at end | +0.031 | +0.023 |

Reading: per epoch the 4-GPU run learns more slowly early on (4x fewer steps) but catches up
on the footprint head by ~epoch 90 and ends slightly better; the background head (whose LR is
already 10x lower) has not converged within 150 epochs and ends ~15% worse. In wall-clock
terms the 4-GPU run is ahead on the footprint loss throughout. This is a single pair of runs
(one seed), so differences of ~0.01 are within what reruns have shown; whether more epochs or
a higher bg LR closes the bg gap is **untested**.

## 4. How it works

1. **The data is loaded once**, by the main process, exactly as before. The ready-made batches
   of the train and test loaders are then copied into shared-memory tensors (same batches,
   same order as the normal loader yields every epoch). Worker processes map that memory; the
   dataset is never duplicated per GPU. The shared-data experiment drivers keep working: one
   load, arms run one after another, each arm starts and stops its own workers.
2. **The main process is rank 0** and keeps all output: W&B, the updates log, checkpoints
   (`*_best.pt`, `*_best_fp.pt`, `*_best_bg.pt`, periodic), metrics, plots, NetCDF export.
   Checkpoints have the same format as single-GPU ones (no `module.` prefix). Workers print
   nothing except errors.
3. Rank `r` of `W` trains on batches `r, r+W, r+2W, …`; after `backward()` the gradients are
   averaged over all ranks with one all-reduce, then every rank takes the same optimizer step.
   Up to `W-1` trailing training batches per epoch are dropped so all ranks take the same
   number of steps (logged). Plain gradient averaging is used instead of
   `DistributedDataParallel` because the trainers freeze/unfreeze heads and swap optimizers
   mid-run (bg freeze, refit), and the model is small enough that it costs nothing.
4. **Validation is sharded too**, and the per-batch losses and predictions are gathered back in
   the original order, so test losses/metrics are computed over exactly the same batches as on
   one GPU, and every rank sees identical epoch losses — early stopping, the bg-head freeze and
   the refit schedule therefore trigger identically everywhere.
5. Logged train losses are the mean over all ranks; the in-epoch `[epoch, i]` progress lines
   show rank 0's shard only.

`in_memory_batches` on a single GPU uses step 1 alone: same batches, same order, same initial
weights, but served from memory instead of going through xarray/xbatcher every epoch. In the
smoke test it reproduced the default loader's train/test losses to every printed digit over 4
epochs (through freeze and refit, deterministic mode, same process) and was ~15% faster. It
needs host memory for a second copy of the batched training data.

## 5. Limitations

- Single node only (≤ 4 GPUs on Isambard-AI).
- Only the dual-head trainers; `launch_train.sh` / the footprint-only and boundary-only
  trainers are unchanged.
- `dual_analysis` trunk-gradient diagnostics are measured on rank 0's batches only.
- `deterministic: true` gives the same guarantees as before (deterministic kernels, not
  bit-exact across processes/nodes); multi-GPU adds NCCL reductions to that.
- If a worker process dies, the main process aborts the whole job (`FATAL: multi-GPU worker
  ...`) rather than hanging; in a sequential sweep the remaining arms do not run.

## 6. Tests

- `python -m pytest tests/test_distributed.py` (CPU, seconds): sharding, padding/gather order,
  config validation, and a two-process check that gradient averaging reproduces a single
  process stepping on the combined batch (also with a frozen head).
- `smoke_tests_multigpu/run_smoke_multigpu.sbatch` (throwaway, 4 GPUs): end-to-end runs of the
  head-LR trainer on 1 GPU (default and in-memory loader), 4 and 2 GPUs including the bg
  freeze → refit transitions, plus `train_dual_model.py` with `num_gpus: "auto"`.
