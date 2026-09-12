# How-To: Dual-Head Training with the Background Freeze / Refit Schedule

This document describes [`train_dual_refit_model.py`](../train_dual_refit_model.py): what the
dual (footprint + background) model is, how the joint loss is composed, and how the
`bg_schedule` block freezes and re-fits the background head during a run. It is written for
someone who has not seen the script before and wants to run or extend it.

The script is a **sibling** of [`train_dual_model.py`](../train_dual_model.py) (the standard
dual-head entry point, unchanged). Everything up to the training loop — data loading,
normalisation, dataloaders, model/optimizer construction, footprint metrics — is imported from
`train_dual_model` and identical. Only the epoch loop differs. **With no `bg_schedule` block in
the parameter file the script behaves exactly like `train_dual_model.py`.**

Contents:

1. [The dual model in one paragraph](#1-the-dual-model-in-one-paragraph)
2. [Data pipeline](#2-data-pipeline)
3. [The joint loss](#3-the-joint-loss)
4. [Per-head optimizer controls: `bg_head`](#4-per-head-optimizer-controls-bg_head)
5. [The freeze / refit schedule: `bg_schedule`](#5-the-freeze--refit-schedule-bg_schedule)
6. [What happens at each transition](#6-what-happens-at-each-transition)
7. [Common schedules (recipes)](#7-common-schedules-recipes)
8. [Checkpoints and outputs](#8-checkpoints-and-outputs)
9. [Logging: W&B and `*_updates.txt`](#9-logging-wb-and-_updatestxt)
10. [Reproducibility](#10-reproducibility)
11. [Running it](#11-running-it)
12. [Testing](#12-testing)
13. [Gotchas and known issues](#13-gotchas-and-known-issues)

---

## 1. The dual model in one paragraph

`GraphSatelliteDualForecaster` ([`model/forecast.py`](../model/forecast.py)) is the GATES GNN
with **one shared trunk and two decoder heads**:

```
met + aux-CAMS inputs ──► encoder ──► processor ──┬──► fp_decoder  ──► "footprint"  [B, num_nodes, 1]
                          (shared trunk)          └──► bg_decoder  ──► "background" [B, num_classes]
```

- The **footprint (fp) head** predicts the per-node LPDM footprint (the original GATES task).
- The **background (bg) head** predicts the boundary-condition contribution to the satellite
  column: one number per sample (`num_classes: 1`, the summed background) or four
  (`num_classes: 4`, north/south/east/west). See
  [HOW_TO_BOUNDARIES.md](HOW_TO_BOUNDARIES.md) for the background data itself.
- `forward` runs the trunk **once** and returns `{"footprint": ..., "background": ...}`. Both
  objectives therefore push on the same trunk parameters — this "trunk competition" is the whole
  reason the freeze/refit schedule exists.

Throughout the script and this doc, **"trunk"** means `model.encoder` + `model.processor`,
**"fp head"** means `model.fp_decoder`, and **"bg head"** means `model.bg_decoder`.

## 2. Data pipeline

All of this is shared with `train_dual_model.py` and is only summarised here (see
[HOW_TO_DATA.md](HOW_TO_DATA.md) and [HOW_TO_BOUNDARIES.md](HOW_TO_BOUNDARIES.md) for detail).

| Step | Where | Notes |
|------|-------|-------|
| Raw load (met, footprints, backgrounds, aux CAMS) for train **and** test | `train_dual_model.load_dual_data` | The expensive step; returns a `DualDataBundle`. The shared-data driver (section 11) loads it once and reuses it for every experiment arm. `background_setup` (`detrend`, `use_auxiliary_bc`, `auxilary_bc_levels`) is resolved by `resolve_background_params` and controls detrending / aux-feature loading. |
| Background class selection | `train_and_save_model` | `model_parameters.num_classes` = 1 → `summed`; 4 → `north/south/east/west`. Anything else falls back to 1 with a warning. |
| Aux CAMS formatting + **background normalisation** | `format_aux_data`, `normalize_boundary_data` | Normalisation stats are fit on **train only** and applied to test; they are saved as `norm_vals` (used later to report a denormalised bg MAE and the ppb time-series plots). |
| Input scaling, footprint scaling, dataloaders | `gates_training_dual.setup_dual_dataloaders` | Inputs scaled by `input_scaler`, footprints by `fp_scaler` (fit on train), aux CAMS concatenated onto the scaled inputs. Each batch is `(inputs, fps, background)`. |
| Model + optimizer + criteria | `gates_training_dual.setup_dual_model` | See sections 3 and 4. |

Artifacts saved before training starts (all under `<model_path>/training_outputs/`, and to W&B
when enabled): `norm_vals_*.json`, `scalers_*.pickle`, `training_settings_*.json` (the fully
resolved parameter dict, including `background_setup`, `num_features` and `plotted_dates`), and
`grid_*.pickle`. Together with the checkpoints these make a run reconstructable (principle 2).

## 3. The joint loss

Configured under `loss_functions` with separate entries per head (the single-head keys
documented in [HOW_TO_PARAMETER_FILE.md](HOW_TO_PARAMETER_FILE.md#loss_functions) are **not**
used by the dual scripts):

```json
"loss_functions": {
  "fp_criterion":        "gates_losses.PixelWeightedMSELoss",
  "fp_criterion_params": {"weight_label": "fp_original", "transform_fn": "scale_and_shift", "w": 1000, "a": 0.5},
  "fp_criterion_test":   "gates_losses.MSELoss",
  "fp_criterion_test_params": {},
  "bg_criterion":        "torch.nn.MSELoss",
  "bg_criterion_test":   "torch.nn.MSELoss",
  "bg_criterion_params": {},
  "bg_loss_weight":      0.1
}
```

| Key | Meaning |
|-----|---------|
| `fp_criterion` / `fp_criterion_params` | Training loss for the footprint head. Constructed with `fp_labels` and the nan-mask label (`fp_nan_mask` when `dataloader.nans_to_zeros` is true, so NaN pixels are excluded). |
| `fp_criterion_test` / `fp_criterion_test_params` | Test loss for the fp head; defaults to the training criterion / its params if unset. Usually plain MSE so test numbers are comparable across weightings. |
| `bg_criterion` / `bg_criterion_test` / `bg_criterion_params` | Same for the background head, on the **normalised** background targets. |
| `bg_loss_weight` (`w`) | The mixing weight. |

During normal joint training each batch minimises

```
total = (1 - w) * fp_loss + w * bg_loss
```

and `validate_and_predict` computes the test `total` with the **same formula, in every phase**
(it uses the `*_test` criteria). Per-head train/test losses are always logged separately, so you
can read `fp` and `bg` regardless of `w`.

Two consequences of the `(1 - w) / w` scaling that matter for the schedule below:

- `w = 0` zeroes the bg term (bg head receives no gradient; its metrics are meaningless).
- `w = 1` zeroes the fp term (fp head receives no gradient). In the frozen / fp-first phase the
  loss is `(1 - w) * fp_loss`, which at `w = 1` is **exactly zero — nothing trains for the whole
  phase**. Never combine `w = 1` (or `w = 0`) with a schedule; if you want "fp first, then bg
  only", use `refit_mode: "bg_trunk"` (section 5).

## 4. Per-head optimizer controls: `bg_head`

Optional block read by `setup_dual_model` (independent of, and older than, `bg_schedule`):

```json
"bg_head": {
  "freeze_patience": 25,     // freeze the bg head after this many epochs without bg test-loss
                             // improvement (null / absent = never)
  "freeze_min_delta": 0.0,   // minimum decrease that counts as an improvement
  "weight_decay": 0.05,      // AdamW weight decay for bg_decoder params only (null = default)
  "lr_scale": 1.0            // bg_decoder LR = learning_rate * lr_scale
}
```

- If `weight_decay` or `lr_scale` are set, the bg decoder gets its **own AdamW parameter group**;
  otherwise there is a single `AdamW(model.parameters(), lr=learning_rate)`.
- `freeze_patience` is the **patience-based** freeze: `HeadCheckpoint("bg").counter` counts epochs
  since the bg test loss last improved by more than `freeze_min_delta`; once it reaches the
  patience the head is frozen. In `train_dual_refit_model.py` this fallback is only consulted
  when `bg_schedule.freeze_epoch` is `null` and the schedule is still in the `joint` phase.

## 5. The freeze / refit schedule: `bg_schedule`

```json
"bg_schedule": {
  "freeze_epoch": 75,        // freeze the bg head at the START of this epoch (null = use bg_head.freeze_patience)
  "refit_start_epoch": 120,  // unfreeze + refit from the START of this epoch (null = never refit)
  "refit_mode": "bg_trunk",  // "bg_trunk" (default) or "joint"
  "refit_lr_scale": 1.0      // bg-decoder LR during the refit = learning_rate * refit_lr_scale
}
```

`refit_mode` must be `"bg_trunk"` or `"joint"`; if both epochs are set, `refit_start_epoch` must
be `> freeze_epoch`. Any other keys in the block are silently ignored.

The run moves through up to three phases. `sched.phase` is read every epoch by
`train_one_epoch`, which changes the loss composition, which optimizers step, and which modules
are held in `eval()` mode:

| Phase (`bg/phase` code) | Epochs | Loss backpropagated | Trunk | fp head | bg head | Optimizers stepping |
|---|---|---|---|---|---|---|
| `joint` (0) | `[0, freeze_epoch)` | `(1-w)·fp + w·bg` | trains | trains | trains | main |
| `frozen` (1) | `[freeze_epoch, refit_start_epoch)` | `(1-w)·fp` (bg logged on **detached** preds) | trains | trains | **frozen**, `eval()` | main |
| `refit_joint` (3) | `[refit_start_epoch, end)` | `(1-w)·fp + w·bg` | trains | trains | trains (fresh optimizer) | main + bg |
| `refit_bg_trunk` (4) | `[refit_start_epoch, end)` | `bg` only, **unweighted** (fp logged on detached preds) | trains | **frozen**, `eval()` | trains (fresh optimizer) | main + bg |

Notes on the table:

- **The fp term keeps its `(1-w)` scaling in the frozen phase**, so the fp gradient magnitude is
  unchanged by the freeze (comparable to the joint phase). In `refit_bg_trunk` the bg loss is
  deliberately **not** multiplied by `w` — with a single term the weight would only rescale the
  effective learning rate.
- **The refit always trains the trunk.** A head-only refit (trunk + fp frozen) was tried in an
  earlier version and cannot recover a trunk that has drifted under fp-only training; that mode
  was removed. The two remaining modes answer different questions:
  - `bg_trunk`: "How good can the bg head get if the trunk is allowed to re-specialise for it,
    and what does that cost the (now fixed) fp head?" — the fp test loss during this phase is the
    price of bg-directed trunk movement. Use the fp checkpoint from *before* the refit for fp
    predictions.
  - `joint`: "Does resuming joint training recover the bg head, and does re-fitting it make the
    fp head drift?"
- `freeze_epoch: 0` is the **fp-first** setting: the bg head never trains before the refit (it
  stays at random initialisation); combined with `refit_mode: "joint"` this is "train fp alone,
  then bring bg online". Because the bg head is untrained at that point it typically needs a
  larger LR (`refit_lr_scale` 10 has been used) — see section 13 for the observed behaviour.

## 6. What happens at each transition

Scheduled transitions are applied at the **start** of an epoch, before any batches
(`sched.maybe_freeze(...)` then `sched.maybe_start_refit(...)` in `run_full_training`).

**Freeze** (`BgSchedule.freeze`, also used by the patience fallback at the *end* of an epoch):
1. `gates_training_dual.freeze_bg_head(model)` → `requires_grad_(False)` on all `bg_decoder` params.
2. `model_ctx.bg_frozen = True`, `phase = "frozen"`, `bg_frozen_epoch = epoch`.
3. A line `Freezing bg head at epoch N (<reason>)` is printed and written to `*_updates.txt`.

**Refit start** (`BgSchedule.maybe_start_refit`), when `epoch >= refit_start_epoch` and the
phase is still `joint` or `frozen`:
1. If the head was never frozen (phase still `joint`) a warning is logged and the refit proceeds
   from the live head.
2. `requires_grad_(True)` on all `bg_decoder` params.
3. The bg decoder's params are **removed from the main optimizer's param groups** and their
   stale AdamW state is dropped, so the two optimizers never both step the same tensors.
4. A **fresh** `AdamW(model.bg_decoder.parameters(), lr = learning_rate * refit_lr_scale)` is
   created as `sched.bg_optimizer`. The trunk and fp head keep the original optimizer *and its
   moment estimates*, so their dynamics stay comparable to a run without a refit.
5. In `bg_trunk` mode, `requires_grad_(False)` on `fp_decoder` and `phase = "refit_bg_trunk"`;
   otherwise `phase = "refit_joint"`. `model_ctx.bg_frozen = False`.
6. Reference values are recorded for the `refit/` metrics: `pre_refit_bg_best`
   (`HeadCheckpoint("bg").best_loss` at that moment) and `fp_test_at_refit_start` (last test fp
   loss). A third `HeadCheckpoint("bg_refit", ..._best_bg_refit.pt)` starts tracking the best bg
   test loss **within the refit**.
7. A summary line (`Starting bg refit at epoch N (mode=..., lr=..., fresh bg optimizer): ...`) is
   printed and written to `*_updates.txt`.

If `refit_start_epoch` is beyond the last epoch a warning is printed at the start of training
and no refit runs. If early stopping fires (section 8) before the refit epoch, no refit runs
either.

## 7. Common schedules (recipes)

All with `epochs.training` ≥ the last transition epoch. Existing experiment files under
[`parameter_files/`](../parameter_files/) are given as examples (their `wandb.notes` fields
explain the purpose of each arm).

| Recipe | `bg_schedule` | Example file |
|--------|---------------|--------------|
| Continuous joint training (control; identical to `train_dual_model.py`) | `{"freeze_epoch": null, "refit_start_epoch": null}` (needed to *cancel* a base-file freeze — see override semantics in section 11) | `experiments_dual_joint300_fpfirst300.json` (`joint300_bg010`) |
| Freeze only | `{"freeze_epoch": 75, "refit_start_epoch": null}` | `experiments_dual_bg_refit.json` (`frozen_only`) |
| Freeze, then bg-specialist refit | `{"freeze_epoch": 75, "refit_start_epoch": 160, "refit_mode": "bg_trunk"}` | `experiments_dual_bg_refit2.json` |
| Freeze, then resume joint | `{"freeze_epoch": 75, "refit_start_epoch": 120, "refit_mode": "joint"}` | `experiments_dual_bg_refit.json` (`refit_joint`) |
| fp-first, then joint | `{"freeze_epoch": 0, "refit_start_epoch": 150, "refit_mode": "joint", "refit_lr_scale": 10}` | `experiments_dual_fp150_joint100_part1.json`, `experiments_dual_refit_lr_scale.json` |
| Interrupted joint (joint → fp-only → joint) | `{"freeze_epoch": 100, "refit_start_epoch": 150, "refit_mode": "joint", "refit_lr_scale": 10}` | `experiments_dual_joint_fp_joint_bg0p5.json` |
| Pure-background baseline | no schedule, `bg_loss_weight: 1.0` (fp head untrained — ignore its metrics) | `experiments_dual_bgonly_baseline.json` |

A single-run "best of both heads" recipe that the study converged on: co-train jointly, then
fork into `bg_trunk`; deploy `*_best_fp.pt` (from the joint phase) for footprints and
`*_best_bg_refit.pt` (from the fork) for the background.

## 8. Checkpoints and outputs

Everything lands in `<model_save_dir>/<model_name>_<YYYYmmdd_HHMMSS>/` (`model_save_dir` from the
parameter file or `config.yml`'s `save_models_dir`). All checkpoints below are **full model
state dicts** (the heads share the trunk, so a per-head checkpoint must include it); at predict
time load the file for the head you care about and use only that head's output.

| File | Written by | Criterion |
|------|-----------|-----------|
| `<name>_best.pt` | `EarlyStopping` | best **test total** loss `(1-w)·fp + w·bg` (always the joint formula). Also the early-stopping monitor: training stops after `epochs.patience` epochs without improvement — set `patience` = `training` to disable, which every schedule experiment does so the refit epoch is reached. |
| `<name>_best_fp.pt` | `HeadCheckpoint("fp")` | best test fp loss over the whole run |
| `<name>_best_bg.pt` | `HeadCheckpoint("bg")` | best test bg loss over the whole run (`bg_head.freeze_min_delta` is its improvement threshold) |
| `<name>_best_bg_refit.pt` | `HeadCheckpoint("bg_refit")` | best test bg loss **within the refit phase only** — exists so the refit result is kept even when it never beats the pre-freeze best |
| `<name>_<epoch>.pt` | periodic, every `epochs.model_save` | dict with `epoch`, `model_state_dict`, `optimizer_state_dict` (**main optimizer only**), `loss` (the losses dict), `learning_rate` |
| `training_outputs/*` | before training | `norm_vals`, `scalers`, `training_settings`, `grid` (section 2) |
| `<name>_updates.txt` | throughout | human-readable log: per-50-batch running losses tagged with the phase, per-epoch summary, every transition, and the final run summary |
| `training_imgs/` | every `epochs.visualize` | footprint prediction panels (`save_training_plots`) and bg time-series panels in ppb (`save_bg_timeseries_plots`; denormalised, and marked as detrended if `background_setup.detrend`) |
| `<name>_predictions.nc` (via `export_results_to_netcdf`) | end of run | the test footprint dataset with `fp_pred` / `fp_transformed_pred` from the **final** epoch plus `bg_true_ppb` / `bg_pred_ppb` |

The final summary line (printed and in `*_updates.txt`) gives both per-head bests with their
epochs and files, the freeze epoch, and — if a refit ran — the mode, the pre-refit bg best, the
refit-phase bg best, and the fp test loss at refit start vs. at the end.

## 9. Logging: W&B and `*_updates.txt`

W&B setup is the same as the standard trainer ([HOW_TO_WandB.md](HOW_TO_WandB.md)), with two
additions: `wandb.notes` (free text shown in the run's Notes field, handy as a per-arm override
explaining the arm) and run naming `<SLURM_JOB_ID>_<SLURM_JOB_NAME>_<suffix>` where `suffix` is
the experiment name when driven by the experiments driver, else `bg<bg_loss_weight>`.

Per-epoch keys (logged with `step=epoch`):

| Key | Meaning |
|-----|---------|
| `train/loss`, `test/loss` | total loss. **Train** total follows the phase's loss composition (section 5); **test** total is always `(1-w)·fp + w·bg`. |
| `train/loss_fp`, `test/loss_fp`, `train/loss_bg`, `test/loss_bg` | per-head losses, always logged (detached where the head is inactive) |
| `test/bg_mae_denorm` | bg MAE in original (denormalised) units |
| `best/test_loss_fp`, `best/test_loss_bg`, `best/epoch_fp`, `best/epoch_bg` | running per-head bests |
| `best/objective` | `best_fp + objective_bg_weight · best_bg` (`sweep.objective_bg_weight`, default 1) — the sweep objective |
| `bg/frozen` | 1 in the `frozen` phase, else 0 |
| `bg/phase` | phase code: 0 joint, 1 frozen, 3 refit_joint, 4 refit_bg_trunk (see `PHASE_CODES`) |
| `refit/best_test_loss_bg` | best bg test loss within the refit so far (only once a refit has started) |
| `refit/bg_gap_vs_pre_refit_best` | `refit best − pre-refit best` (negative = the refit beat the pre-freeze minimum) |
| `refit/fp_drift` | `current test fp − test fp at refit start` (positive = the refit is costing the fp head) |
| `metrics_transformed-*`, `metrics_original-*`, `metrics_fluxes_static/<mode>/*` | the standard GATES footprint metric pipeline on the test set (see [HOW_TO_evaluation.md](HOW_TO_evaluation.md)) |
| `fps_epoch_<N>`, `bg_timeseries_epoch_<N>` | the diagnostic images |

Run-summary keys (set once at the end): `best/epoch_fp`, `best/epoch_bg`, `bg/frozen_epoch`,
`refit/started_epoch`, `refit/mode`, and if a refit ran `refit/best_test_loss_bg`,
`refit/best_epoch_bg`, `refit/pre_refit_best_test_loss_bg`, `refit/fp_test_loss_at_start`,
`refit/fp_test_loss_final`. The three head checkpoints are uploaded once as artifacts
`best_fp`, `best_bg`, `best_bg_refit` (not per-improvement, to avoid artifact spam; `*_best.pt`
is uploaded per improvement by `EarlyStopping` as before).

## 10. Reproducibility

- `seed` (default 34) → `set_reproducibility`: seeds Python / NumPy / torch (+CUDA), sets cuDNN
  deterministic and `benchmark=False`.
- That alone does **not** make GPU runs bit-identical: the GNN's scatter operations use atomics,
  and identically configured arms have been observed to differ by ~0.01–0.03 in test loss
  (treat smaller cross-arm differences as noise).
- `"deterministic": true` (top-level, default off) additionally sets
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `torch.use_deterministic_algorithms(True)`. Verified
  bit-identical on GH200 by [`tests/gpu_determinism_test.py`](../tests/gpu_determinism_test.py);
  the speed cost has not been measured.
- The fully resolved parameter dict (including the `bg_schedule` block) is saved as
  `training_settings_*.json` and pushed to `wandb.config`.

## 11. Running it

**Directly** (one run, its own data load):

```bash
python train_dual_refit_model.py parameter_dual_refit_isambard.json          # from config.yml's parameter_files_dir
python train_dual_refit_model.py my_params.json --file_path /path/to/dir
```

**Several arms sharing one data load** (the normal way for the schedule experiments), via
[`run_dual_experiments_shared_data.py`](../run_dual_experiments_shared_data.py) with
`--trainer train_dual_refit_model`:

```bash
python run_dual_experiments_shared_data.py parameter_dual_refit_isambard.json experiments_dual_bg_refit.json --list
python run_dual_experiments_shared_data.py parameter_dual_refit_isambard.json experiments_dual_bg_refit.json --trainer train_dual_refit_model
python run_dual_experiments_shared_data.py ... --trainer train_dual_refit_model --index 1     # a single arm
```

The experiments file is `{"experiments": [{"name": ..., "overrides": {...}}, ...]}`. Override
semantics (from `gates/training/experiment_utils.py`):

- Overrides are **deep-merged** onto the base file: nested dicts merge key by key, any non-dict
  value (including lists and `null`) replaces the base value. So `"bg_schedule":
  {"freeze_epoch": null}` cancels a base-file freeze while keeping the base file's other schedule
  keys, and dotted keys such as `"loss_functions.bg_loss_weight": 0.5` address nested fields.
- `model_name` becomes `<base model_name>_<experiment name>`, and the experiment name is the
  W&B run-name suffix.
- Data-loading keys (`train_load_data`, `test_load_data`, `variables`, `background_setup`,
  `data_dirs`, `load_into_memory`) **cannot** be overridden (the shared bundle would be invalid;
  the driver refuses). Use `run_dual_experiments.py` for those sweeps.
- The shared load is logged as its own short W&B run (`log_data_loading_summary_run`).

**On Isambard** — [`launch_dual_refit_isambard.sh`](../launch_dual_refit_isambard.sh) wraps the
driver in the container (see [HOW_TO_CONTAINERS_AND_JOBS.md](HOW_TO_CONTAINERS_AND_JOBS.md)):

```bash
sbatch launch_dual_refit_isambard.sh
PARAM_FILE=base.json EXPERIMENTS_FILE=arms.json sbatch launch_dual_refit_isambard.sh
EXP_INDEX=1 sbatch launch_dual_refit_isambard.sh          # one arm
WANDB_MODE=offline sbatch launch_dual_refit_isambard.sh   # then `wandb sync wandb/wandb/offline-run-*`
```

Arms run **sequentially** in one job. For multi-arm jobs with large test sets prefer
`WANDB_MODE=offline` and sync afterwards: online artifact upload from a compute node has stalled
for hours between arms.

## 12. Testing

[`tests/bg_refit_schedule_test.py`](../tests/bg_refit_schedule_test.py) runs `BgSchedule` and
`train_one_epoch` on a tiny dummy dual model (CPU, seconds, no data) and asserts, phase by phase,
exactly which parameter blocks move:

- `joint`: trunk, fp head, bg head all change.
- `frozen`: bg head unchanged; trunk + fp head change.
- `refit bg_trunk`: trunk + bg head change; fp head parameters unchanged (its *predictions*
  still move because the trunk under it moves).
- `refit joint`: everything changes, the bg decoder on the fresh optimizer and no longer present
  in the main optimizer (no double stepping).
- fp-first (`freeze_epoch: 0`) and config validation (`refit_mode`, epoch ordering).

Run inside the project container from the repo root: `python tests/bg_refit_schedule_test.py`.
Extend it whenever a phase's behaviour changes.

## 13. Gotchas and known issues

- **Stale keys in the base template.**
  [`parameter_dual_refit_isambard.json`](../parameter_files/parameter_dual_refit_isambard.json)
  still carries `"refit_mode": "bg_only"` and `"linear_probe": true` from an earlier version
  (the head-only refit and the linear probe were removed on 2026-08-18). `linear_probe` is
  silently ignored, but `"bg_only"` fails `BgSchedule` validation — so the template only works
  through an experiments file whose arms override `refit_mode` (as all recent ones do). Running
  the template directly, or the round-1 `experiments_dual_bg_refit.json` (`frozen_only` does not
  override `refit_mode`; `refit_trunk_frozen` asks for `bg_only`) — which is also the launcher's
  default `EXPERIMENTS_FILE` — will raise `ValueError` at start-up. Fix the template's
  `refit_mode` to `"bg_trunk"` before relying on either.
- **`w = 1` / `w = 0` with a schedule** zeroes an entire phase (section 3). A `w = 1` fp-first
  run trained nothing for 150 epochs and then collapsed to a constant bg predictor.
- **Periodic checkpoints do not include the refit optimizer** (`sched.bg_optimizer`) or the
  schedule state, and the script has no resume-from-checkpoint path (`epoch_so_far` is always
  0). Plan runs to finish within one job.
- **`bg_head.weight_decay` / `lr_scale` are not carried into the refit.** The refit optimizer is
  a plain `AdamW(lr = learning_rate * refit_lr_scale)` with PyTorch's default weight decay
  (0.01); `refit_lr_scale` applies to the bg decoder only — the trunk keeps its original LR.
- **Early stopping monitors the joint test total** in every phase, so it can trigger during the
  frozen phase; the schedule experiments set `epochs.patience = epochs.training`.
- **`test/loss` vs `train/loss` are not the same quantity in non-joint phases** (section 9).
- **Only the fp head has the full metric pipeline**; the bg head is evaluated by its criterion,
  the denormalised MAE, and the time-series plots.
- The `DualModelContext` docstring in `gates/training/training_dataclasses.py` describes the
  total as `fp + w·bg`; the implemented formula is `(1-w)·fp + w·bg`.
- **What has been learnt from the schedule so far** (single-seed, train 2014 / test 2015 unless
  stated; see the W&B group `isambard_bg_refit` notes for the full record): freezing the bg head
  degrades its test loss within an epoch as the trunk drifts under fp-only training; a head-only
  refit cannot recover it (that mode was removed); `joint` refit recovers most of it at
  negligible fp cost; `bg_trunk` gives the best bg of the study at a large fp cost (use the
  pre-refit fp checkpoint); fp-first is consistently worse for bg than co-training even with a
  ×10 head LR, with the gap driven by train→test generalisation rather than optimisation.

## See Also

- [HOW_TO_BOUNDARIES.md](HOW_TO_BOUNDARIES.md) — the background / boundary-condition data and model
- [HOW_TO_PARAMETER_FILE.md](HOW_TO_PARAMETER_FILE.md) — the shared parameter-file fields
- [HOW_TO_WandB.md](HOW_TO_WandB.md) — W&B setup, dual-model sweeps (`best/objective`), shared-data logging
- [HOW_TO_CONTAINERS_AND_JOBS.md](HOW_TO_CONTAINERS_AND_JOBS.md) — Isambard containers and SLURM
- [principles.md](../principles.md) — project principles (this script touches 1, 2, 3, 5, 8)
