# How to run parameter sweeps

There are two sweep drivers. Both read the **same** `"__sweep__"` section written inline in a
parameter JSON, so you describe the combinations once and pick the driver by how you want the data
loaded. The expansion/naming logic is shared in
[`gates/training/experiment_utils.py`](../gates/training/experiment_utils.py)
(`expand_sweep`, `sweep_to_experiments`, `make_suffix`).

## The `__sweep__` section

Add a `"__sweep__"` key anywhere in the parameter file. Two forms are accepted:

```jsonc
// Cartesian product: every combination of the listed values (this example → 3 × 2 = 6 runs)
"__sweep__": {
    "loss_functions.bg_loss_weight": [0.0, 0.5, 1.0],
    "learning_rate": [5e-05, 1e-04]
}
```

```jsonc
// Explicit list: each dict is used verbatim (2 runs)
"__sweep__": [
    {"learning_rate": 1e-4, "model_parameters.num_blocks": 2},
    {"learning_rate": 1e-5, "model_parameters.num_blocks": 6}
]
```

Dot-notation (`"model_parameters.num_blocks"`) addresses nested keys at any depth. Everything else
in the file is used as-is for every combination. The `"__sweep__"` key itself is stripped before a
run's config is recorded, so it never leaks into W&B or the saved settings.

## Driver 1 — dual model, one process (recommended for the dual head)

[`run_dual_sweep.py`](../run_dual_sweep.py) expands `"__sweep__"` and trains every combination
**in one process** with `train_dual_model.train_and_save_model`. It has two data modes:

* **Shared (default)** — the footprint/met/background data is loaded a single time and the
  `DualDataBundle` is reused across runs (the same shared-load idea as
  `run_dual_experiments_shared_data.py`, but driven by `"__sweep__"` instead of a separate
  experiments file). Fastest, but no combination may sweep a data-loading key.
* **Reload (`--reload-data`)** — each combination loads its own data from its merged parameters
  (the same per-run load as `run_dual_experiments.py`), so data-loading keys **may** be swept.
  The load is repeated per combination, so use this only when the sweep actually varies the data.
  Sweeping `test_load_data` makes runs incomparable on a common held-out set — usually sweep the
  training data and keep the test data fixed.

```bash
# List the combinations without loading or training anything
python run_dual_sweep.py parameter_template_dual_small.json --list

# Run the whole sweep, loading the data once (shared mode)
python run_dual_sweep.py parameter_template_dual_small.json

# Sweep data-loading keys (e.g. train_load_data.freq), re-loading per combination
python run_dual_sweep.py my_data_sweep.json --reload-data

# Run a single combination for debugging
python run_dual_sweep.py parameter_template_dual_small.json --index 2
```

On the cluster, submit the single job with
[`launch_dual_sweep_shared_data.sh`](../launch_dual_sweep_shared_data.sh) (Isambard apptainer
container; edit `PARAM_FILE` at the top — and add `--reload-data` to the `python` line for a
data-varying sweep):

```bash
sbatch launch_dual_sweep_shared_data.sh
# online W&B (after `wandb login` on the login node):
WANDB_MODE=online sbatch launch_dual_sweep_shared_data.sh
```

**Constraint (shared mode only):** because the data is loaded once, no combination may sweep a
data-loading key (`train_load_data`, `test_load_data`, `variables`, `background_setup`,
`data_dirs`, `load_into_memory` — see `DATA_LOADING_KEYS` in `experiment_utils.py`). The driver
checks this up front and exits with a clear message before any expensive loading if a sweep
violates it. To sweep those keys, pass `--reload-data` (or use Driver 2).

## Driver 2 — single GATES model, one SLURM job per combination

[`tests/train_GATES_sweep.py`](../tests/train_GATES_sweep.py) writes one parameter file per
combination into `<param_dir>/sweep_configs/` and submits a **separate** SLURM job for each (via
[`tests/launch_train_sweep.sh`](../tests/launch_train_sweep.sh)), each running
`train_GATES_model.py`. Every job re-loads its own data, so this is the option when you need to
sweep data-loading keys or run combinations in parallel across nodes.

```bash
python tests/train_GATES_sweep.py my_config.json --dry-run   # print jobs without submitting
python tests/train_GATES_sweep.py my_config.json             # submit one job per combination
```

## Which driver do I want?

| | Driver 1 (`run_dual_sweep.py`) | Driver 1 + `--reload-data` | Driver 2 (`tests/train_GATES_sweep.py`) |
|---|---|---|---|
| Model | dual head (`train_dual_model`) | dual head (`train_dual_model`) | single GATES (`train_GATES_model`) |
| Data load | once, shared across all runs | once per combination | once per job |
| SLURM jobs | one (combinations run sequentially) | one (combinations run sequentially) | one per combination (parallel) |
| Can sweep data-loading keys? | no (rejected up front) | yes | yes |
