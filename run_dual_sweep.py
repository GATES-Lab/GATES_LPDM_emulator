"""Sweep the dual-head model over a ``__sweep__`` section written inline in the parameter file.

Two data modes:

* **Shared (default)** — the footprint/met/background data is loaded ONCE from the base parameters
  and the resulting ``DualDataBundle`` is reused for every combination (the same shared-load idea
  as ``run_dual_experiments_shared_data.py``). Fastest, but every combination must then use the
  same data-loading configuration: sweeping any key in
  ``gates.training.experiment_utils.DATA_LOADING_KEYS`` (train_load_data, test_load_data,
  variables, background_setup, data_dirs, load_into_memory) is rejected up front, before any
  expensive loading.
* **Reload (``--reload-data``)** — each combination loads its own data inside
  ``train_and_save_model`` (the same per-run load as ``run_dual_experiments.py``), so data-loading
  keys MAY be swept. The load is repeated for every combination, so prefer this only when the
  sweep actually varies the data. Note: sweeping ``test_load_data`` makes runs incomparable on a
  common held-out set — usually sweep the training data and keep the test data fixed.

Either way, every combination still builds its own class selection, normalisation, scalers,
dataloaders and model. The sweep-expansion logic is shared with ``train_GATES_sweep.py`` (which
instead submits one SLURM job per combination for the single-head GATES model).

Sweep section format — add this anywhere inside the parameter JSON::

    "__sweep__": {
        "learning_rate": [1e-4, 5e-5, 1e-5],
        "model_parameters.num_blocks": [2, 4, 6],
        "loss_functions.bg_loss_weight": [0.0, 1.0]
    }

Dot-notation addresses nested keys at any depth. Everything else in the JSON is used as-is for
every combination. An explicit list of ``{key: value}`` dicts is also accepted (see
``expand_sweep``).

Examples
--------
List the combinations without loading or training anything::

    python run_dual_sweep.py parameter_template_dual_small.json --list

Run every combination sequentially, loading the data once::

    python run_dual_sweep.py parameter_template_dual_small.json

Sweep data-loading keys, re-loading the data for each combination::

    python run_dual_sweep.py my_data_sweep.json --reload-data

Run a single combination for debugging::

    python run_dual_sweep.py parameter_template_dual_small.json --index 2

Note on SLURM job arrays: in shared mode this script is not meant for arrays (an ``--index`` array
would re-load the data per task and defeat the purpose). With ``--reload-data`` the situation
flips: each combination loads its own data anyway, so an array over ``--index`` is a sensible way
to run a data-varying sweep in parallel.
"""

import sys
import argparse
from pathlib import Path

import wandb

import gates
from gates.training.training_helperfuns import load_parameter_file
from gates.training.experiment_utils import (
    build_experiment_params, sweep_to_experiments, data_loading_clashes,
)


def _resolve_path(file_name, file_path, cfg):
    if file_path is None:
        file_path = cfg.parameter_files_dir
    return Path(file_path) / file_name


def _check_no_data_overrides(experiments):
    """Fail loudly if any combination sweeps a data-loading key (see DATA_LOADING_KEYS)."""
    offenders = []
    for i, exp in enumerate(experiments):
        clashes = data_loading_clashes(exp.get("overrides", {}))
        if clashes:
            offenders.append((i, exp.get("name", "<unnamed>"), sorted(clashes)))
    if offenders:
        print("ERROR: these sweep combinations override data-loading keys, which is incompatible "
              "with loading the data once:")
        for i, name, clashes in offenders:
            print(f"  [{i}] {name}: {clashes}")
        print("Either re-run with --reload-data (each combination re-loads its own data), or "
              "submit independent jobs with train_GATES_sweep.py.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Sweep the dual-head model over a '__sweep__' section. By default the data is "
                    "loaded once and shared; with --reload-data each combination loads its own.")
    parser.add_argument("base_file", help="Base parameter file name (JSON) containing a '__sweep__' section.")
    parser.add_argument("--file_path", default=None,
                        help="Directory holding the file. Defaults to config.yml's parameter_files_dir.")
    parser.add_argument("--reload-data", action="store_true",
                        help="Load the data per combination instead of once up front. Allows "
                             "sweeping data-loading keys (train_load_data, variables, ...) at the "
                             "cost of repeating the load for every combination.")
    parser.add_argument("--index", type=int, default=None,
                        help="Run only the combination at this index.")
    parser.add_argument("--list", action="store_true",
                        help="List the combinations and exit (no loading/training).")
    args = parser.parse_args()

    cfg = gates.config.get_config()

    base_params = load_parameter_file(_resolve_path(args.base_file, args.file_path, cfg))
    if base_params is None:
        print("Error loading base parameter file. Exiting.")
        sys.exit(1)

    # The sweep spec lives inline in the base file; strip it so it never leaks into a run's config.
    sweep_spec = base_params.pop("__sweep__", None)
    if not sweep_spec:
        print("No '__sweep__' section found in the parameter file. Nothing to do.")
        sys.exit(1)

    experiments = sweep_to_experiments(sweep_spec)

    if args.list:
        print(f"{len(experiments)} combination(s) expanded from '__sweep__' in {args.base_file}:")
        for i, exp in enumerate(experiments):
            print(f"  [{i}] {exp['name']}  overrides={exp['sweep_combination']}")
        return

    # In shared mode, guard before doing any expensive loading: the shared data would be invalid
    # otherwise. In reload mode each combination loads from its own params, so data sweeps are fine.
    if not args.reload_data:
        _check_no_data_overrides(experiments)

    # Decide which combinations to run.
    if args.index is not None:
        if args.index < 0 or args.index >= len(experiments):
            print(f"Index {args.index} out of range (0..{len(experiments)-1}). Exiting.")
            sys.exit(1)
        selected = [(args.index, experiments[args.index])]
    else:
        selected = list(enumerate(experiments))

    # Resolve where models are saved (same logic as train_dual_model).
    if base_params.get("model_save_dir", None) is None:
        model_saving_dir = cfg.save_models_dir
    else:
        model_saving_dir = Path(base_params["model_save_dir"])

    if base_params.get("use_wandb", False):
        wandb.login()

    # Imported here (not at module top) so --list / arg validation work without the heavy GNN
    # dependencies. train_and_save_model's CLI is guarded by __main__, so this is safe.
    from train_dual_model import train_and_save_model, load_dual_data

    if args.reload_data:
        # Each combination loads its own data inside train_and_save_model (data_bundle=None), so
        # swept data-loading keys take effect per run.
        data_bundle = None
        print("=" * 70)
        print("Reload mode: each combination loads its own data.")
        print("=" * 70)
    else:
        # --- Load the data ONCE, from the base parameters, and reuse it for every combination. ---
        print("=" * 70)
        print("Loading shared data once from base file:", args.base_file)
        print("=" * 70)
        data_bundle = load_dual_data(base_params, verbose=base_params.get("verbose", True))

    for i, experiment in selected:
        name, params = build_experiment_params(base_params, experiment)
        print("\n" + "=" * 70)
        print(f"Running combination [{i}]: {name}  ->  model_name={params['model_name']}")
        print("=" * 70)
        train_and_save_model(params, model_save_dir=model_saving_dir, wandb_name=name,
                             data_bundle=data_bundle)

    print("\nAll selected combinations finished.")


if __name__ == "__main__":
    main()
