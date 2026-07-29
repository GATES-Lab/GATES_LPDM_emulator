"""Run several dual-model experiments while loading the (expensive) data only ONCE.

This is a sibling of ``run_dual_experiments.py``. The difference: instead of re-loading the
footprint/met/background data for every experiment, the raw data is loaded a single time from the
base parameter file (via ``train_dual_model.load_dual_data``) and the resulting
``DualDataBundle`` is reused across all experiments. Each experiment still builds its own class
selection, normalisation, scalers, dataloaders and model, so only the raw load is shared.

Because the data is loaded once, every experiment MUST use the same data-loading configuration.
Overriding any of the following keys would silently invalidate the shared data, so it is rejected:

    train_load_data, test_load_data, variables, background_setup, data_dirs, load_into_memory

If you need to sweep those, use ``run_dual_experiments.py`` instead (it re-loads per experiment).

Examples
--------
List the experiments::

    python run_dual_experiments_shared_data.py parameter_template_dual_small.json \
        experiments_dual_bg_sweep.json --list

Run every experiment sequentially, loading the data once::

    python run_dual_experiments_shared_data.py parameter_template_dual_small.json \
        experiments_dual_bg_sweep.json

Note: this script loads the data once and runs the selected experiments in a single process, so it
is not meant for SLURM job arrays (a ``--index`` array would re-load the data per task and defeat
the purpose). It does accept ``--index`` to run a single experiment for debugging.
"""

import sys
import argparse
from pathlib import Path

import wandb

import gates
from gates.training.training_helperfuns import load_parameter_file
from gates.training.experiment_utils import (
    load_experiments, build_experiment_params, expand_dotted_keys,
)

# Overriding any of these per experiment would change what data should be loaded, so the shared
# bundle would no longer be valid for that experiment.
DATA_LOADING_KEYS = {
    "train_load_data", "test_load_data", "variables", "background_setup",
    "data_dirs", "load_into_memory",
}


def _resolve_path(file_name, file_path, cfg):
    if file_path is None:
        file_path = cfg.parameter_files_dir
    return Path(file_path) / file_name


def _check_no_data_overrides(experiments):
    """Fail loudly if any experiment overrides a data-loading key (see DATA_LOADING_KEYS)."""
    offenders = []
    for i, exp in enumerate(experiments):
        touched = set(expand_dotted_keys(exp.get("overrides", {})).keys())
        clashes = touched & DATA_LOADING_KEYS
        if clashes:
            offenders.append((i, exp.get("name", "<unnamed>"), sorted(clashes)))
    if offenders:
        print("ERROR: these experiments override data-loading keys, which is incompatible with "
              "loading the data once:")
        for i, name, clashes in offenders:
            print(f"  [{i}] {name}: {clashes}")
        print("Use run_dual_experiments.py (re-loads per experiment) for these sweeps instead.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run dual-model experiments, loading the data only once and reusing it.")
    parser.add_argument("base_file", help="Base parameter file name (JSON).")
    parser.add_argument("experiments_file", help="Experiments file name (JSON) with an 'experiments' list.")
    parser.add_argument("--file_path", default=None,
                        help="Directory holding both files. Defaults to config.yml's parameter_files_dir.")
    parser.add_argument("--index", type=int, default=None,
                        help="Run only the experiment at this index (still loads the shared data once).")
    parser.add_argument("--list", action="store_true", help="List the experiments and exit (no loading/training).")
    args = parser.parse_args()

    cfg = gates.config.get_config()

    base_params = load_parameter_file(_resolve_path(args.base_file, args.file_path, cfg))
    if base_params is None:
        print("Error loading base parameter file. Exiting.")
        sys.exit(1)

    experiments_obj = load_parameter_file(_resolve_path(args.experiments_file, args.file_path, cfg))
    if experiments_obj is None:
        print("Error loading experiments file. Exiting.")
        sys.exit(1)

    experiments = load_experiments(experiments_obj)

    if args.list:
        print(f"{len(experiments)} experiment(s) defined in {args.experiments_file}:")
        for i, exp in enumerate(experiments):
            print(f"  [{i}] {exp.get('name', '<unnamed>')}  overrides={exp.get('overrides', {})}")
        return

    # Guard before doing any expensive loading: the shared data would be invalid otherwise.
    _check_no_data_overrides(experiments)

    # Decide which experiments to run.
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
    from train_dual_model import train_and_save_model, load_dual_data, log_data_loading_summary_run

    # --- Load the data ONCE, from the base parameters, and reuse it for every experiment. ---
    print("=" * 70)
    print("Loading shared data once from base file:", args.base_file)
    print("=" * 70)
    data_bundle, load_summary = load_dual_data(
        base_params, verbose=base_params.get("verbose", True), return_summary=True)

    # Record the shared load in its own W&B run (finished immediately), before any training run.
    if base_params.get("use_wandb", False):
        log_data_loading_summary_run(base_params, load_summary)

    for i, experiment in selected:
        name, params = build_experiment_params(base_params, experiment)
        print("\n" + "=" * 70)
        print(f"Running experiment [{i}]: {name or '<unnamed>'}  ->  model_name={params['model_name']}")
        print("=" * 70)
        train_and_save_model(params, model_save_dir=model_saving_dir, wandb_name=name,
                             data_bundle=data_bundle)

    print("\nAll selected experiments finished.")


if __name__ == "__main__":
    main()
