"""Run several dual-model experiments from one base parameter file + an overrides file.

Each experiment deep-merges its ``overrides`` onto the base parameters (see
``gates.training.experiment_utils``) and then runs the standard
``train_dual_model.train_and_save_model`` training routine.

Examples
--------
List the experiments defined in an experiments file::

    python run_dual_experiments.py parameter_template_dual.json experiments_dual_example.json --list

Run every experiment sequentially::

    python run_dual_experiments.py parameter_template_dual.json experiments_dual_example.json

Run a single experiment by index (handy for SLURM job arrays, e.g. SLURM_ARRAY_TASK_ID)::

    python run_dual_experiments.py parameter_template_dual.json experiments_dual_example.json --index 2

If ``--index`` is omitted the script falls back to the ``SLURM_ARRAY_TASK_ID`` environment
variable when it is set, so a launch script with ``#SBATCH --array=0-N`` runs one experiment
per array task automatically.
"""

import os
import sys
import argparse
from pathlib import Path

import wandb

import gates
from gates.training.training_helperfuns import load_parameter_file
from gates.training.experiment_utils import load_experiments, build_experiment_params


def _resolve_path(file_name, file_path, cfg):
    if file_path is None:
        file_path = cfg.parameter_files_dir
    return Path(file_path) / file_name


def main():
    parser = argparse.ArgumentParser(description="Run dual-model experiments from a base file + overrides file.")
    parser.add_argument("base_file", help="Base parameter file name (JSON).")
    parser.add_argument("experiments_file", help="Experiments file name (JSON) with an 'experiments' list.")
    parser.add_argument("--file_path", default=None,
                        help="Directory holding both files. Defaults to config.yml's parameter_files_dir.")
    parser.add_argument("--index", type=int, default=None,
                        help="Run only the experiment at this index. Falls back to SLURM_ARRAY_TASK_ID if unset.")
    parser.add_argument("--list", action="store_true", help="List the experiments and exit (no training).")
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

    # Decide which experiments to run
    index = args.index
    if index is None and os.environ.get("SLURM_ARRAY_TASK_ID") is not None:
        index = int(os.environ["SLURM_ARRAY_TASK_ID"])

    if index is not None:
        if index < 0 or index >= len(experiments):
            print(f"Index {index} out of range (0..{len(experiments)-1}). Exiting.")
            sys.exit(1)
        selected = [(index, experiments[index])]
    else:
        selected = list(enumerate(experiments))

    # Resolve where models are saved (same logic as train_dual_model)
    if base_params.get("model_save_dir", None) is None:
        model_saving_dir = cfg.save_models_dir
    else:
        model_saving_dir = Path(base_params["model_save_dir"])

    if base_params.get("use_wandb", False):
        wandb.login()

    # Imported here (not at module top) so --list / arg validation work without the heavy
    # GNN dependencies. train_and_save_model's CLI is guarded by __main__, so this is safe.
    from train_dual_model import train_and_save_model

    for i, experiment in selected:
        name, params = build_experiment_params(base_params, experiment)
        print("\n" + "=" * 70)
        print(f"Running experiment [{i}]: {name or '<unnamed>'}  ->  model_name={params['model_name']}")
        print("=" * 70)
        train_and_save_model(params, model_save_dir=model_saving_dir, wandb_name=name)

    print("\nAll selected experiments finished.")


if __name__ == "__main__":
    main()
