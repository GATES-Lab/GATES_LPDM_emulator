"""Drive a W&B hyperparameter sweep for the dual-head model.

A sweep is defined by a YAML file (e.g. ``parameter_files/sweep_dual_wandb.yaml``) whose
parameter names are dotted paths into the base parameter JSON. Each trial deep-merges the
sweep-chosen values onto the base parameters and calls ``train_dual_model.train_and_save_model``.
Like ``run_dual_experiments_shared_data.py``, the raw data is loaded ONCE per agent process and
reused across trials, so the sweep must not touch data-loading keys (this is checked).

Workflow
--------
1. Create the sweep (a lightweight API call — no data loading, no GPU needed)::

       python run_dual_sweep_wandb.py parameter_template_dual.json --create sweep_dual_wandb.yaml

   This prints a sweep id like ``entity/project/abc123``.

2. Run one or more agents on GPU nodes (each agent pulls trials from the sweep until
   ``--count`` trials are done or the job ends). Submit several jobs for parallel trials::

       sbatch launch_dual_wandb_sweep.sh entity/project/abc123 10

The sweep optimises ``best/objective`` (see ``run_full_training`` in train_dual_model.py):
the sum of the best-so-far per-head test losses, with the bg term weighted by the constant
``sweep.objective_bg_weight`` from the base parameter file (default 1.0). Keep that constant
fixed across the sweep so trials are comparable.
"""

import argparse
import sys
import traceback
from pathlib import Path

import wandb
import yaml

import gates
from gates.training.training_helperfuns import load_parameter_file
from gates.training.experiment_utils import expand_dotted_keys, deep_update
from run_dual_experiments_shared_data import DATA_LOADING_KEYS


def _resolve_path(file_name, file_path, cfg):
    if file_path is None:
        file_path = cfg.parameter_files_dir
    return Path(file_path) / file_name


def check_sweep_space(sweep_cfg):
    """Reject sweep parameters that would invalidate the shared data bundle."""
    offenders = sorted(
        k for k in sweep_cfg.get("parameters", {})
        if k.split(".")[0] in DATA_LOADING_KEYS
    )
    if offenders:
        print("ERROR: the sweep config varies data-loading keys, which is incompatible with "
              "loading the data once per agent:")
        for k in offenders:
            print(f"  {k}")
        print("Remove them from the sweep YAML (or sweep them manually with run_dual_experiments.py).")
        sys.exit(1)


def create_sweep(sweep_yaml_path, base_params):
    with open(sweep_yaml_path) as f:
        sweep_cfg = yaml.safe_load(f)
    check_sweep_space(sweep_cfg)

    entity = base_params.get("wandb", {}).get("entity")
    project = base_params.get("wandb", {}).get("project")
    if entity is None or project is None:
        print("ERROR: the base parameter file must set wandb.entity and wandb.project.")
        sys.exit(1)

    sweep_id = wandb.sweep(sweep_cfg, entity=entity, project=project)
    full_id = f"{entity}/{project}/{sweep_id}"
    print("\nCreated sweep:", full_id)
    print(f"View it at: https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}")
    print(f"Run agents with: sbatch launch_dual_wandb_sweep.sh {full_id} <count>")
    return full_id


def run_agent(sweep_id, base_params, count):
    # Heavy imports deferred so --create stays a lightweight API call.
    import torch
    from train_dual_model import train_and_save_model, load_dual_data

    cfg = gates.config.get_config()
    if base_params.get("model_save_dir", None) is None:
        model_saving_dir = cfg.save_models_dir
    else:
        model_saving_dir = Path(base_params["model_save_dir"])

    base_params["use_wandb"] = True

    print("=" * 70)
    print("Loading shared data once from the base parameter file (reused for every trial)")
    print("=" * 70)
    data_bundle = load_dual_data(base_params, verbose=base_params.get("verbose", True))

    def run_trial():
        run = wandb.init()
        try:
            # In a sweep, run.config holds exactly the sweep-chosen (dotted) parameters.
            overrides = expand_dotted_keys(dict(run.config))
            params = deep_update(base_params, overrides)
            params["model_name"] = f"{params['model_name']}_sw_{run.id}"
            print(f"\nSweep trial {run.id} overrides: {dict(run.config)}")
            # train_and_save_model reuses the active run and calls wandb.finish() at the end.
            train_and_save_model(params, model_save_dir=model_saving_dir, data_bundle=data_bundle)
        except Exception:
            # Keep the agent alive for the remaining trials (e.g. an OOM on a large config).
            traceback.print_exc()
            wandb.finish(exit_code=1)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # sweep_id may be "entity/project/id" (self-contained) or a bare id (needs kwargs).
    agent_kwargs = {}
    if "/" not in sweep_id:
        agent_kwargs["entity"] = base_params.get("wandb", {}).get("entity")
        agent_kwargs["project"] = base_params.get("wandb", {}).get("project")
    wandb.agent(sweep_id, function=run_trial, count=count, **agent_kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="Create or run agents for a W&B sweep over the dual-head model.")
    parser.add_argument("base_file", help="Base parameter file name (JSON), e.g. parameter_template_dual.json")
    parser.add_argument("--file_path", default=None,
                        help="Directory holding the parameter/sweep files. Defaults to config.yml's parameter_files_dir.")
    parser.add_argument("--create", metavar="SWEEP_YAML", default=None,
                        help="Create the sweep from this YAML file and print its id (no training).")
    parser.add_argument("--sweep_id", default=None,
                        help="Run an agent for this sweep id (accepts entity/project/id or bare id).")
    parser.add_argument("--count", type=int, default=None,
                        help="Max trials for this agent (default: keep going until the sweep is done).")
    args = parser.parse_args()

    if (args.create is None) == (args.sweep_id is None):
        parser.error("pass exactly one of --create or --sweep_id")

    cfg = gates.config.get_config()
    base_params = load_parameter_file(_resolve_path(args.base_file, args.file_path, cfg))
    if base_params is None:
        print("Error loading base parameter file. Exiting.")
        sys.exit(1)

    wandb.login()

    if args.create is not None:
        create_sweep(_resolve_path(args.create, args.file_path, cfg), base_params)
    else:
        run_agent(args.sweep_id, base_params, args.count)


if __name__ == "__main__":
    main()
