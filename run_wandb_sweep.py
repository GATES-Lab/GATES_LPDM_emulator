import argparse
import ast
import os
from pathlib import Path

import wandb

import gates
from gates.training.training_helperfuns import load_parameter_file
from train_GATES_model import train_and_save_model
from train_GATES_model_multiregion import train_and_save_model_multiregion


def _set_nested(config_dict, dotted_key, value):
    """Set nested dictionary values using dot-notation keys."""
    keys = dotted_key.split(".")
    cursor = config_dict
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[keys[-1]] = value


def _coerce_cli_value(value):
    """Convert a CLI string value into a Python object when possible."""
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


def _apply_cli_overrides(parameters, overrides):
    """Merge CLI key=value overrides into base training parameters."""
    reserved_keys = {"_wandb", "wandb_version", "parameter_file", "file_path", "model_save_dir"}

    for key, value in overrides.items():
        if key in reserved_keys:
            continue
        if key == "multiregion.shared_train_freq":
            if "regions" not in parameters or not isinstance(parameters["regions"], dict):
                raise ValueError("Received multiregion.shared_train_freq but parameter file has no 'regions' dictionary")
            for region_cfg in parameters["regions"].values():
                if "train_load_data" not in region_cfg or not isinstance(region_cfg["train_load_data"], dict):
                    region_cfg["train_load_data"] = {}
                region_cfg["train_load_data"]["freq"] = value
            continue
        if "." in key:
            _set_nested(parameters, key, value)
        else:
            parameters[key] = value


def main():
    parser = argparse.ArgumentParser(
        description="Run one W&B sweep trial with GATES training."
    )
    parser.add_argument("file_name", help="Base parameter JSON file name")
    parser.add_argument(
        "--file_path",
        default=None,
        help="Directory containing the parameter file. Defaults to config.yml parameter_files_dir.",
    )
    parser.add_argument(
        "--model_save_dir",
        default=None,
        help="Optional override for model save directory.",
    )
    parser.add_argument("--project", default=None, help="Optional W&B project override for non-agent runs.")
    parser.add_argument("--entity", default=None, help="Optional W&B entity override for non-agent runs.")

    args, unknown_args = parser.parse_known_args()

    cli_overrides = {}
    for token in unknown_args:
        if "=" not in token:
            raise ValueError(f"Unrecognised argument format: {token}")
        key, value = token.split("=", 1)
        cli_overrides[key] = _coerce_cli_value(value)

    cfg = gates.config.get_config()

    parameter_dir = Path(args.file_path) if args.file_path is not None else Path(cfg.parameter_files_dir)
    parameter_path = parameter_dir / args.file_name

    parameters = load_parameter_file(parameter_path)
    if parameters is None:
        raise FileNotFoundError(f"Could not load parameter file at {parameter_path}")

    _apply_cli_overrides(parameters, cli_overrides)

    # Sweep runs should always log to W&B.
    parameters["use_wandb"] = True

    wandb_kwargs = {}
    is_sweep_agent_run = os.environ.get("WANDB_SWEEP_ID") is not None

    if is_sweep_agent_run:
        # Under wandb agent, project/entity are inferred from the sweep context.
        wandb_kwargs["config"] = parameters
    else:
        params_wandb = parameters.get("wandb", {})
        project = args.project or params_wandb.get("project")
        entity = args.entity or params_wandb.get("entity")
        tags = params_wandb.get("tags", [])

        wandb_kwargs.update({
            "project": project,
            "entity": entity,
            "tags": tags,
            "config": parameters,
        })

    with wandb.init(**wandb_kwargs):
        if wandb.run is not None:
            wandb.config.update(parameters, allow_val_change=True)

        if args.model_save_dir is None:
            model_save_dir = Path(parameters.get("model_save_dir", cfg.save_models_dir))
        else:
            model_save_dir = Path(args.model_save_dir)

        if "regions" in parameters and isinstance(parameters["regions"], dict):
            train_and_save_model_multiregion(parameters, model_save_dir=model_save_dir)
        else:
            train_and_save_model(parameters, model_save_dir=model_save_dir)


if __name__ == "__main__":
    main()
