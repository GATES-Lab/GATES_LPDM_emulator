"""
predict_boundary.py

Load a trained boundary condition model and run monthly predictions for a given test year.
Predictions are saved as per-month NetCDF files with predicted and true boundary conditions.

Usage
-----
    python predict_boundary.py \\
        --test_year 2019 \\
        --reference_model my_model_20240115_143022 \\
        [--month 06] \\
        [--region SAHARA] \\
        [--model_path /path/to/models/] \\
        [--checkpoint best|<epoch_int>] \\
        [--save_path /path/to/outputs/] \\
        [--model_save_name custom_name] \\
        [--dry_run]
"""

import sys
import re
import copy
import json
import pickle
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import xarray as xr
import pandas as pd

sys.path.insert(0, "/user/work/yl18410/new_graphnet")
sys.path.insert(1, "/user/work/yl18410/new_graphnet/graphnet_LPDM_emulator")

import gates
import gates.config
import gates.data.datasets as gates_datasets
from gates.training.training import load_GATES_data, load_GATES_data_v2, make_cluster
from gates.training.training_helperfuns import load_parameter_file
from gates.training.training_dataclasses import PathContext

from model.forecast import GraphSatelliteForecasterConvClassifier, GraphSatelliteForecasterClassifier

# Import boundary condition utilities from training script
from train_boundary_model import (
    baseline_mol_correction_xr,
    normalize_boundary_data,
    parse_years,
)


# ---------------------------------------------------------------------------
# Module-level helpers (reused from predict_GATES_model.py)
# ---------------------------------------------------------------------------

def find_model_dir(model_path: str, reference_model: str) -> Path:
    """Resolve the reference model directory."""
    model_path = Path(model_path)
    if re.match(r".*_\d{8}_\d{6}$", reference_model):
        model_dir = model_path / reference_model
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        return model_dir

    candidates = sorted(d for d in model_path.glob(f"{reference_model}_*/") if d.is_dir())
    if not candidates:
        raise FileNotFoundError(
            f"No model directory found for '{reference_model}' in {model_path}"
        )
    return candidates[-1]


def load_model_artifacts(model_dir: Path, model_name: str):
    """Load scalers, grid, norm_vals, and training settings from training_outputs."""
    outputs_dir = model_dir / "training_outputs"

    with open(outputs_dir / f"scalers_{model_name}.pickle", "rb") as f:
        scalers = pickle.load(f)

    with open(outputs_dir / f"grid_{model_name}.pickle", "rb") as f:
        grid = pickle.load(f)

    # Load normalisation values for boundary condition outputs
    norm_vals_path = outputs_dir / f"norm_vals_{model_name}.json"
    if norm_vals_path.exists():
        with open(norm_vals_path, "r") as f:
            norm_vals_raw = json.load(f)
        # Convert lists back to tuples
        norm_vals = {k: tuple(v) for k, v in norm_vals_raw.items()}
    else:
        print(f"Warning: norm_vals not found at {norm_vals_path}. Normalisation will not be applied at inference.")
        norm_vals = None

    training_params = load_parameter_file(outputs_dir / f"training_settings_{model_name}.json")
    if training_params is None:
        raise RuntimeError(f"Failed to load training settings from {outputs_dir}")

    return scalers, grid, norm_vals, training_params


def determine_save_name(model_save_name_arg, model_dir: Path) -> str:
    if model_save_name_arg:
        return model_save_name_arg
    return re.sub(r"_\d{8}_\d{6}$", "", model_dir.name)


def normalize_month(month) -> str:
    try:
        month_int = int(str(month).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid month value: {month!r}") from exc
    if not 1 <= month_int <= 12:
        raise ValueError(f"Month must be between 1 and 12, got {month!r}")
    return f"{month_int:02d}"


def load_checkpoint(model, model_dir, model_name, checkpoint, device):
    """Load model weights from a checkpoint file."""
    if checkpoint == "best":
        path = model_dir / f"{model_name}_best.pt"
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"Loaded best checkpoint from {path}")
    else:
        epoch = int(checkpoint)
        path = model_dir / f"{model_name}_{epoch}.pt"
        state = torch.load(path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        print(f"Loaded epoch {epoch} checkpoint from {path}")


@torch.no_grad()
def run_inference(model, loader, device):
    """Run the model over all batches; return concatenated predictions."""
    model.eval()
    preds = []
    for batch in loader:
        preds.append(model(batch[0].to(device)).cpu())
    return torch.cat(preds, dim=0).numpy()


# ---------------------------------------------------------------------------
# Predictor class
# ---------------------------------------------------------------------------

class BoundaryPredictor:
    """
    A loaded boundary condition model ready to generate predictions.

    Initialise once via ``BoundaryPredictor.from_args(args)``;
    then call ``predict_month(year, month)`` or ``predict_year(year)``.
    """

    def __init__(
        self,
        model,
        scalers,
        norm_vals,
        model_name,
        model_save_name,
        prediction_folder_name,
        save_path,
        device,
        data_params,
        input_variables,
        datapath_args,
        test_batch_size,
        checkpoint,
        parameter_file,
        training_params,
        domain,
        aux_dim,
        height_indices,
        output_format,
        dry_run=False,
    ):
        self.model = model
        self.scalers = scalers
        self.norm_vals = norm_vals
        self.model_name = model_name
        self.model_save_name = model_save_name
        self.prediction_folder_name = prediction_folder_name
        self.save_path = Path(save_path)
        self.device = device
        self.data_params = data_params
        self.input_variables = input_variables
        self.datapath_args = datapath_args
        self.test_batch_size = test_batch_size
        self.checkpoint = checkpoint
        self.parameter_file = parameter_file
        self.training_params = training_params
        self.domain = domain
        self.aux_dim = aux_dim
        self.height_indices = height_indices
        self.output_format = output_format
        self.dry_run = dry_run

    @classmethod
    def from_args(cls, args):
        """Build a BoundaryPredictor from parsed CLI arguments."""
        cfg = gates.config.get_config()

        model_path = args.model_path or cfg.save_models_dir
        model_dir = find_model_dir(model_path, args.reference_model)
        model_name = model_dir.name
        print(f"Reference model: {model_dir}")

        print("Loading scalers, grid, norm_vals and training settings...")
        scalers, grid, norm_vals, training_params = load_model_artifacts(model_dir, model_name)

        model_save_name = determine_save_name(args.model_save_name, model_dir)
        save_path = Path(args.save_path) if args.save_path else model_dir
        prediction_folder_name = "predictions" + (f"_{args.region}" if args.region else "")
        print(f"Predictions root: {save_path / prediction_folder_name}/")

        # Data config from training settings
        paths_ctx = PathContext(
            model_save_dir=model_path, model_name=model_name, model_path=model_dir
        )
        datapath_args = paths_ctx.resolve_datapath_args(training_params)
        data_params = copy.deepcopy(training_params["train_load_data"])
        data_params.update(training_params.get("test_load_data", {}))
        if args.region:
            data_params["region"] = args.region
            data_params.pop("domain", None)
        input_variables = training_params["variables"]
        data_params.pop("years", None)
        data_params.pop("months", None)

        # Domain for CAMS files
        region = data_params.get("region", "SAHARA")
        domain = cfg.domains[region]["domain_name"]

        # Height indices and output format from training params
        height_indices = [4, 5, 6, 7] if training_params.get("auxiliary") == "multiple" else [4]
        output_format = training_params.get("output_format", "corrected")

        # aux_dim — number of auxiliary CAMS features appended to inputs during training
        aux_dim = training_params.get("aux_dim", 0)

        # Feature dim — must include aux_dim since the model was trained with
        # met features + aux features concatenated along the variable_name dimension
        feature_dim = training_params.get("num_features", None)

        if feature_dim is not None:
            feature_dim = feature_dim + aux_dim
            print(f"Using feature_dim={feature_dim} (num_features={feature_dim - aux_dim} + aux_dim={aux_dim})")
        else:
            print("num_features not found in training settings, probing data to determine feature_dim...")
            for probe_month in [f"{m:02d}" for m in range(1, 13)]:
                try:
                    probe_params = {
                        **data_params,
                        "year": str(args.test_year),
                        "month": probe_month,
                        "freq": 50,
                    }
                    _, probe_inputs = load_GATES_data(
                        probe_params, input_variables, datapath_args, verbose=False
                    )
                    if probe_inputs is not None and probe_inputs.sizes.get("fp_time", 0) > 0:
                        feature_dim = probe_inputs.sizes["variable_name"] + aux_dim
                        print(f"  feature_dim = {feature_dim} (from month {probe_month})")
                        break
                except Exception:
                    continue

            if feature_dim is None:
                raise RuntimeError(
                    "Could not determine feature_dim from data. "
                    "Ensure num_features is saved in training_settings.json."
                )

        # Build model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        num_classes = training_params.get("num_classes", 1)
        size = data_params.get("size", 10)

        if training_params.get("network_decoder") == "conv":
            print("Using conv network")
            model = GraphSatelliteForecasterConvClassifier(
                grid, whole_world=False, feature_dim=feature_dim,
                aux_dim=aux_dim, num_classes=num_classes,
                input_height=size, input_width=size,
                **training_params["model_parameters"]
            )
        else:
            print("Using classifier network")
            model = GraphSatelliteForecasterClassifier(
                grid, whole_world=False, feature_dim=feature_dim,
                aux_dim=aux_dim, num_classes=num_classes,
                **training_params["model_parameters"]
            )

        model = model.to(device)
        load_checkpoint(model, model_dir, model_name, args.checkpoint, device)

        training_settings_path = str(
            model_dir / "training_outputs" / f"training_settings_{model_name}.json"
        )

        return cls(
            model=model,
            scalers=scalers,
            norm_vals=norm_vals,
            model_name=model_name,
            model_save_name=model_save_name,
            prediction_folder_name=prediction_folder_name,
            save_path=save_path,
            device=device,
            data_params=data_params,
            input_variables=input_variables,
            datapath_args=datapath_args,
            test_batch_size=training_params.get("dataloader", {}).get("test_batch_size", 5),
            checkpoint=args.checkpoint,
            parameter_file=training_settings_path,
            training_params=training_params,
            domain=domain,
            aux_dim=aux_dim,
            height_indices=height_indices,
            output_format=output_format,
            dry_run=args.dry_run,
        )

    # ------------------------------------------------------------------
    # Core prediction methods
    # ------------------------------------------------------------------

    def predict_month(self, test_year: int, month_str: str, verbose: bool = True) -> bool:
        """Load one month of data, run inference, save predictions as NetCDF."""
        monthly_params = {
            **self.data_params,
            "year": str(test_year),
            "month": month_str,
            "freq": 60 if self.dry_run else 1,
        }

        try:
            data, inputs = load_GATES_data(
                monthly_params, self.input_variables, self.datapath_args, verbose=verbose
            )
        except Exception as exc:
            print(f"  Skipping {month_str}: data load failed — {exc}")
            return False

        if inputs is None or inputs.sizes.get("fp_time", 0) == 0:
            print(f"  Skipping {month_str}: no timesteps loaded.")
            return False

        print(f"  Loaded {inputs.sizes['fp_time']} timesteps")
        inputs = inputs.compute()

        # Compute true boundary condition outputs
        print("  Computing boundary conditions...")
        months_list = [month_str]
        years_list = [test_year]

        try:
            _, true_outputs, auxiliary_cams, _, _ = self._load_boundary_data(
                data, months_list, years_list
            )
        except Exception as exc:
            print(f"  Warning: could not compute boundary conditions — {exc}")
            true_outputs = None
            auxiliary_cams = None

        # Scale inputs
        scaled_inputs = self.scalers["inputs_scaler"].transform(inputs)
        scaled_inputs = scaled_inputs.compute() if hasattr(scaled_inputs, "compute") else scaled_inputs

        # Append auxiliary CAMS to inputs if needed
        if self.aux_dim > 0 and auxiliary_cams is not None:
            from gates.training.training import concat_auxiliary_to_inputs
            scaled_inputs = concat_auxiliary_to_inputs(scaled_inputs, auxiliary_cams)
            scaled_inputs = scaled_inputs.compute() if hasattr(scaled_inputs, "compute") else scaled_inputs

        # Trim and build dataloader
        n = scaled_inputs.sizes["fp_time"]
        remainder = n % self.test_batch_size
        if remainder != 0:
            scaled_inputs = scaled_inputs.isel(fp_time=slice(None, n - remainder))
            if true_outputs is not None:
                true_outputs = true_outputs.isel(time=slice(None, n - remainder))

        loader, _ = gates_datasets.make_boundary_dataloader(
            scaled_inputs,
            true_outputs if true_outputs is not None else self._make_dummy_outputs(scaled_inputs),
            batch_size=self.test_batch_size,
            randomize=False,
            dataloader_params={
                "num_workers": 0,
                "persistent_workers": False,
                "prefetch_factor": None,
            },
            flatten=True,
        )

        print("  Running inference...")
        preds_norm = run_inference(self.model, loader, self.device)  # (n, num_classes)

        # Denormalise predictions
        if self.norm_vals is not None:
            outputs_mean, outputs_std = self.norm_vals["outputs"]
            preds_denorm = (preds_norm * outputs_std) + outputs_mean
        else:
            preds_denorm = preds_norm

        # Build output dataset
        times = data.fp_xr.time.values[:preds_norm.shape[0]]
        ds = xr.Dataset(
            {
                "bc_pred_normalised": (["time", "num_classes"], preds_norm),
                "bc_pred": (["time", "num_classes"], preds_denorm),
            },
            coords={"time": times},
            attrs={
                "creation_date": str(datetime.now()),
                "model_save_name": self.model_save_name,
                "reference_model": self.model_name,
                "test_year": str(test_year),
                "month": month_str,
                "output_format": self.output_format,
            },
        )

        # Add true outputs if available
        if true_outputs is not None:
            true_vals = true_outputs.values[:preds_norm.shape[0]]
            ds["bc_true_normalised"] = (["time", "num_classes"], true_vals)
            if self.norm_vals is not None:
                outputs_mean, outputs_std = self.norm_vals["outputs"]
                ds["bc_true"] = (["time", "num_classes"], (true_vals * outputs_std) + outputs_mean)

        out_dir = self.save_path / self.model_save_name / self.prediction_folder_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"predictions_{test_year}_{month_str}.nc"
        ds.to_netcdf(out_path)
        print(f"  Saved → {out_path}")
        return True

    def _load_boundary_data(self, data, months, years):
        """Load and normalise boundary condition outputs using training norm_vals."""
        baseline_list, outputs, auxiliary_cams, corrections = baseline_mol_correction_xr(
            data, months, years,
            output_format=self.output_format,
            height_indices=self.height_indices,
            domain=self.domain,
        )
        outputs_norm, _ = normalize_boundary_data(outputs, self.norm_vals.get("outputs") if self.norm_vals else None)
        auxiliary_norm, _ = normalize_boundary_data(auxiliary_cams, self.norm_vals.get("auxiliary") if self.norm_vals else None)
        return baseline_list, outputs_norm, auxiliary_norm, corrections, None

    def _make_dummy_outputs(self, scaled_inputs):
        """Make a dummy outputs xarray when true outputs are unavailable."""
        n = scaled_inputs.sizes["fp_time"]
        times = scaled_inputs.fp_time.values
        return xr.DataArray(
            np.zeros((n, 1), dtype=np.float32),
            dims=["time", "num_classes"],
            coords={"time": times, "num_classes": [0]},
            name="boundary_outputs"
        )

    def predict_year(self, test_year: int, months: list = None, verbose: bool = True):
        """Run predictions for every month of test_year."""
        if months is None:
            months = [f"{m:02d}" for m in range(1, 13)]

        if self.dry_run:
            months = months[:1]
            print(f"Dry run: processing only month {months[0]}.")

        out_dir = self.save_path / self.model_save_name / self.prediction_folder_name
        out_dir.mkdir(parents=True, exist_ok=True)

        run_record = {
            "created_at": datetime.now().isoformat(),
            "test_year": test_year,
            "reference_model": self.model_name,
            "model_save_name": self.model_save_name,
            "checkpoint": self.checkpoint,
            "parameter_file": self.parameter_file,
            "save_path": str(out_dir),
            "region": self.data_params.get("region"),
            "domain": self.domain,
            "output_format": self.output_format,
            "dry_run": self.dry_run,
        }
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        record_path = out_dir / f"run_record_{timestamp}.json"
        with open(record_path, "w") as f:
            json.dump(run_record, f, indent=2)
        print(f"Run record saved → {record_path}")

        n_saved = 0
        for month_str in months:
            print(f"\n--- Month {month_str} ---")
            if self.predict_month(test_year, month_str, verbose=verbose):
                n_saved += 1

        print(f"\nDone. Saved predictions for {n_saved}/{len(months)} months.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run boundary condition model predictions for a test year."
    )
    parser.add_argument("--test_year", type=int, required=True)
    parser.add_argument("--month", default=None)
    parser.add_argument("--region", default=None)
    parser.add_argument("--reference_model", required=True)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--model_save_name", default=None)
    parser.add_argument("--dry_run", action="store_true")

    args = parser.parse_args()

    client, cluster = make_cluster()
    try:
        print("Building BoundaryPredictor...")
        predictor = BoundaryPredictor.from_args(args)
        months = None
        if args.month is not None:
            months = [normalize_month(args.month)]
        predictor.predict_year(args.test_year, months=months)
    finally:
        if cluster is not None:
            cluster.close()
            client.close()


if __name__ == "__main__":
    main()