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

from model.forecast import GraphSatelliteBackgroundPredictor
import torch
import xarray as xr

sys.path.insert(0, "/user/work/yl18410/new_graphnet")
sys.path.insert(1, "/user/work/yl18410/new_graphnet/graphnet_LPDM_emulator")

import gates
import gates.config
import gates.data.datasets as gates_datasets
from gates.training.training import load_GATES_data, make_cluster
from gates.training.training_helperfuns import load_parameter_file
from gates.training.training_dataclasses import PathContext
# Reuse the exact same data/CAMS/background functions used during training, so
# inference computes boundary conditions and auxiliary CAMS features identically.
from gates.training.training_background import (
    load_GATES_data_with_bg,
    format_aux_data,
    normalize_boundary_data,
    concat_auxiliary_to_inputs,
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
        num_classes,
        detrend,
        use_auxiliary_bc,
        aux_indeces,
        load_into_memory,
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
        self.num_classes = num_classes
        self.detrend = detrend
        self.use_auxiliary_bc = use_auxiliary_bc
        self.aux_indeces = aux_indeces
        self.load_into_memory = load_into_memory
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
        data_params.pop("year", None)
        data_params.pop("months", None)

        # Domain for CAMS files (load_GATES_data_with_bg also derives it internally)
        region = data_params.get("region", "SAHARA")
        domain = cfg.domains[region]["domain_name"]

        # Background / CAMS settings — mirror train_boundary_model's defaults exactly
        # so inference builds backgrounds and auxiliary CAMS the same way as training.
        background_setup = training_params.get("background_setup", {})
        background_params = {
            "detrend": True,
            "use_auxiliary_bc": True,
            "auxilary_bc_levels": [4, 5, 6, 7],
        }
        background_params.update(background_setup)
        detrend = background_params["detrend"]
        use_auxiliary_bc = background_params["use_auxiliary_bc"]
        aux_indeces = background_params["auxilary_bc_levels"]

        load_into_memory = training_params.get("load_into_memory", False)
        output_format = training_params.get("output_format", "corrected")

        # num_classes drives both the boundary-output selection and the model head;
        # read it exactly as setup_boundary_model does.
        num_classes = training_params["model_parameters"].get("num_classes", 1)

        # Auxiliary CAMS feature count: 4 boundaries x len(height levels), matching
        # get_auxiliary_bc_data + format_aux_data. The model receives these as a
        # separate aux input, so encoder input width = feature_dim + aux_dim.
        aux_dim = 4 * len(aux_indeces) if use_auxiliary_bc else 0

        # num_features (saved) is the TOTAL input width (met + aux); the model's
        # feature_dim is the met-only count, so subtract aux_dim.
        total_features = training_params.get("num_features", None)
        if total_features is not None:
            feature_dim = total_features - aux_dim
            print(f"feature_dim={feature_dim} (met-only) from num_features={total_features}, aux_dim={aux_dim}")
        else:
            print("num_features not found in training settings, probing data to determine feature_dim...")
            feature_dim = None
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
                        feature_dim = probe_inputs.sizes["variable_name"]
                        print(f"  feature_dim = {feature_dim} (met-only, from month {probe_month})")
                        break
                except Exception:
                    continue

            if feature_dim is None:
                raise RuntimeError(
                    "Could not determine feature_dim from data. "
                    "Ensure num_features is saved in training_settings.json."
                )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        size = data_params.get("size", 10)

        # Build model the same way setup_boundary_model does: decoder and num_classes
        # are passed explicitly, not via **model_parameters (so pop them off a copy).
        model_params = copy.deepcopy(training_params["model_parameters"])
        decoder = model_params.pop("decoder", training_params.get("network_decoder", "conv"))
        model_params.pop("num_classes", None)


        model = GraphSatelliteBackgroundPredictor(
                grid, whole_world=False, feature_dim=feature_dim,
                aux_dim=aux_dim, num_classes=num_classes,
                input_height=size, input_width=size,
                **model_params
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
            num_classes=num_classes,
            detrend=detrend,
            use_auxiliary_bc=use_auxiliary_bc,
            aux_indeces=aux_indeces,
            load_into_memory=load_into_memory,
            output_format=output_format,
            dry_run=args.dry_run,
        )

    # ------------------------------------------------------------------
    # Core prediction methods
    # ------------------------------------------------------------------

    def predict_month(self, test_year: int, month_str: str, verbose: bool = True) -> bool:
        """Load one month of data, run inference, save predictions as NetCDF.

        Uses the same data/CAMS/background pipeline as training
        (load_GATES_data_with_bg → boundary selection → format_aux_data →
        normalize_boundary_data → concat_auxiliary_to_inputs), so the inputs and
        ground-truth boundary conditions are built identically to train time.
        """
        monthly_params = {
            **self.data_params,
            "year": str(test_year),
            "month": month_str,
            "freq": 60 if self.dry_run else 1,
        }

        # Load footprints, met inputs, background (bgs) and auxiliary CAMS exactly
        # as training does — same function, same detrend/aux options.
        try:
            fp_xr, inputs, bgs, aux_data = load_GATES_data_with_bg(
                monthly_params,
                input_variables=self.input_variables,
                datapath_args=self.datapath_args,
                verbose=verbose,
                load_into_memory=self.load_into_memory,
                detrend=self.detrend,
                use_aux_bc=self.use_auxiliary_bc,
                aux_indeces=self.aux_indeces,
            )
        except Exception as exc:
            print(f"  Skipping {month_str}: data load failed — {exc}")
            return False

        if inputs is None or inputs.sizes.get("fp_time", 0) == 0:
            print(f"  Skipping {month_str}: no timesteps loaded.")
            return False

        print(f"  Loaded {inputs.sizes['fp_time']} timesteps")

        # Select boundary outputs by num_classes, identical to train_and_save_model.
        if self.num_classes == 4:
            bgs = bgs[["north", "south", "east", "west"]].to_dataarray()
        else:
            bgs = bgs[["summed"]].to_dataarray()

        # Format auxiliary CAMS, then normalise both outputs and aux using the
        # training normalisation values (norm_vals) so scaling matches train time.
        if self.use_auxiliary_bc:
            aux_data = format_aux_data(aux_data, time_coord=fp_xr.time)

        norm_bgs, norm_aux, _ = normalize_boundary_data(
            bgs,
            aux_data=aux_data if self.use_auxiliary_bc else None,
            norm_vals=self.norm_vals,
        )

        # Scale met inputs with the fitted scaler saved at train time.
        scaled_inputs = self.scalers["inputs_scaler"].transform(inputs)

        # Append the normalised auxiliary CAMS features (same as training).
        if self.use_auxiliary_bc:
            norm_aux.load()
            scaled_inputs = concat_auxiliary_to_inputs(scaled_inputs, norm_aux)

        # Trim both inputs and outputs to a whole number of batches.
        scaled_inputs, norm_bgs = gates_datasets.trim_to_batch_size(
            scaled_inputs, norm_bgs, self.test_batch_size
        )

        loader = gates_datasets.make_boundary_dataloader(
            scaled_inputs,
            norm_bgs,
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
        n_pred = preds_norm.shape[0]

        # Ground-truth (normalised) boundary conditions, aligned (time, num_classes).
        norm_bgs = norm_bgs.transpose("time", ...)
        true_norm = norm_bgs.values[:n_pred]
        times = norm_bgs["time"].values[:n_pred]

        # Denormalise predictions and truth using the training output statistics.
        if self.norm_vals is not None:
            outputs_mean, outputs_std = self.norm_vals["outputs"]
            preds_denorm = (preds_norm * outputs_std) + outputs_mean
        else:
            preds_denorm = preds_norm

        ds = xr.Dataset(
            {
                "bc_pred_normalised": (["time", "num_classes"], preds_norm),
                "bc_pred": (["time", "num_classes"], preds_denorm),
                "bc_true_normalised": (["time", "num_classes"], true_norm),
            },
            coords={"time": times},
            attrs={
                "creation_date": str(datetime.now()),
                "model_save_name": self.model_save_name,
                "reference_model": self.model_name,
                "test_year": str(test_year),
                "month": month_str,
                "output_format": self.output_format,
                "num_classes": self.num_classes,
                "detrend": str(self.detrend),
                "use_auxiliary_bc": str(self.use_auxiliary_bc),
            },
        )
        if self.norm_vals is not None:
            ds["bc_true"] = (["time", "num_classes"], (true_norm * outputs_std) + outputs_mean)

        out_dir = self.save_path / self.model_save_name / self.prediction_folder_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"predictions_{test_year}_{month_str}.nc"
        ds.to_netcdf(out_path)
        print(f"  Saved → {out_path}")
        return True

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