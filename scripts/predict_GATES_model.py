"""
predict_GATES_model.py

Load a trained GATES model and run monthly predictions for a given test year.
Predictions are saved as per-month NetCDF files with both transformed and
original-space outputs alongside the ground-truth footprints.

Usage
-----
    python predict_GATES_model.py \\
        --test_year 2019 \\
        --reference_model my_model_20240115_143022 \\
        [--month 06] \\
        [--region BRAZIL] \\
        [--model_path /path/to/models/] \\
        [--checkpoint best|<epoch_int>] \\
        [--save_path /path/to/outputs/] \\
        [--model_save_name custom_name] \\
        [--dry_run]

Notes
-----
- All data and model configuration is read from the reference model's saved
  training settings (``training_outputs/training_settings_<model_name>.json``).
  No separate parameter file is needed.
- ``reference_model`` accepts either a full timestamped directory name
  (e.g. ``my_model_20240115_143022``) or a base name (e.g. ``my_model``),
  in which case the most recently created matching directory is used.
- ``month`` accepts either an integer month or a zero-padded string and
  restricts prediction to that single month.
- ``region`` overrides the region stored in the training settings before data
  loading and path resolution.
- ``model_save_name`` defaults to the reference model's base name (timestamp
  stripped). Pass an explicit name when predicting on a different region or size.
- Dynamic edges configuration is read from the training settings and
  reconstructed automatically; no extra arguments are needed.
- Output NetCDF files are written to
  ``{save_path}/{model_save_name}/predictions/predictions_{year}_{month}.nc``.
  If the region or prediction size differ from the training settings, the
  prediction folder name is automatically suffixed with the region and/or size
  (e.g. ``predictions_BRAZIL``).
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

sys.path.insert(0, "/user/work/ef17148/GCN/graphnet/")
sys.path.insert(1, "/user/work/ef17148/GCN/graphnet/graphnet_LPDM_emulator/")

import gates
import gates.config
import gates.data.datasets as gates_datasets
from gates.training.training import load_GATES_data, make_cluster, setup_dynamic_edges
from gates.training.training_helperfuns import load_parameter_file
from gates.training.training_dataclasses import PathContext

from gates.model.forecast import GraphSatelliteForecaster


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def find_model_dir(model_path: str, reference_model: str) -> Path:
    """Resolve the reference model directory.

    Accepts either a full timestamped name (``my_model_20240115_143022``) for
    an exact lookup, or a base name (``my_model``) to select the most recently
    created matching directory.
    """
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
    """Load scalers, grid, and training settings from a model's training_outputs folder."""
    outputs_dir = model_dir / "training_outputs"

    with open(outputs_dir / f"scalers_{model_name}.pickle", "rb") as f:
        scalers = pickle.load(f)

    with open(outputs_dir / f"grid_{model_name}.pickle", "rb") as f:
        grid = pickle.load(f)

    training_params = load_parameter_file(outputs_dir / f"training_settings_{model_name}.json")
    if training_params is None:
        raise RuntimeError(f"Failed to load training settings from {outputs_dir}")

    return scalers, grid, training_params


def determine_save_name(model_save_name_arg, model_dir: Path) -> str:
    """Determine the model_save_name used to organise output files.

    Defaults to the reference model's base name (timestamp stripped). Pass an
    explicit name when predicting on a different region or size.
    """
    if model_save_name_arg:
        return model_save_name_arg

    return re.sub(r"_\d{8}_\d{6}$", "", model_dir.name)



def normalize_month(month) -> str:
    """Normalize a month argument to a zero-padded two-digit string."""
    try:
        month_int = int(str(month).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid month value: {month!r}") from exc

    if not 1 <= month_int <= 12:
        raise ValueError(f"Month must be between 1 and 12, got {month!r}")

    return f"{month_int:02d}"


def load_checkpoint(
    model: torch.nn.Module,
    model_dir: Path,
    model_name: str,
    checkpoint: str,
    device: torch.device,
):
    """Load model weights from a checkpoint file."""
    if checkpoint == "best":
        path = model_dir / f"{model_name}_best.pt"
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"Loaded best model checkpoint from {path}")
    else:
        epoch = int(checkpoint)
        path = model_dir / f"{model_name}_{epoch}.pt"
        state = torch.load(path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        print(f"Loaded epoch {epoch} checkpoint from {path}")


@torch.no_grad()
def run_inference(model: torch.nn.Module, loader, device: torch.device) -> np.ndarray:
    """Run the model over all batches in loader; return concatenated predictions."""
    model.eval()
    preds = []
    for batch in loader:
        preds.append(model(batch[0].to(device)).cpu())
    return torch.cat(preds, dim=0).numpy()


# ---------------------------------------------------------------------------
# Predictor class
# ---------------------------------------------------------------------------

class GATESPredictor:
    """
    A loaded GATES model ready to generate predictions for arbitrary years/months.

    Initialise once via ``GATESPredictor.from_args(args)``; then call
    ``predict_month(year, month)`` or ``predict_year(year)`` as needed.

    Attributes
    ----------
    model_name : str
        Full timestamped name of the reference model directory.
    model_save_name : str
        Name used to organise output NetCDF files.
    save_path : Path
        Root directory under which ``{model_save_name}/`` is created.
    dynamic_edges_params : dict
        Keyword arguments reconstructed from the training settings and unpacked
        into ``GraphSatelliteForecaster`` at construction. Empty dict when the
        training run did not use dynamic edges.
    dry_run : bool
        When True, only the first month is processed with a coarse sampling
        frequency (``freq=60``) for fast pipeline verification.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        scalers: dict,
        model_name: str,
        model_save_name: str,
        prediction_folder_name: str,
        save_path: Path,
        device: torch.device,
        data_params: dict,
        input_variables: dict,
        datapath_args: dict,
        nans_to_zeros: bool,
        test_batch_size: int,
        checkpoint: str,
        parameter_file: str,
        dynamic_edges_params: dict,
        dry_run: bool = False,
    ):
        self.model = model
        self.scalers = scalers
        self.model_name = model_name
        self.model_save_name = model_save_name
        self.prediction_folder_name = prediction_folder_name
        self.save_path = Path(save_path)
        self.device = device
        self.data_params = data_params
        self.input_variables = input_variables
        self.datapath_args = datapath_args
        self.nans_to_zeros = nans_to_zeros
        self.test_batch_size = test_batch_size
        self.checkpoint = checkpoint
        self.parameter_file = parameter_file
        self.dynamic_edges_params = dynamic_edges_params
        self.dry_run = dry_run

    @classmethod
    def from_args(cls, args):
        """Build a GATESPredictor from parsed CLI arguments.

        All data and model configuration comes from the reference model's saved
        training settings. If ``num_features`` is stored in the training
        settings it is used directly; otherwise ``args.test_year`` is probed
        month-by-month to determine the input feature dimension. Dynamic edges
        configuration is read from the training settings and reconstructed via
        ``setup_dynamic_edges`` before the model is instantiated. The returned
        predictor works for any year with the same variable configuration.
        """
        cfg = gates.config.get_config()

        # Reference model — all configuration comes from its saved training settings
        model_path = args.model_path or cfg.save_models_dir
        model_dir = find_model_dir(model_path, args.reference_model)
        model_name = model_dir.name
        print(f"Reference model: {model_dir}")

        print("Loading scalers, grid, and training settings...")
        scalers, grid, training_params = load_model_artifacts(model_dir, model_name)

        # Save name and output path
        model_save_name = determine_save_name(args.model_save_name, model_dir)
        save_path = Path(args.save_path) / model_save_name if args.save_path else model_dir 
        ## attach args.region or args.size to prediction_folder_name if provided
        prediction_folder_name = "predictions" + (f"_{args.region}" if args.region else "") + (f"_size{args.size}" if args.size else "")

        print(f"model_save_name : {model_save_name}")
        print(f"Predictions root: {save_path / prediction_folder_name}/")

        # Data config — taken entirely from the reference model's training settings
        paths_ctx = PathContext(
            model_save_dir=model_path, model_name=model_name, model_path=model_dir
        )
        datapath_args = paths_ctx.resolve_datapath_args(training_params)
        data_params = copy.deepcopy(training_params["train_load_data"])
        data_params.update(training_params.get("test_load_data", {}))
        if args.region:
            data_params["region"] = args.region
            # remove parameter domain if it is in data_params
            data_params.pop("domain", None)
        input_variables = training_params["variables"]

        # remove parameters "years" and "months" from the data_params since they will be set dynamically during prediction
        data_params.pop("years", None)
        data_params.pop("months", None)
        

        # Check if the parameter file specifies the number of input features; if not, probe one month of data to determine it.
        if "num_features" in training_params.keys():
            feature_dim = training_params["num_features"]
            print(f"Using feature_dim from training parameters: {feature_dim}")
        else:
            print(f"\nProbing {args.test_year} data to determine feature_dim...")
            feature_dim = None
            for probe_month in [f"{m:02d}" for m in range(1, 13)]:
                try:
                    probe_params = {**data_params, "year": str(args.test_year), "month": probe_month, "freq": 50}
                    _, probe_inputs = load_GATES_data(
                        probe_params, input_variables, datapath_args, verbose=False
                    )
                    if probe_inputs is not None and probe_inputs.sizes.get("fp_time", 0) > 0:
                        feature_dim = probe_inputs.sizes["variable_name"]
                        print(f"  feature_dim = {feature_dim} (from month {probe_month})")
                        break
                except Exception:
                    continue

        if feature_dim is None:
            raise RuntimeError(
                f"Could not load any data for {args.test_year} to determine feature_dim."
            )

        # Dynamic edges
        if training_params.get("dynamic_edges", None) is not None:
            if isinstance(training_params["dynamic_edges"], dict):
                dynamic_edges_params = setup_dynamic_edges(input_names=scalers["input_names"], **training_params["dynamic_edges"])
            elif training_params.get("dynamic_edges") is True:
                dynamic_edges_params = setup_dynamic_edges(input_names=scalers["input_names"])
            else:
                dynamic_edges_params = {}
        else:
            dynamic_edges_params = {}

        # Model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        model = GraphSatelliteForecaster(
            grid, whole_world=False, feature_dim=feature_dim,
            **training_params["model_parameters"],
            **dynamic_edges_params,
        )
        model = model.to(device)
        load_checkpoint(model, model_dir, model_name, args.checkpoint, device)

        training_settings_path = str(
            model_dir / "training_outputs" / f"training_settings_{model_name}.json"
        )

        return cls(
            model=model,
            scalers=scalers,
            model_name=model_name,
            model_save_name=model_save_name,
            prediction_folder_name=prediction_folder_name,
            save_path=save_path,
            device=device,
            data_params=data_params,
            input_variables=input_variables,
            datapath_args=datapath_args,
            nans_to_zeros=training_params.get("dataloader", {}).get("nans_to_zeros", True),
            test_batch_size=training_params.get("dataloader", {}).get("test_batch_size", 5),
            checkpoint=args.checkpoint,
            parameter_file=training_settings_path,
            dynamic_edges_params=dynamic_edges_params,
            dry_run=args.dry_run,
        )

    # ------------------------------------------------------------------
    # Core prediction methods
    # ------------------------------------------------------------------

    def predict_month(self, test_year: int, month_str: str, verbose: bool = True) -> bool:
        """Load one month of data, run inference, and save predictions as NetCDF.

        Returns True when predictions are saved, False when data is unavailable.
        """
        monthly_params = {**self.data_params, "year": str(test_year), "month": month_str}
        if self.dry_run:
            monthly_params["freq"] = 60
        else:
            monthly_params["freq"] = 1

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

        print(f"  Loading {inputs.sizes['fp_time']} timesteps into memory with size: {inputs.nbytes / 1e9:.2f} GB")
        inputs.load()  # ensure all data is in memory before scaling and inference

        # Scale inputs (transform only — scaler is already fitted)
        scaled_inputs = self.scalers["inputs_scaler"].transform(inputs)

        # Scale footprints via a FootprintDataset wrapper with the injected
        # pre-fitted scaler, so fp_nan_mask is handled consistently with training.
        fp_wrapper = gates_datasets.FootprintDataset(
            data.fp_xr, add_nan_mask=self.nans_to_zeros
        )
        fp_wrapper.scaler = self.scalers["fp_scaler"]
        scaled_fps_ds = fp_wrapper.transform(data.fp_xr)

        scaled_inputs, scaled_fps_ds = gates_datasets.trim_to_batch_size(
            scaled_inputs, scaled_fps_ds, self.test_batch_size
        )

        loader, _ = gates_datasets.make_dataloader(
            scaled_inputs,
            scaled_fps_ds,
            batch_size=self.test_batch_size,
            randomize=False,
            dataloader_params={"num_workers": 0, "persistent_workers": False, "prefetch_factor": None},
            flatten=True,
        )

        print("  Running inference...")
        preds_t = run_inference(self.model, loader, self.device)  # (n, flat_lat_lon [, 1])
        if preds_t.ndim == 3 and preds_t.shape[-1] == 1:
            preds_t = preds_t.squeeze(-1)                          # (n, flat_lat_lon)

        preds_o = self.scalers["fp_scaler"].inverse_transform(preds_t)
        
        fps_dataset = scaled_fps_ds.isel(time=slice(preds_t.shape[0]))  # trim to same time range as predictions
        fps_dataset["fp_transformed_pred"] = (("time", "lat", "lon"), preds_t.reshape(*fps_dataset.fp_original.shape))
        fps_dataset["fp_pred"] = (("time", "lat", "lon"), preds_o.reshape(*fps_dataset.fp_original.shape))

        fps_dataset["lat_coords"] = data.fp_xr.sel(time=fps_dataset.time).lat_coords
        fps_dataset["lon_coords"] = data.fp_xr.sel(time=fps_dataset.time).lon_coords

        attrs_dict={
                "creation_date": str(datetime.now()),
                "model_save_name": self.model_save_name,
                "reference_model": self.model_name,
                "test_year": str(test_year),
                "month": month_str,
            }
        
        fps_dataset.attrs.update(attrs_dict)
        
        #n_times = preds_t.time.values
        #size = data.size
        #time_vals = scaled_fps_ds.time.values[:n_times]
        #lat_vals = scaled_fps_ds.lat.values
        #lon_vals = scaled_fps_ds.lon.values
        #fp_orig = scaled_fps_ds["fp_original"].values[:n_times]
        #fp_trans = scaled_fps_ds["fp_transformed"].values[:n_times]
        

        # ds = xr.Dataset(
        #     {
        #         "fp_transformed_pred": (
        #             ["time", "lat", "lon"],
        #             preds_t.reshape(n_times, size, size),
        #             {"space": "transformed", "type": "prediction", "emulated_with": self.model_save_name},
        #         ),
        #         "fp_pred": (
        #             ["time", "lat", "lon"],
        #             preds_o.reshape(n_times, size, size),
        #             {"space": "original", "type": "prediction", "emulated_with": self.model_save_name},
        #         ),
        #         "fp_original": (
        #             ["time", "lat", "lon"],
        #             fp_orig,
        #             {"space": "original", "type": "truth"},
        #         ),
        #         "fp_transformed": (
        #             ["time", "lat", "lon"],
        #             fp_trans,
        #             {"space": "transformed", "type": "truth"},
        #         ),
        #     },
        #     coords={"time": time_vals, "lat": lat_vals, "lon": lon_vals},
        #     attrs={
        #         "creation_date": str(datetime.now()),
        #         "model_save_name": self.model_save_name,
        #         "reference_model": self.model_name,
        #         "test_year": str(test_year),
        #         "month": month_str,
        #     },
        # )

        out_dir = self.save_path / self.prediction_folder_name
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = f"predictions_{test_year}_{month_str}.nc" if not self.dry_run else f"predictions_{test_year}_{month_str}_DRYRUN.nc"
        out_path = out_dir / filename
        fps_dataset.to_netcdf(out_path)
        print(f"  Saved → {out_path}")
        return True

    def predict_year(self, test_year: int, months: list = None, verbose: bool = True):
        """Run predictions for every month of ``test_year`` and save a run record.

        Parameters
        ----------
        test_year : int
            The year to predict.
        months : list of str, optional
            Month strings to process, e.g. ``["01", "06"]``. Defaults to all 12 months.
        verbose : bool
            Passed through to data loading.
        """
        if months is None:
            months = [f"{m:02d}" for m in range(1, 13)]

        if self.dry_run:
            months = months[:1]
            print(f"Dry run: processing only month {months[0]} with freq=60.")

        out_dir = self.save_path / self.prediction_folder_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Write run record before predictions start so there is always a trace,
        # even if the run fails partway through.
        run_record = {
            "created_at": datetime.now().isoformat(),
            "test_year": test_year,
            "reference_model": self.model_name,
            "model_save_name": self.model_save_name,
            "checkpoint": self.checkpoint,
            "parameter_file": self.parameter_file,
            "save_path": str(out_dir),
            "region": self.data_params.get("region"),
            "size": self.data_params.get("size"),
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
        description=(
            "Run GATES model predictions for a test year. All configuration is "
            "read from the reference model's saved training settings. "
            "Example: python predict_GATES_model.py "
            "--test_year 2019 --reference_model my_model_20240115_143022"
        )
    )
    parser.add_argument(
        "--test_year", type=int, required=True, help="Year to run predictions for"
    )
    parser.add_argument(
        "--month",
        default=None,
        help=(
            "Optional month to predict. Accepts an integer month (1-12) or a "
            "zero-padded string (01-12). If omitted, all months are predicted."
        ),
    )
    parser.add_argument(
        "--region",
        default=None,
        help=(
            "Region to predict footprints for. Optional, by default will use the training region."
        ),
    )

    parser.add_argument(
        "--size",
        default=None,
        help=(
            "Footprint size to predict. Optional, by default will use the training size. THIS IS CURRENTLY A PLACEHOLDER- size is always taken from the training settings. Future versions may allow overriding this."
        ),
    )

    parser.add_argument(
        "--reference_model",
        required=True,
        help=(
            "Trained model name/directory. Pass the full timestamped name "
            "(e.g. my_model_20240115_143022) for an exact match, or the base "
            "name (e.g. my_model) to use the most recently created run."
        ),
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="Root directory to search for trained models (default: from config.yml)",
    )
    parser.add_argument(
        "--checkpoint",
        default="best",
        help="Checkpoint to load: 'best' (default) or an epoch integer",
    )
    parser.add_argument(
        "--save_path",
        default=None,
        help="Root directory for output NetCDF files (default: {model_dir}/predictions/)",
    )
    parser.add_argument(
        "--model_save_name",
        default=None,
        help=(
            "Name used to organise output files. Auto-derived from the reference "
            "model's base name when region and size match. Required when predicting "
            "on a different region or size."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help=(
            "Quick diagnostic run: process only the first month with freq=20 "
            "(every 20th sample). Useful for checking the pipeline before a full run."
        ),
    )

    args = parser.parse_args()

    client, cluster = make_cluster()
    try:
        print("Building Predictor")
        predictor = GATESPredictor.from_args(args)
        print("Predicting test year")
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
