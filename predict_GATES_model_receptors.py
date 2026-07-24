"""
predict_GATES_model_receptors.py

Load a trained GATES model (sample_mode="receptors") and make predictions on its
held-out validation or test set, or a new, unseen region.

Usage
-----
    python predict_GATES_model_receptors.py \\
        --reference_model my_model_20240115_143022 \\
        --predict_mode val|test|region \\
        --region Taranaki1a \\
        [--model_path /path/to/models/] \\
        [--checkpoint best|<epoch_int>] \\
        [--save_path /path/to/outputs/] \\
        [--model_save_name custom_name] \\
        [--sample_plot]

Notes
-----
- All data and model configuration is read from the reference model's saved
  training settings (``training_outputs/training_settings_<model_name>.json``)
  and its saved receptor split (``training_outputs/receptor_split_<model_name>.json``).
  No separate parameter file is needed.
- ``reference_model`` accepts either a full timestamped directory name
  (e.g. ``my_model_20240115_143022``) or a base name (e.g. ``my_model``),
  in which case the most recently created matching directory is used.
- This script only works for models trained with ``sample_mode: "receptors"``
  (see predict_GATES_model.py for the satellite-mode pathway).
- ``--predict_mode region`` ignores the saved split entirely and fully loads ``--region``
  (met data included) instead, independent of the region(s) used during training. If
  ``--region`` was actually part of the training split, a warning is printed.
- Output NetCDF file is written to
  ``{save_path}/{model_save_name}/predictions/predictions_{val|test|region_<region>}_{checkpoint_tag}.nc``.

"""

import sys
import copy
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import xarray as xr

import gates
import gates.config
import gates.data.datasets as gates_datasets
from gates import LoadReceptorData
from gates.data.datasets import get_square_satellite_inputs_v2
from gates.training.training import make_cluster, setup_dynamic_edges
from gates.training.training_helperfuns import load_parameter_file
from gates.training.training_dataclasses import PathContext

from gates.model.forecast import GraphSatelliteForecaster

from predict_GATES_model import (
    find_model_dir,
    load_model_artifacts,
    determine_save_name,
    load_checkpoint,
    run_inference,
)


# ---------------------------------------------------------------------------
# Receptor data loading
# ---------------------------------------------------------------------------

def load_receptor_data_full(data_parameters, input_variables, datapath_args={}, verbose=True, load_into_memory=False):
    """
    Load all regions' receptor footprints and inputs without applying any train/val/test
    split. Used to reconstruct the samples recorded in a saved receptor_split JSON.
    """
    if "met_args" in data_parameters and "met_args" in datapath_args:
        merged_met_args = {**data_parameters["met_args"], **datapath_args["met_args"]}
        data_parameters["met_args"] = merged_met_args

    if "regions" in data_parameters and isinstance(data_parameters["regions"], list):
        regions = data_parameters["regions"]
    else:
        regions = [data_parameters.get("region")]

    base_params = {
        k: v for k, v in data_parameters.items()
        if k not in ("year", "years", "month", "months", "load_into_memory", "regions", "region")
    }

    met_args = dict(base_params.get("met_args", {}))
    met_args.setdefault("met_variables", input_variables.get("met_variables", []))
    met_args.setdefault("met_levels", input_variables.get("met_levels", []))
    base_params["met_args"] = met_args
    datapath_args.pop("met_args", None)

    all_inputs, all_fp_xr = [], []
    for region in regions:
        if verbose:
            print("Loading receptor data for region:", region)

        data = LoadReceptorData("*", region=region, **base_params, **datapath_args, verbose=verbose, load_everything=True)
        inputs, data = get_square_satellite_inputs_v2(data, **input_variables, verbose=verbose)

        data.fp_xr = data.fp_xr.assign_coords(region=("sample_id", [region] * data.fp_xr.sizes["sample_id"]))
        inputs = inputs.assign_coords(region=("sample_id", [region] * inputs.sizes["sample_id"]))

        all_inputs.append(inputs)
        all_fp_xr.append(data.fp_xr)

    fp_xr = xr.concat(all_fp_xr, dim="sample_id")
    inputs = xr.concat(all_inputs, dim="sample_id")

    sample_ids = np.arange(1, fp_xr.sizes["sample_id"] + 1)
    fp_xr = fp_xr.assign_coords(sample_id=sample_ids)
    inputs = inputs.assign_coords(sample_id=sample_ids)

    if load_into_memory:
        fp_xr = fp_xr.compute()
        inputs = inputs.compute()

    return fp_xr, inputs


def filter_receptor_split(fp_xr, inputs, split_lookup, split_name):
    """
    Select the subset of a fully-loaded (unsplit) receptor dataset from
    load_receptor_data_full that matches a saved (region, receptor, time) split recorded
    during training.
    """
    predict_mode = {(d["region"], d["receptor"], d["time"]) for d in split_lookup[split_name]}
    keys = list(zip(fp_xr.region.values, fp_xr.receptor.values.astype(int), [str(t) for t in fp_xr.time.values]))
    mask = np.array([k in predict_mode for k in keys])

    if not mask.any():
        raise ValueError(
            f"No samples matched the saved '{split_name}' split — check that the loaded "
            "regions/data match the training run that produced the split file."
        )

    matched_ids = fp_xr.sample_id.values[mask]
    return fp_xr.sel(sample_id=matched_ids), inputs.sel(sample_id=matched_ids)


# ---------------------------------------------------------------------------
# Predictor class
# ---------------------------------------------------------------------------

class GATESReceptorPredictor:
    """
    A loaded GATES model (sample_mode="receptors"), ready to run inference on its
    saved validation/test split, or on a completely new, unseen region.

    Initialise once via ``GATESReceptorPredictor.from_args(args)``; then call
    ``predict("val")``, ``predict("test")``, or ``predict("region", region="Taranaki1a")``.
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
        split_lookup: dict,
        nans_to_zeros: bool,
        test_batch_size: int,
        checkpoint: str,
        parameter_file: str,
        dynamic_edges_params: dict,
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
        self.split_lookup = split_lookup
        self.nans_to_zeros = nans_to_zeros
        self.test_batch_size = test_batch_size
        self.checkpoint = checkpoint
        self.parameter_file = parameter_file
        self.dynamic_edges_params = dynamic_edges_params

    @classmethod
    def from_args(cls, args):
        """
        Build a GATESReceptorPredictor from parsed CLI arguments.

        All data and model configuration comes from the reference model's saved
        training settings and its saved receptor split. Dynamic edges configuration
        is read from the training settings and reconstructed via setup_dynamic_edges
        before the model is instantiated.
        """
        cfg = gates.config.get_config()

        # Reference model — all configuration comes from its saved training settings
        model_path = args.model_path or cfg.save_models_dir
        model_dir = find_model_dir(model_path, args.reference_model)
        model_name = model_dir.name
        print(f"Reference model: {model_dir}")

        print("Loading scalers, grid, and training settings...")
        scalers, grid, training_params = load_model_artifacts(model_dir, model_name)

        if training_params.get("sample_mode") != "receptors":
            raise ValueError(
                f"Reference model {model_name!r} was trained with sample_mode="
                f"{training_params.get('sample_mode')!r}, not 'receptors'. Use "
                "predict_GATES_model.py for satellite-mode models."
            )

        # Save name and output path
        model_save_name = determine_save_name(args.model_save_name, model_dir)
        save_path = Path(args.save_path) / model_save_name if args.save_path else model_dir
        prediction_folder_name = "predictions"

        print(f"model_save_name : {model_save_name}")
        print(f"Predictions root: {save_path / prediction_folder_name}/")

        # Data config — taken entirely from the reference model's training settings.
        paths_ctx = PathContext(
            model_save_dir=model_path, model_name=model_name, model_path=model_dir
        )
        datapath_args = paths_ctx.resolve_datapath_args(training_params)
        data_params = copy.deepcopy(training_params["train_load_data"])
        input_variables = training_params["variables"]

        feature_dim = training_params.get("num_features")
        if feature_dim is None:
            raise RuntimeError(
                f"training_settings for {model_name!r} has no 'num_features' — cannot "
                "build the model. This should always be present for models trained "
                "after num_features was added to train_and_save_model."
            )
        print(f"Using feature_dim from training parameters: {feature_dim}")

        split_lookup_path = model_dir / "training_outputs" / f"receptor_split_{model_name}.json"
        split_lookup = load_parameter_file(split_lookup_path)
        if split_lookup is None:
            raise RuntimeError(f"Failed to load receptor split from {split_lookup_path}")

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
            split_lookup=split_lookup,
            nans_to_zeros=training_params.get("dataloader", {}).get("nans_to_zeros", True),
            test_batch_size=training_params.get("dataloader", {}).get("test_batch_size", 5),
            checkpoint=args.checkpoint,
            parameter_file=training_settings_path,
            dynamic_edges_params=dynamic_edges_params,
        )

    # ------------------------------------------------------------------
    # Core prediction method
    # ------------------------------------------------------------------

    def predict(self, predict_mode: str, region: str = None, verbose: bool = True) -> Path:
        """
        Run inference for a saved val/test split, or for a completely new region.

        predict_mode:
            - "val" / "test": reuse the exact (region, receptor, time) samples recorded
              in the saved receptor_split JSON from training.
            - "region": ignore the saved split entirely and fully load `region` (met data
              included), independent of the region(s) used during training. All loaded
              samples for that region are used.
        """
        if predict_mode in ("val", "test"):
            print(f"Loading full receptor data for regions: {self.data_params.get('regions') or [self.data_params.get('region')]}")
            fp_xr_full, inputs_full = load_receptor_data_full(
                self.data_params, self.input_variables, self.datapath_args, verbose=verbose,
            )
            fp_xr, inputs = filter_receptor_split(fp_xr_full, inputs_full, self.split_lookup, predict_mode)
            print(f"Matched {fp_xr.sizes['sample_id']} samples for split '{predict_mode}'")
            name_tag = predict_mode
        elif predict_mode == "region":
            if not region:
                raise ValueError("predict_mode='region' requires a --region.")

            trained_regions = {
                d["region"] for split in ("train", "val", "test") for d in self.split_lookup.get(split, [])
            }
            region_seen_in_training = region in trained_regions
            if region_seen_in_training:
                print(
                    f"WARNING: region '{region}' appears in the saved receptor split from "
                    "training — it is NOT unseen. Predictions on it do not test generalisation "
                    "to a new region."
                )

            region_params = copy.deepcopy(self.data_params)
            region_params.pop("regions", None)
            region_params["region"] = region
            print(f"Loading full receptor data for region: {region}")
            fp_xr, inputs = load_receptor_data_full(
                region_params, self.input_variables, self.datapath_args, verbose=verbose,
            )
            print(f"Loaded {fp_xr.sizes['sample_id']} samples for region '{region}'")
            name_tag = f"region_{region}"
        else:
            raise ValueError(f"predict_mode must be 'val', 'test', or 'region', got {predict_mode!r}")

        print(f"  Loading into memory with size: {inputs.nbytes / 1e9:.2f} GB")
        inputs.load()

        # Scale inputs (transform only — scaler is already fitted)
        scaled_inputs = self.scalers["inputs_scaler"].transform(inputs)

        # Scale footprints via a FootprintDataset wrapper with the injected
        # pre-fitted scaler, so fp_nan_mask is handled consistently with training.
        fp_wrapper = gates_datasets.FootprintDataset(fp_xr, add_nan_mask=self.nans_to_zeros)
        fp_wrapper.scaler = self.scalers["fp_scaler"]
        scaled_fps_ds = fp_wrapper.transform(fp_xr)

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
        preds_t = run_inference(self.model, loader, self.device)
        if preds_t.ndim == 3 and preds_t.shape[-1] == 1:
            preds_t = preds_t.squeeze(-1)

        preds_o = self.scalers["fp_scaler"].inverse_transform(preds_t)

        out_ds = scaled_fps_ds.isel(sample_id=slice(preds_t.shape[0]))  # trim to same range as predictions
        out_ds["fp_transformed_pred"] = (("sample_id", "lat", "lon"), preds_t.reshape(*out_ds.fp_original.shape))
        out_ds["fp_pred"] = (("sample_id", "lat", "lon"), preds_o.reshape(*out_ds.fp_original.shape))

        matched = fp_xr.sel(sample_id=out_ds.sample_id)
        out_ds["lat_coords"] = matched.lat_coords
        out_ds["lon_coords"] = matched.lon_coords
        out_ds["region"] = matched.region
        out_ds["receptor"] = matched.receptor

        out_ds.attrs.update({
            "creation_date": str(datetime.now()),
            "model_save_name": self.model_save_name,
            "reference_model": self.model_name,
            "predict_mode": predict_mode,
        })

        checkpoint_tag = "best" if str(self.checkpoint) == "best" else f"epoch{self.checkpoint}"

        out_dir = self.save_path / self.prediction_folder_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"predictions_{name_tag}_{checkpoint_tag}.nc"
        out_ds.to_netcdf(out_path)
        print(f"  Saved → {out_path}")
        return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run GATES model predictions on the saved val/test receptor split. All "
            "configuration is read from the reference model's saved training settings. "
            "Example: python predict_GATES_model_receptors.py "
            "--reference_model my_model_20240115_143022 --predict_mode val"
        )
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
        "--predict_mode",
        required=True,
        choices=["val", "test", "region"],
        help=(
            "'val'/'test' reuse the saved receptor split from training. 'region' ignores "
            "the saved split and fully loads a new, unseen region instead (requires --region)."
        ),
    )
    parser.add_argument(
        "--region",
        default=None,
        help="Region to fully load when --predict_mode is 'region' (e.g. Taranaki1a). Ignored otherwise.",
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
        help="Name used to organise output files. Auto-derived from the reference model's base name.",
    )
    parser.add_argument(
        "--sample_plot",
        action="store_true",
        help="After prediction, also generate a quick sample plot via plot_sample_GATES_predictions.py.",
    )

    args = parser.parse_args()

    if args.predict_mode == "region" and not args.region:
        parser.error("--region is required when --predict_mode is 'region'.")

    client, cluster = make_cluster()
    try:
        print("Building Predictor")
        predictor = GATESReceptorPredictor.from_args(args)
        print(f"Predicting '{args.predict_mode}'" + (f" (region={args.region})" if args.predict_mode == "region" else ""))
        out_path = predictor.predict(args.predict_mode, region=args.region)
    finally:
        if cluster is not None:
            cluster.close()
            client.close()

    if args.sample_plot:
        # create plot of 4 sample predictions
        plot_script = Path(__file__).resolve().parent / "plot_sample_GATES_predictions.py"
        plot_cmd = [
            sys.executable, str(plot_script),
            "--reference_model", predictor.model_name,
            "--predictions_file", out_path.name,
        ]
        if args.model_path:
            plot_cmd += ["--model_path", args.model_path]
        if args.save_path:
            plot_cmd += ["--save_path", args.save_path]
        if args.model_save_name:
            plot_cmd += ["--model_save_name", args.model_save_name]
        print(f"Running sample plot: {' '.join(plot_cmd)}")
        subprocess.run(plot_cmd, check=True)


if __name__ == "__main__":
    main()