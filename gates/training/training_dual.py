"""Setup helpers for the dual-head model (footprint + background).

This module mirrors the structure of ``gates.training.training`` (footprint model) and
``gates.training.training_background`` (background model), but wires up a single
``GraphSatelliteDualForecaster`` that carries both decoder heads. The data pipeline reuses
``load_GATES_data_with_bg`` (which returns aligned footprints, inputs, backgrounds and aux
data), scales the inputs (with auxiliary CAMS appended) and footprints, normalises the
backgrounds, and builds a combined dataloader that yields ``(inputs, fps, background)``.
"""

import copy

import torch
import torch.optim as optim

import gates.data.datasets as gates_datasets
import gates.evaluation.loss_functions as gates_losses

from .training import setup_input_dataset, setup_fp_dataset
from .training_background import concat_auxiliary_to_inputs
from .training_helperfuns import EarlyStopping
from .training_dataclasses import DualModelContext

from model.forecast import GraphSatelliteDualForecaster


def _trim_dual_to_batch_size(inputs, fps, bgs, batch_size):
    """Trim the trailing timepoints from inputs/fps/bgs so the count divides batch_size.

    All three are sorted by time identically, so removing the same number of trailing
    timepoints keeps them aligned.
    """
    n = inputs.sizes["fp_time"]
    remainder = n % batch_size
    if remainder != 0:
        inputs = inputs.isel(fp_time=slice(None, n - remainder))
        fps = fps.isel(time=slice(None, n - remainder))
        bgs = bgs.isel(time=slice(None, n - remainder))
    return inputs, fps, bgs


def setup_dual_dataloaders(parameters, train_inputs, train_fps, train_bgs,
                            test_inputs, test_fps, test_bgs,
                            train_auxiliary_cams=None, test_auxiliary_cams=None):
    """Scale inputs/footprints, append aux CAMS, and build dual train/test DataLoaders.

    Args:
        parameters (dict): Full parameter dict (needs 'background_setup', 'dataloader',
            'input_scaler', 'fp_scaler').
        train_inputs / test_inputs (xr.DataArray): Met inputs (fp_time, lat, lon, variable_name).
        train_fps / test_fps (xr.DataArray or xr.Dataset): Footprint targets (time, lat, lon).
        train_bgs / test_bgs (xr.DataArray): Normalised background targets (time, num_classes).
        train_auxiliary_cams / test_auxiliary_cams (xr.DataArray or None): Aux CAMS features.

    Returns:
        train_loader (DataLoader): yields (inputs, fps, background).
        test_loader (DataLoader): yields (inputs, fps, background).
        fp_labels (list): footprint variable label(s).
        test_scaled_fp (xr.Dataset): transformed test footprints (fp_transformed/fp_original/mask),
            reused for footprint-head evaluation.
        scalers (dict): {'inputs_scaler', 'fp_scaler'}.
    """
    # --- Inputs (scale, then append auxiliary CAMS, as in the background pipeline) ---
    input_dataset = setup_input_dataset(parameters, train_inputs)
    train_scaled_inputs = input_dataset.transform(train_inputs)
    test_scaled_inputs = input_dataset.transform(test_inputs)

    use_auxiliary_bc = parameters.get("background_setup", {}).get("use_auxiliary_bc", False)
    if use_auxiliary_bc:
        train_auxiliary_cams.load()
        test_auxiliary_cams.load()
        print("Concatenating inputs and auxiliary cams")
        train_scaled_inputs = concat_auxiliary_to_inputs(train_scaled_inputs, train_auxiliary_cams)
        test_scaled_inputs = concat_auxiliary_to_inputs(test_scaled_inputs, test_auxiliary_cams)

    # --- Footprints (scale, as in the footprint pipeline) ---
    fp_dataset = setup_fp_dataset(parameters, train_fps)
    train_scaled_fp = fp_dataset.transform(train_fps)
    test_scaled_fp = fp_dataset.transform(test_fps)

    dataloader_info = parameters.get("dataloader", {})
    batch_size = dataloader_info.get("batch_size", 5)
    test_batch_size = dataloader_info.get("test_batch_size", 5)
    dataloader_params = dataloader_info.get("dataloader_params", {})
    if "prefetch_factor" in dataloader_params and dataloader_params["prefetch_factor"] == 0:
        dataloader_params["prefetch_factor"] = None

    # Trim inputs/fps/bgs together so each divides its batch size and stays aligned
    train_scaled_inputs, train_scaled_fp, train_bgs = _trim_dual_to_batch_size(
        train_scaled_inputs, train_scaled_fp, train_bgs, batch_size
    )
    test_scaled_inputs, test_scaled_fp, test_bgs = _trim_dual_to_batch_size(
        test_scaled_inputs, test_scaled_fp, test_bgs, test_batch_size
    )

    train_loader, fp_labels = gates_datasets.make_dual_dataloader(
        train_scaled_inputs, train_scaled_fp, train_bgs,
        batch_size=batch_size, randomize=True,
        dataloader_params=dataloader_params, flatten=True
    )

    test_dataloader_params = {"num_workers": 0, "persistent_workers": False, "prefetch_factor": None}
    print("SPECIAL TEST PARAMS", test_dataloader_params)
    test_loader, fp_labels_test = gates_datasets.make_dual_dataloader(
        test_scaled_inputs, test_scaled_fp, test_bgs,
        batch_size=test_batch_size, randomize=False,
        dataloader_params=test_dataloader_params, flatten=True
    )

    if fp_labels != fp_labels_test:
        raise ValueError("Train and test footprint labels do not match - check the data loading.")

    scalers = {"inputs_scaler": input_dataset.scaler, "fp_scaler": fp_dataset.scaler}

    return train_loader, test_loader, fp_labels, test_scaled_fp, scalers


def setup_dual_model(parameters, training_ctx, paths_ctx):
    """Instantiate the dual-head model, optimizer, per-head criteria and early stopping.

    The loss config lives under ``parameters['loss_functions']`` with separate entries for
    each head, e.g.::

        "loss_functions": {
            "fp_criterion": "gates_losses.MSELoss",
            "fp_criterion_test": "gates_losses.MSELoss",
            "fp_criterion_params": {},
            "bg_criterion": "torch.nn.MSELoss",
            "bg_criterion_test": "torch.nn.MSELoss",
            "bg_criterion_params": {},
            "bg_loss_weight": 1.0
        }

    Returns:
        model (nn.Module), model_ctx (DualModelContext).
    """
    lr = parameters["learning_rate"]

    model_parameters = copy.deepcopy(parameters["model_parameters"])
    decoder = model_parameters.pop("decoder", "conv")
    num_classes = model_parameters.pop("num_classes", 1)

    model = GraphSatelliteDualForecaster(
        training_ctx.grid,
        whole_world=False,
        feature_dim=training_ctx.n_variables,
        aux_dim=training_ctx.aux_dim,
        num_classes=num_classes,
        input_height=training_ctx.size,
        input_width=training_ctx.size,
        decoder_type=decoder,
        **model_parameters,
    )

    loss_cfg = parameters["loss_functions"]

    # nan-mask handling for the footprint head (matches setup_GATES_model)
    if parameters.get("dataloader", {}).get("nans_to_zeros", True):
        nan_mask_label = "fp_nan_mask"
    else:
        nan_mask_label = None

    fp_loss_fn = eval(loss_cfg["fp_criterion"])
    fp_loss_fn_test = eval(loss_cfg.get("fp_criterion_test", loss_cfg["fp_criterion"]))
    fp_params = loss_cfg.get("fp_criterion_params", {})
    # The test criterion can be a different loss (e.g. plain MSE) that does not accept the
    # training loss's params, so allow its own params and fall back to fp_params if unset.
    fp_test_params = loss_cfg.get("fp_criterion_test_params", fp_params)
    fp_criterion = fp_loss_fn(fp_labels=training_ctx.fp_labels, nan_mask_label=nan_mask_label, **fp_params)
    fp_criterion_test = fp_loss_fn_test(fp_labels=training_ctx.fp_labels, nan_mask_label=nan_mask_label, **fp_test_params)

    bg_loss_fn = eval(loss_cfg["bg_criterion"])
    bg_loss_fn_test = eval(loss_cfg.get("bg_criterion_test", loss_cfg["bg_criterion"]))
    bg_params = loss_cfg.get("bg_criterion_params", {})
    bg_criterion = bg_loss_fn(**bg_params)
    bg_criterion_test = bg_loss_fn_test(**bg_params)

    bg_loss_weight = loss_cfg.get("bg_loss_weight", 1.0)

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    early_stopping = EarlyStopping(
        patience=parameters["epochs"]["patience"],
        verbose=parameters.get("verbose", True),
        path=paths_ctx.model_path / f"{paths_ctx.model_name}_best.pt",
        use_wandb=parameters["use_wandb"],
        model_name=paths_ctx.model_name,
    )

    if torch.cuda.is_available():
        model.cuda()

    model_ctx = DualModelContext(
        model_name=parameters["model_name"],
        use_wandb=parameters["use_wandb"],
        device=training_ctx.device,
        optimizer=optimizer,
        fp_criterion=fp_criterion,
        fp_criterion_test=fp_criterion_test,
        bg_criterion=bg_criterion,
        bg_criterion_test=bg_criterion_test,
        bg_loss_weight=bg_loss_weight,
        lr=lr,
        early_stopping=early_stopping,
        epochs_num=parameters["epochs"]["training"],
        epochs_visualise=parameters["epochs"].get("visualize", 5),
        epochs_save=parameters["epochs"]["model_save"],
        epochs_patience=parameters["epochs"]["patience"],
    )

    return model, model_ctx


def initialise_dual_losses():
    """Loss tracking dict for the dual-head training run."""
    metrics_dict = {"nmae": [], "mse": [], "bias": [], "mae": [], "iou": []}
    flux_metrics_dict = {"corrcoef": [], "mae": [], "mean_bias": [], "r2_score": []}
    losses = {
        "train": [],
        "test": [],
        "train_fp": [],
        "test_fp": [],
        "train_bg": [],
        "test_bg": [],
        "test_bg_mae_denorm": [],
        "metrics_transformed": metrics_dict.copy(),
        "metrics_original": metrics_dict.copy(),
        "metrics_fluxes_static": {
            "uniform": flux_metrics_dict.copy(),
            "checkerboard": flux_metrics_dict.copy(),
            "checkerboard_10": flux_metrics_dict.copy(),
        },
    }
    return losses
