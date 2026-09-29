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
                            train_auxiliary_cams=None, test_auxiliary_cams=None,
                            input_dataset=None):
    """Scale inputs/footprints, append aux CAMS, and build dual train/test DataLoaders.

    Args:
        parameters (dict): Full parameter dict (needs 'background_setup', 'dataloader',
            'input_scaler', 'fp_scaler').
        train_inputs / test_inputs (xr.DataArray): Met inputs (fp_time, lat, lon, variable_name).
        train_fps / test_fps (xr.DataArray or xr.Dataset): Footprint targets (time, lat, lon).
        train_bgs / test_bgs (xr.DataArray): Normalised background targets (time, num_classes).
        train_auxiliary_cams / test_auxiliary_cams (xr.DataArray or None): Aux CAMS features.
        input_dataset (optional): An already FITTED input-scaler wrapper (anything with
            ``transform(inputs)`` and a ``scaler`` attribute, e.g. the ``InputsDataset`` of an
            earlier run). When given, the input scaler is NOT refitted on ``train_inputs``: the
            inputs are transformed with the saved statistics. Used by ``predict_dual_model.py`` to
            apply a trained model to a data configuration it was not trained on (a different
            window size), where refitting would change the input space (principle 3). Default
            None = fit on ``train_inputs`` as during training.

    Returns:
        train_loader (DataLoader): yields (inputs, fps, background).
        test_loader (DataLoader): yields (inputs, fps, background).
        fp_labels (list): footprint variable label(s).
        test_scaled_fp (xr.Dataset): transformed test footprints (fp_transformed/fp_original/mask),
            reused for footprint-head evaluation.
        scalers (dict): {'inputs_scaler', 'fp_scaler'}.
    """
    # --- Inputs (scale, then append auxiliary CAMS, as in the background pipeline) ---
    if input_dataset is None:
        input_dataset = setup_input_dataset(parameters, train_inputs)
    else:
        print("Using a pre-fitted input scaler (not refitted on these training inputs)")
    train_scaled_inputs = input_dataset.transform(train_inputs)

    use_auxiliary_bc = parameters.get("background_setup", {}).get("use_auxiliary_bc", False)
    if use_auxiliary_bc:
        train_auxiliary_cams.load()
        print("Concatenating inputs and auxiliary cams")
        train_scaled_inputs = concat_auxiliary_to_inputs(train_scaled_inputs, train_auxiliary_cams)

    # --- Footprints (scale, as in the footprint pipeline) ---
    fp_dataset = setup_fp_dataset(parameters, train_fps)
    train_scaled_fp = fp_dataset.transform(train_fps)

    batch_size, _, dataloader_params = _dual_dataloader_settings(parameters)

    # Trim inputs/fps/bgs together so each divides its batch size and stays aligned
    train_scaled_inputs, train_scaled_fp, train_bgs = _trim_dual_to_batch_size(
        train_scaled_inputs, train_scaled_fp, train_bgs, batch_size
    )

    train_loader, fp_labels = gates_datasets.make_dual_dataloader(
        train_scaled_inputs, train_scaled_fp, train_bgs,
        batch_size=batch_size, randomize=True,
        dataloader_params=dataloader_params, flatten=True
    )

    test_loader, fp_labels_test, test_scaled_fp = setup_dual_eval_loader(
        parameters, input_dataset, fp_dataset, test_inputs, test_fps, test_bgs, test_auxiliary_cams)

    if fp_labels != fp_labels_test:
        raise ValueError("Train and test footprint labels do not match - check the data loading.")

    scalers = {"inputs_scaler": input_dataset.scaler, "fp_scaler": fp_dataset.scaler}

    return train_loader, test_loader, fp_labels, test_scaled_fp, scalers


def _dual_dataloader_settings(parameters):
    """``(batch_size, test_batch_size, dataloader_params)`` of ``parameters["dataloader"]``."""
    dataloader_info = parameters.get("dataloader", {})
    batch_size = dataloader_info.get("batch_size", 5)
    test_batch_size = dataloader_info.get("test_batch_size", 5)
    dataloader_params = dataloader_info.get("dataloader_params", {})
    if "prefetch_factor" in dataloader_params and dataloader_params["prefetch_factor"] == 0:
        dataloader_params["prefetch_factor"] = None
    return batch_size, test_batch_size, dataloader_params


def setup_dual_eval_loader(parameters, input_dataset, fp_dataset, test_inputs, test_fps, test_bgs,
                           test_auxiliary_cams=None):
    """Scale the test set with already FITTED scalers and build its (unshuffled) DataLoader.

    The test half of :func:`setup_dual_dataloaders`, shared with the month-cache pipeline
    (``gates/training/lean_dual_data.py``), which builds the training tensors itself.

    Args:
        parameters (dict): Full parameter dict (needs 'background_setup', 'dataloader').
        input_dataset: Fitted input-scaler wrapper (``transform(inputs)``).
        fp_dataset: Fitted ``FootprintDataset``.
        test_inputs (xr.DataArray): Met inputs (fp_time, lat, lon, variable_name).
        test_fps (xr.DataArray or xr.Dataset): Footprint targets (time, lat, lon).
        test_bgs (xr.DataArray): Normalised background targets (time, num_classes).
        test_auxiliary_cams (xr.DataArray or None): Aux CAMS features.

    Returns:
        test_loader (DataLoader): yields (inputs, fps, background).
        fp_labels (list): footprint variable label(s).
        test_scaled_fp (xr.Dataset): transformed (and trimmed) test footprints.
    """
    test_scaled_inputs = input_dataset.transform(test_inputs)
    if parameters.get("background_setup", {}).get("use_auxiliary_bc", False):
        test_auxiliary_cams.load()
        test_scaled_inputs = concat_auxiliary_to_inputs(test_scaled_inputs, test_auxiliary_cams)

    test_scaled_fp = fp_dataset.transform(test_fps)

    _, test_batch_size, _ = _dual_dataloader_settings(parameters)
    test_scaled_inputs, test_scaled_fp, test_bgs = _trim_dual_to_batch_size(
        test_scaled_inputs, test_scaled_fp, test_bgs, test_batch_size
    )

    test_dataloader_params = {"num_workers": 0, "persistent_workers": False, "prefetch_factor": None}
    print("SPECIAL TEST PARAMS", test_dataloader_params)
    test_loader, fp_labels = gates_datasets.make_dual_dataloader(
        test_scaled_inputs, test_scaled_fp, test_bgs,
        batch_size=test_batch_size, randomize=False,
        dataloader_params=test_dataloader_params, flatten=True
    )
    return test_loader, fp_labels, test_scaled_fp


def resolve_dual_dynamic_edges(parameters, input_names):
    """Resolve the optional ``dynamic_edges`` block into model keyword arguments (dual model).

    The dual model's encoder supports WIND-AWARE mesh edges only: for every mesh edge the mean
    over its two endpoint mesh nodes of the input channels named in ``wind_tuples`` is appended
    to the edge attributes, per sample. The channel positions are resolved once, here, and stored
    in ``parameters["dynamic_edges_resolved"]`` so they are written to the training settings and
    every rebuild of the model (multi-GPU workers, ``predict_dual_model.py``) uses the same ones.

    Args:
        parameters (dict): Full parameter dict. Reads ``parameters["dynamic_edges"]``, e.g.
            ``{"dynamic_wind": true, "wind_tuples": [["x_wind", 3, 0], ["y_wind", 3, 0]]}``
            (the default tuples if omitted). Modified in place.
        input_names (iterable): ``(variable, level, time_delta)`` of the model's met/static input
            channels, in input order. Auxiliary boundary channels follow them in the model input
            and are never selected.

    Returns:
        dict: ``{}`` when no wind edges are requested, else
        ``{"wind_mesh_edges": True, "wind_indices": [...]}``.

    Raises:
        ValueError: for the unsupported ``dynamic_latlon`` / ``dynamic_earthdistance`` options,
            duplicated tuples, or tuples that are not among ``input_names``.
    """
    cfg = parameters.get("dynamic_edges") or {}
    if cfg.get("dynamic_latlon") or cfg.get("dynamic_earthdistance"):
        raise ValueError("dynamic_edges: dynamic_latlon / dynamic_earthdistance are not supported by the dual "
                         "model's encoder (only dynamic_wind is)")
    if not cfg.get("dynamic_wind", False):
        parameters.pop("dynamic_edges_resolved", None)
        return {}
    tuples = cfg.get("wind_tuples") or [["x_wind", 3, 0], ["y_wind", 3, 0]]
    wanted = [(str(v), int(level), int(delta)) for v, level, delta in tuples]
    if len(set(wanted)) != len(wanted):
        raise ValueError(f"dynamic_edges.wind_tuples contains duplicates: {wanted}")
    names = [(str(n[0]), int(n[1]), int(n[2])) for n in input_names]
    missing = [w for w in wanted if w not in names]
    if missing:
        raise ValueError(f"dynamic_edges.wind_tuples not found among the model inputs: {missing}")
    indices = [names.index(w) for w in wanted]
    parameters["dynamic_edges_resolved"] = {
        "wind_mesh_edges": True, "wind_indices": indices, "wind_channels": [list(w) for w in wanted]}
    print(f"Wind-aware mesh edges: {len(indices)} channels {wanted} at input positions {indices}")
    return {"wind_mesh_edges": True, "wind_indices": indices}


def dual_dynamic_edge_kwargs(parameters):
    """Model kwargs for the dynamic edges from ``parameters["dynamic_edges_resolved"]``.

    Fails loudly when wind edges are requested but were never resolved, so a trainer that does
    not call :func:`resolve_dual_dynamic_edges` cannot silently train without them.
    """
    resolved = parameters.get("dynamic_edges_resolved")
    requested = (parameters.get("dynamic_edges") or {}).get("dynamic_wind", False)
    if requested and not resolved:
        raise ValueError("dynamic_edges.dynamic_wind is set but parameters['dynamic_edges_resolved'] is missing: "
                         "call resolve_dual_dynamic_edges(parameters, input_names) before building the model")
    if not resolved:
        return {}
    return {"wind_mesh_edges": bool(resolved["wind_mesh_edges"]), "wind_indices": list(resolved["wind_indices"])}


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

    The background head converges (and starts overfitting) long before the footprint head,
    so it can optionally get its own training controls under ``parameters['bg_head']``
    (all keys optional; the defaults reproduce the previous behaviour exactly)::

        "bg_head": {
            "freeze_patience": 25,     # freeze the bg head after this many epochs without
                                       # improvement in its test loss (null = never freeze)
            "freeze_min_delta": 0.0,   # minimum decrease that counts as an improvement
            "weight_decay": 0.05,      # AdamW weight decay for bg_decoder params only
                                       # (null = optimizer default, as for the rest)
            "lr_scale": 1.0            # multiplier on learning_rate for bg_decoder params
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
        **dual_dynamic_edge_kwargs(parameters),
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

    # --- Optional background-head-specific training controls (see docstring) ---
    bg_head_cfg = parameters.get("bg_head", {})
    bg_freeze_patience = bg_head_cfg.get("freeze_patience", None)
    if bg_freeze_patience is not None:
        # patience < 1 would freeze the head immediately after the first epoch
        bg_freeze_patience = max(1, int(bg_freeze_patience))
    bg_freeze_min_delta = bg_head_cfg.get("freeze_min_delta", 0.0)
    bg_weight_decay = bg_head_cfg.get("weight_decay", None)
    bg_lr_scale = bg_head_cfg.get("lr_scale", 1.0)

    if bg_weight_decay is None and bg_lr_scale == 1.0:
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    else:
        # Separate parameter group so the bg head gets its own weight decay / LR while the
        # shared trunk and fp head keep the optimizer defaults.
        bg_decoder_params = list(model.bg_decoder.parameters())
        bg_param_ids = {id(p) for p in bg_decoder_params}
        shared_params = [p for p in model.parameters() if id(p) not in bg_param_ids]
        bg_group = {"params": bg_decoder_params, "lr": lr * bg_lr_scale}
        if bg_weight_decay is not None:
            bg_group["weight_decay"] = bg_weight_decay
        optimizer = optim.AdamW([{"params": shared_params}, bg_group], lr=lr)
        print(f"bg_decoder optimizer group: lr={lr * bg_lr_scale}, "
              f"weight_decay={bg_group.get('weight_decay', 'default')}")

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
        bg_freeze_patience=bg_freeze_patience,
        bg_freeze_min_delta=bg_freeze_min_delta,
    )

    return model, model_ctx


def freeze_bg_head(model):
    """Freeze the background decoder head: disable its grads so the optimizer skips it.

    The training loop must ALSO drop the bg term from the joint loss and put
    ``model.bg_decoder`` in eval mode during training (see ``train_one_epoch`` in
    ``train_dual_model.py``): gradients would otherwise still flow *through* the frozen
    head into the shared trunk, and its dropout/norm statistics would keep changing.
    """
    for p in model.bg_decoder.parameters():
        p.requires_grad_(False)


def initialise_dual_losses():
    """Loss tracking dict for the dual-head training run."""
    # Every metric group needs its OWN lists. (Until 2026-09-27 the groups were shallow copies of
    # one dict, so they shared the inner lists: the histories saved in the periodic checkpoints
    # interleave transformed/original values and the three flux patterns. Loss series, W&B logging
    # and the trained weights were never affected; scripts/mf_metric_curves.py reads both layouts.)
    def metrics_dict():
        return {"nmae": [], "mse": [], "bias": [], "mae": [], "iou": []}

    def flux_metrics_dict():
        return {"corrcoef": [], "mae": [], "mean_bias": [], "r2_score": []}

    losses = {
        "train": [],
        "test": [],
        "train_fp": [],
        "test_fp": [],
        "train_bg": [],
        "test_bg": [],
        "test_bg_mae_denorm": [],
        "metrics_transformed": metrics_dict(),
        "metrics_original": metrics_dict(),
        "metrics_fluxes_static": {
            "uniform": flux_metrics_dict(),
            "checkerboard": flux_metrics_dict(),
            "checkerboard_10": flux_metrics_dict(),
        },
    }
    return losses
