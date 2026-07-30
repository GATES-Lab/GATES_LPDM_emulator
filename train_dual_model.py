"""Train the dual-head GATES model (footprint + background) jointly.

This combines the two existing pipelines:
  - data is loaded with ``load_GATES_data_with_bg`` (footprints + met inputs + backgrounds
    + auxiliary CAMS, all aligned in time), exactly as in ``train_boundary_model.py``;
  - the model is a ``GraphSatelliteDualForecaster`` with both decoder heads;
  - each batch yields ``(inputs, fps, background)`` and the loss is the weighted sum of the
    per-node footprint loss and the whole-grid background loss.

Footprint-head metrics reuse the standard GATES evaluation (``calculate_losses`` + NetCDF
export); the background head additionally reports a denormalised MAE.

The two heads peak at different epochs, so alongside the combined-loss ``*_best.pt``
(EarlyStopping) each head's own best test loss saves a full-model snapshot
(``*_best_fp.pt`` / ``*_best_bg.pt``); the optional ``bg_head`` parameter block can
additionally freeze the bg head once its test loss stagnates and give it its own
weight decay / learning rate (see ``setup_dual_model`` in ``gates.training.training_dual``).
"""

import os
import sys
import copy
import time
import argparse
import random
from collections import namedtuple
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import wandb

'''
sys.path.insert(0, "/user/work/yl18410/new_graphnet")
sys.path.insert(1, "/user/work/yl18410/new_graphnet/graphnet_LPDM_emulator")
'''

import gates
import gates.training.training as gates_training
import gates.training.training_background as gates_training_background
import gates.training.training_dual as gates_training_dual
from gates.data.load_data import get_grid
from gates.training.training_background import format_aux_data, normalize_boundary_data, denormalize
from gates.training.training_dataclasses import PathContext, BoundaryTrainingContext
from gates.training.training_helperfuns import (
    load_parameter_file, save_object, write_to_file, save_training_plots,
    save_bg_timeseries_plots, export_results_to_netcdf, set_reproducibility,
    HeadCheckpoint, save_wandb_artifact,
)


# ---------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------

def _fp_true_values(fp_batch):
    """Extract the per-node target footprint (first variable) as shape (B, N, 1)."""
    if len(fp_batch.shape) == 3:
        return fp_batch[:, :, 0].unsqueeze(-1)
    return fp_batch.unsqueeze(-1)


def train_one_epoch(model, loader, model_ctx, epoch, paths_ctx=None):
    """One training epoch over (inputs, fps, background) batches with the joint loss.

    Once ``model_ctx.bg_frozen`` is set (see ``run_full_training``), the bg head is kept in
    eval mode, its loss is dropped from the joint loss (it is still computed on detached
    predictions for logging), and the trunk trains on the footprint objective alone.
    """
    model.train()
    if model_ctx.bg_frozen:
        # keep the frozen head's dropout / norm statistics fixed
        model.bg_decoder.eval()
    running_total = 0.0
    running_fp = 0.0
    running_bg = 0.0
    n_batches = 0

    for i, batch in enumerate(loader):
        features = batch[0].to(model_ctx.device)
        fp_batch = batch[1].to(model_ctx.device)
        bg_batch = batch[2].to(model_ctx.device)

        true_values = _fp_true_values(fp_batch)

        model_ctx.optimizer.zero_grad()
        outputs = model(features)
        # Nawid - predicts footprint and backgrpund
        fp_pred = outputs["footprint"]
        bg_pred = outputs["background"]

        fp_loss = model_ctx.fp_criterion(fp_pred, true_values, fp_batch)
        # Nawid - background loss
        if model_ctx.bg_frozen:
            # Frozen bg head: log its loss on detached predictions and drop it from the
            # joint loss — gradients would otherwise still reach the shared trunk through
            # the frozen head. The fp term keeps its (1 - w) scaling so the fp gradient
            # magnitude is unchanged by the freeze.
            bg_loss = model_ctx.bg_criterion(bg_pred.detach(), bg_batch)
            loss = (1 - model_ctx.bg_loss_weight) * fp_loss
        else:
            bg_loss = model_ctx.bg_criterion(bg_pred, bg_batch)
            # Nawid - Made it so that it uses both the different losses
            loss = (1-model_ctx.bg_loss_weight)*fp_loss + model_ctx.bg_loss_weight * bg_loss

        loss.backward()
        model_ctx.optimizer.step()

        with torch.no_grad():
            running_total += loss.item()
            running_fp += fp_loss.item()
            running_bg += bg_loss.item()
        n_batches += 1

        if i % 50 == 0:
            print(f"[{epoch}, {i:5d}] total: {running_total/(i+1):.4f} "
                  f"fp: {running_fp/(i+1):.4f} bg: {running_bg/(i+1):.4f}")
            if paths_ctx is not None:
                write_to_file(f"[{epoch}, {i:5d}] total {running_total/(i+1):.4f} "
                              f"fp {running_fp/(i+1):.4f} bg {running_bg/(i+1):.4f}", paths_ctx.updates_path)

    denom = max(n_batches, 1)
    return running_total / denom, running_fp / denom, running_bg / denom


@torch.no_grad()
def validate_and_predict(model, loader, model_ctx, output_norm=None):
    """Validate the joint model; returns losses plus stacked footprint and background predictions.

    Returns ``(avg_total, avg_fp, avg_bg, bg_mae_denorm, fp_preds, bg_preds, bg_trues)`` where
    ``fp_preds`` are the normalised footprint-head predictions stacked across batches (shape
    (N_samples, n_nodes)) and ``bg_preds``/``bg_trues`` are the normalised background-head
    predictions/targets stacked across batches (shape (N_samples, num_classes)). The test loader
    is not shuffled, so these stay aligned with the test time coordinate.
    """
    model.eval()
    total_err = 0.0
    fp_err = 0.0
    bg_err = 0.0
    bg_mae_denorm = 0.0
    fp_preds = []
    bg_preds = []
    bg_trues = []
    n_batches = 0

    for batch in loader:
        features = batch[0].to(model_ctx.device)
        fp_batch = batch[1].to(model_ctx.device)
        bg_batch = batch[2].to(model_ctx.device)

        true_values = _fp_true_values(fp_batch)

        outputs = model(features)
        fp_pred = outputs["footprint"]
        bg_pred = outputs["background"]

        fp_loss = model_ctx.fp_criterion_test(fp_pred, true_values, fp_batch)
        bg_loss = model_ctx.bg_criterion_test(bg_pred, bg_batch)
        # Match the training loss combination (see train_one_epoch).
        total_loss = (1 - model_ctx.bg_loss_weight) * fp_loss + model_ctx.bg_loss_weight * bg_loss

        total_err += total_loss.item()
        fp_err += fp_loss.item()
        bg_err += bg_loss.item()
        # Nawid - get denormalized loss
        if output_norm is not None:
            mean, std = output_norm
            diff = torch.abs(denormalize(bg_pred, mean, std) - denormalize(bg_batch, mean, std))
            bg_mae_denorm += float(diff.reshape(diff.shape[0], -1).mean().item())

        fp_preds.append(fp_pred.cpu().numpy().reshape(fp_pred.shape[0], -1))
        bg_preds.append(bg_pred.cpu().numpy().reshape(bg_pred.shape[0], -1))
        bg_trues.append(bg_batch.cpu().numpy().reshape(bg_batch.shape[0], -1))
        n_batches += 1

    denom = max(n_batches, 1)
    return (total_err / denom, fp_err / denom, bg_err / denom,
            bg_mae_denorm / denom if output_norm is not None else None,
            np.vstack(fp_preds), np.vstack(bg_preds), np.vstack(bg_trues))


def run_full_training(model, model_ctx, training_ctx, paths_ctx, train_loader, test_loader,
                      test_fp_dataset, losses, output_norm=None, epoch_so_far=0,
                      bg_detrended=True):
    """Joint training loop with footprint-head metrics and background-head MAE.

    Background time-series plots (denormalised, ppb) and their NetCDF export are produced
    for single-class (``num_classes=1``, summed) runs only; ``bg_detrended`` records whether
    the backgrounds were detrended (affects only plot labels). The plot windows are
    configurable via the ``bg_timeseries_plot`` parameter block (keys ``n_windows``,
    ``window_days``; defaults 4 and 7).
    """
    write_to_file("starting dual training loop", paths_ctx.updates_path)

    bg_true_ppb = bg_pred_ppb = None
    bg_plot_params = training_ctx.parameters.get("bg_timeseries_plot", {})
    bg_plot_n_windows = bg_plot_params.get("n_windows", 4)
    bg_plot_window_days = bg_plot_params.get("window_days", 7)

    # Best-so-far per-head test losses, logged every epoch. These are monotone, so the last
    # value equals the run's best — which makes "best/objective" a robust W&B sweep metric
    # (see run_dual_sweep_wandb.py). The per-head bests can occur at different epochs; the
    # objective therefore reflects per-head checkpoint selection, not a single checkpoint.
    objective_bg_weight = training_ctx.parameters.get("sweep", {}).get("objective_bg_weight", 1.0)
    best_test_fp = float("inf")
    best_test_bg = float("inf")

    # Per-head checkpointing: each head's own best test loss saves a full-model snapshot
    # (the heads share the trunk, so a "head checkpoint" is a whole model — use each file
    # for its own head's predictions). This realises the per-head selection that
    # "best/objective" assumes; the combined-loss *_best.pt from EarlyStopping is unchanged.
    verbose = training_ctx.parameters.get("verbose", True)
    fp_ckpt = HeadCheckpoint("fp", paths_ctx.model_path / f"{paths_ctx.model_name}_best_fp.pt",
                             verbose=verbose)
    bg_ckpt = HeadCheckpoint("bg", paths_ctx.model_path / f"{paths_ctx.model_name}_best_bg.pt",
                             delta=model_ctx.bg_freeze_min_delta, verbose=verbose)
    bg_frozen_epoch = None

    for epoch_idx in range(model_ctx.epochs_num):
        epoch = epoch_idx + epoch_so_far
        print(f"\n--- Start Epoch: {epoch} ---")
        # Nawid - get the training losses
        avg_train_total, avg_train_fp, avg_train_bg = train_one_epoch(
            model, train_loader, model_ctx, epoch, paths_ctx=paths_ctx
        )
        # Nawid- get the validation parameters
        avg_test_total, avg_test_fp, avg_test_bg, bg_mae_denorm, fp_test_out, bg_test_out, bg_test_true = validate_and_predict(
            model, test_loader, model_ctx, output_norm=output_norm
        )

        losses["train"].append(avg_train_total)
        losses["test"].append(avg_test_total)
        losses["train_fp"].append(avg_train_fp)
        losses["test_fp"].append(avg_test_fp)
        losses["train_bg"].append(avg_train_bg)
        losses["test_bg"].append(avg_test_bg)
        best_test_fp = min(best_test_fp, avg_test_fp)
        best_test_bg = min(best_test_bg, avg_test_bg)
        fp_ckpt.step(avg_test_fp, model, epoch)
        bg_ckpt.step(avg_test_bg, model, epoch)
        if bg_mae_denorm is not None:
            losses["test_bg_mae_denorm"].append(bg_mae_denorm)

        # --- Footprint-head evaluation (reuse the standard GATES metric pipeline) ---
        outputs_original_space = training_ctx.scalers["fp_scaler"].inverse_transform(fp_test_out)
        test_fp_dataset["fp_transformed_pred"] = (
            ("time", "lat", "lon"), fp_test_out.reshape(*test_fp_dataset.fp_original.shape))
        test_fp_dataset["fp_pred"] = (
            ("time", "lat", "lon"), outputs_original_space.reshape(*test_fp_dataset.fp_original.shape))

        losses, computed_metrics = gates_training.calculate_losses(losses, test_fp_dataset)

        list_of_metrics = {f"metrics_transformed-{k}": v for k, v in computed_metrics["transformed_eval_metrics"].items()}
        list_of_metrics.update({f"metrics_original-{k}": v for k, v in computed_metrics["eval_metrics"].items()})
        for flux_mode, metrics in computed_metrics["static_mf_eval_metrics"].items():
            list_of_metrics.update({f"metrics_fluxes_static/{flux_mode}/{k}": v for k, v in metrics.items()})

        if model_ctx.use_wandb:
            logging_dict = {
                "epoch": epoch + 1,
                "train/loss": avg_train_total,
                "test/loss": avg_test_total,
                "train/loss_fp": avg_train_fp,
                "test/loss_fp": avg_test_fp,
                "train/loss_bg": avg_train_bg,
                "test/loss_bg": avg_test_bg,
                "best/test_loss_fp": best_test_fp,
                "best/test_loss_bg": best_test_bg,
                "best/objective": best_test_fp + objective_bg_weight * best_test_bg,
                "best/epoch_fp": fp_ckpt.best_epoch,
                "best/epoch_bg": bg_ckpt.best_epoch,
                "bg/frozen": int(model_ctx.bg_frozen),
                **list_of_metrics,
            }
            if bg_mae_denorm is not None:
                logging_dict["test/bg_mae_denorm"] = bg_mae_denorm
            wandb.log(logging_dict, step=epoch)

        model_ctx.early_stopping(avg_test_total, model)
        if model_ctx.early_stopping.early_stop:
            print("Early stopping triggered. Ending training.")
            break

        # Freeze the bg head once its test loss has stagnated (it typically peaks around
        # epoch 50-90 and then overfits while the fp head keeps improving): from the next
        # epoch the trunk trains on the footprint objective alone. Opt-in via
        # parameters["bg_head"]["freeze_patience"]; its best checkpoint is already saved.
        if (model_ctx.bg_freeze_patience is not None and not model_ctx.bg_frozen
                and bg_ckpt.counter >= model_ctx.bg_freeze_patience):
            gates_training_dual.freeze_bg_head(model)
            model_ctx.bg_frozen = True
            bg_frozen_epoch = epoch
            freeze_msg = (f"Freezing bg head at epoch {epoch}: no improvement for "
                          f"{bg_ckpt.counter} epochs (best {bg_ckpt.best_loss:.6f} at "
                          f"epoch {bg_ckpt.best_epoch})")
            print(freeze_msg)
            write_to_file(freeze_msg, paths_ctx.updates_path)

        log_text = (f"Epoch {epoch}, total {avg_train_total:.4f}/{avg_test_total:.4f}, "
                    f"fp {avg_train_fp:.4f}/{avg_test_fp:.4f}, bg {avg_train_bg:.4f}/{avg_test_bg:.4f}")
        if bg_mae_denorm is not None:
            log_text += f", bg MAE(denorm) {bg_mae_denorm:.4e}"
        write_to_file(log_text, paths_ctx.updates_path)

        # Nawid - denormalise the background head outputs and convert mol/mol -> ppb so the
        # time-series plots and the NetCDF export are in physical units. Only done for the
        # single-class (summed) case.
        if output_norm is not None and bg_test_out.shape[1] == 1:
            mean, std = output_norm
            bg_true_ppb = denormalize(bg_test_true[:, 0], mean, std) * 1e9
            bg_pred_ppb = denormalize(bg_test_out[:, 0], mean, std) * 1e9

        if epoch % model_ctx.epochs_visualise == 0:
            img_save_path = save_training_plots(epoch, test_fp_dataset, training_ctx, paths_ctx.model_path, paths_ctx.model_name)
            if model_ctx.use_wandb:
                wandb.log({f"fps_epoch_{epoch}": wandb.Image(img_save_path)}, step=epoch)

            # Nawid - background time series (true vs predicted, denormalised, in ppb) over
            # week-long windows spread across the test period.
            if bg_true_ppb is not None:
                bg_img_path = save_bg_timeseries_plots(
                    epoch, test_fp_dataset.time.values, bg_true_ppb, bg_pred_ppb,
                    paths_ctx.model_path, paths_ctx.model_name, detrended=bg_detrended,
                    n_windows=bg_plot_n_windows, window_days=bg_plot_window_days,
                )
                if model_ctx.use_wandb:
                    wandb.log({f"bg_timeseries_epoch_{epoch}": wandb.Image(str(bg_img_path))}, step=epoch)

        if epoch % model_ctx.epochs_save == 0:
            checkpoint_path = paths_ctx.model_path / f"{paths_ctx.model_name}_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model_ctx.optimizer.state_dict(),
                'loss': losses,
                'learning_rate': model_ctx.lr,
            }, checkpoint_path)

    # Nawid - include the last epoch's background time series (denormalised, ppb) in the export
    # so the raw values behind the plots are saved alongside the footprint predictions.
    if bg_true_ppb is not None:
        test_fp_dataset["bg_true_ppb"] = (("time",), bg_true_ppb)
        test_fp_dataset["bg_pred_ppb"] = (("time",), bg_pred_ppb)
        test_fp_dataset["bg_true_ppb"].attrs["units"] = "ppb"
        test_fp_dataset["bg_pred_ppb"].attrs["units"] = "ppb"
        if bg_detrended:
            note = "detrended: relative to the south-boundary midpoint baseline"
            test_fp_dataset["bg_true_ppb"].attrs["note"] = note
            test_fp_dataset["bg_pred_ppb"].attrs["note"] = note

    # Record where each head peaked and upload the per-head best checkpoints once (per-epoch
    # artifact upload would spam versions; the combined best.pt already versions via
    # EarlyStopping).
    summary_msg = (f"Per-head bests: fp {fp_ckpt.best_loss:.6f} at epoch {fp_ckpt.best_epoch} "
                   f"-> {Path(fp_ckpt.path).name}; bg {bg_ckpt.best_loss:.6f} at epoch "
                   f"{bg_ckpt.best_epoch} -> {Path(bg_ckpt.path).name}")
    if bg_frozen_epoch is not None:
        summary_msg += f"; bg head frozen at epoch {bg_frozen_epoch}"
    print(summary_msg)
    write_to_file(summary_msg, paths_ctx.updates_path)

    if model_ctx.use_wandb:
        for head_ckpt in (fp_ckpt, bg_ckpt):
            if head_ckpt.best_epoch is not None:
                save_wandb_artifact(
                    model_ctx.model_name, f"best_{head_ckpt.head_name}", "model",
                    f"Best {head_ckpt.head_name}-head checkpoint (test {head_ckpt.head_name} "
                    f"loss {head_ckpt.best_loss:.6f} at epoch {head_ckpt.best_epoch})",
                    str(head_ckpt.path),
                )
        wandb.run.summary.update({
            "best/epoch_fp": fp_ckpt.best_epoch,
            "best/epoch_bg": bg_ckpt.best_epoch,
            "bg/frozen_epoch": bg_frozen_epoch,
        })

    export_results_to_netcdf(test_fp_dataset, paths_ctx.model_path, model_ctx.model_name, use_wandb=model_ctx.use_wandb)
    print("Finished Training.")


# ---------------------------------------------------------------
# Shared raw-data loading
# ---------------------------------------------------------------

# The raw (un-normalised, un-scaled) arrays produced by the expensive load step. Every
# per-experiment run derives its own class selection, normalisation, scalers and dataloaders
# from these, so a single bundle can be shared across experiments that use the same
# data-loading configuration (see run_dual_experiments_shared_data.py).
DualDataBundle = namedtuple("DualDataBundle", [
    "train_fp_data", "train_inputs", "train_bgs", "train_aux_cams_data",
    "test_fp_data", "test_inputs", "test_bgs", "test_aux_cams_data",
])


def resolve_background_params(parameters):
    """Merge ``background_setup`` onto the defaults and return the resolved dict.

    Kept as one place so both the loader and the trainer agree on detrend / auxiliary-BC
    settings (these must match for a shared data bundle to be valid).
    """
    default_bg_params = {"detrend": True, "use_auxiliary_bc": True, "auxilary_bc_levels": [4, 5, 6, 7]}
    background_params = default_bg_params.copy()
    background_params.update(parameters.get("background_setup", {}))
    return background_params


def load_dual_data(parameters, verbose=True, return_summary=False):
    """Load the raw footprint/met/background data for a dual-head run.

    This is the expensive step (``load_GATES_data_with_bg`` for both train and test). It is
    factored out of :func:`train_and_save_model` so a driver can load the data once and reuse it
    across many experiments that share the same data-loading configuration — i.e. the same
    ``train_load_data``, ``test_load_data``, ``variables`` and ``background_setup`` (see
    ``run_dual_experiments_shared_data.py``).

    Returns a :class:`DualDataBundle` of the raw arrays. Downstream steps (class selection,
    normalisation, scalers, dataloaders) are *not* done here, so each experiment still builds its
    own transforms from these shared arrays.

    With ``return_summary=True``, returns ``(bundle, summary)`` where ``summary`` is a dict of
    loading statistics (per-month/total wall time in minutes, sample counts) suitable for
    :func:`log_data_loading_summary_run`.
    """
    train_load_data_params = copy.deepcopy(parameters["train_load_data"])
    test_load_data_params = copy.deepcopy(parameters["train_load_data"])
    test_load_data_params.update(parameters["test_load_data"])
    input_variables = parameters["variables"]
    # resolve_datapath_args only reads `parameters`; a throwaway PathContext creates no dirs.
    datapath_args = PathContext(
        model_save_dir=".", model_name="_dataload", model_path=Path(".")
    ).resolve_datapath_args(parameters)

    background_params = resolve_background_params(parameters)

    print("Loading shared met, fp AND BACKGROUND data (train + test)...")
    client, cluster = gates_training.make_cluster()

    train_month_mins = {}
    load_start = time.perf_counter()
    train_fp_data, train_inputs, train_bgs, train_aux_cams_data = gates_training_background.load_GATES_data_with_bg(
        train_load_data_params, input_variables=input_variables, datapath_args=datapath_args, verbose=verbose,
        load_into_memory=parameters.get("load_into_memory", False), detrend=background_params["detrend"],
        use_aux_bc=background_params["use_auxiliary_bc"], aux_indeces=background_params["auxilary_bc_levels"],
        loading_times_out=train_month_mins,
    )
    train_load_mins = (time.perf_counter() - load_start) / 60
    print("Successfully loaded training data with", len(train_fp_data.time), "time samples")

    test_month_mins = {}
    load_start = time.perf_counter()
    test_fp_data, test_inputs, test_bgs, test_aux_cams_data = gates_training_background.load_GATES_data_with_bg(
        test_load_data_params, input_variables=input_variables, datapath_args=datapath_args, verbose=verbose,
        load_into_memory=parameters.get("load_into_memory", False), detrend=background_params["detrend"],
        use_aux_bc=background_params["use_auxiliary_bc"], aux_indeces=background_params["auxilary_bc_levels"],
        loading_times_out=test_month_mins,
    )
    test_load_mins = (time.perf_counter() - load_start) / 60
    print("Successfully loaded test data with", len(test_fp_data.time), "time samples")

    if cluster is not None:
        cluster.close()
        client.close()

    bundle = DualDataBundle(
        train_fp_data, train_inputs, train_bgs, train_aux_cams_data,
        test_fp_data, test_inputs, test_bgs, test_aux_cams_data,
    )
    if not return_summary:
        return bundle

    summary = {
        "train_load_mins": train_load_mins,
        "test_load_mins": test_load_mins,
        "total_load_mins": train_load_mins + test_load_mins,
        "n_train_samples": int(len(train_fp_data.time)),
        "n_test_samples": int(len(test_fp_data.time)),
        "train_month_mins": train_month_mins,
        "test_month_mins": test_month_mins,
    }
    return bundle, summary


def log_data_loading_summary_run(parameters, summary, run_suffix="dataload"):
    """Log the shared data-loading summary as its OWN (immediately finished) W&B run.

    Used by the shared-data drivers (run_dual_experiments_shared_data.py,
    run_dual_sweep_wandb.py): the data is loaded once per process, so the loading summary
    belongs to no single training run. This records it in a separate run (job_type
    "data_loading", same group as the training runs) and calls ``wandb.finish()`` before
    returning, so the subsequent training runs are unaffected (train_and_save_model /
    wandb.agent each start from no active run, exactly as before).
    """
    wandb_cfg = parameters.get("wandb", {})
    entity, project = wandb_cfg.get("entity"), wandb_cfg.get("project")
    if entity is None or project is None:
        print("Warning: no wandb.entity/project configured — skipping the data-loading summary run.")
        return
    if wandb.run is not None:
        # Never hijack an existing run (would break the reuse logic in train_and_save_model).
        print("Warning: a W&B run is already active — skipping the separate data-loading summary run.")
        return

    job_id = os.environ.get("SLURM_JOB_ID", "local")
    job_name = os.environ.get("SLURM_JOB_NAME", "run")
    run = wandb.init(
        entity=entity, project=project, group=wandb_cfg.get("group", None),
        tags=list(wandb_cfg.get("tags", [])) + ["data_loading"],
        name=f"{job_id}_{job_name}_{run_suffix}", job_type="data_loading",
        config={
            "train_load_data": parameters.get("train_load_data"),
            "test_load_data": parameters.get("test_load_data"),
            "variables": parameters.get("variables"),
            "background_setup": resolve_background_params(parameters),
            "load_into_memory": parameters.get("load_into_memory", False),
        },
    )
    run_name = run.name
    try:
        table = wandb.Table(columns=["split", "month", "load_mins"])
        for split in ("train", "test"):
            for month_key, mins in summary.get(f"{split}_month_mins", {}).items():
                table.add_data(split, month_key, mins)
        wandb.log({"data_loading/month_load_mins": table})
        run.summary.update({k: v for k, v in summary.items() if not isinstance(v, dict)})
    finally:
        wandb.finish()
    print(f"Logged data-loading summary to its own W&B run: {run_name}")


# ---------------------------------------------------------------
# Top-level training entry point
# ---------------------------------------------------------------

def train_and_save_model(parameters, model_save_dir, wandb_name=None, data_bundle=None):
    """Top-level entry point for a full dual-head training run.

    ``wandb_name`` optionally sets the W&B run name (e.g. the experiment name
    when driven by run_dual_experiments.py). If omitted, a name is built from
    the SLURM job info and this run's bg_loss_weight.

    ``data_bundle`` optionally supplies a pre-loaded :class:`DualDataBundle` (from
    :func:`load_dual_data`) so several experiments can reuse one expensive load. When omitted the
    data is loaded here as before. The caller is responsible for ensuring the bundle was loaded
    with a matching data-loading configuration.
    """
    use_wandb = parameters.get('use_wandb', True)
    cfg = gates.config.get_config()
    verbose = parameters.get("verbose", True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{parameters['model_name']}_{timestamp}"
    model_path = Path(model_save_dir) / model_name
    print(f"Initialising dual model run for model_name: {model_name}")

    paths_ctx = PathContext(model_save_dir=model_save_dir, model_name=model_name, model_path=model_path)
    paths_ctx.make_dirs()

    set_reproducibility(parameters.get("seed", 34))

    if use_wandb and wandb.run is not None:
        # A W&B run is already active — e.g. created by wandb.agent during a sweep
        # (run_dual_sweep_wandb.py). Reuse it instead of starting a second run, and
        # record the fully-merged parameters on it (sweep-chosen values keep their
        # sweep-assigned entries, hence allow_val_change).
        wandb.config.update(parameters, allow_val_change=True)
    elif use_wandb:
        wandb_project = parameters.get("wandb", {}).get("project", None)
        wandb_entity = parameters.get("wandb", {}).get("entity", None)
        wandb_tags = parameters.get("wandb", {}).get("tags", [])
        wandb_group = parameters.get("wandb", {}).get("group", None)
        if wandb_project is None or wandb_entity is None:
            print("Warning: use_wandb is True but no wandb.project/entity specified. W&B disabled.")
            use_wandb = False
            parameters["use_wandb"] = False
        if use_wandb:
            # Build a descriptive run name from the SLURM job info. For an
            # experiment run the caller supplies wandb_name (the experiment
            # name); otherwise fall back to this run's bg_loss_weight.
            job_id = os.environ.get("SLURM_JOB_ID", "local")
            job_name = os.environ.get("SLURM_JOB_NAME", "run")
            if wandb_name:
                suffix = wandb_name
            else:
                bg_loss_weight = parameters.get("loss_functions", {}).get("bg_loss_weight")
                suffix = f"bg{bg_loss_weight}"
            run_name = f"{job_id}_{job_name}_{suffix}"
            wandb.init(entity=wandb_entity, project=wandb_project, config=parameters, tags=wandb_tags, name=run_name, group=wandb_group)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_to_file(f"using device {device}, starting at " + datetime.now().strftime("%d/%m/%y %H:%M:%S"), paths_ctx.updates_path)

    # Nawid - get background information (must match what load_dual_data used)
    background_params = resolve_background_params(parameters)
    parameters["background_setup"] = background_params

    num_classes = parameters["model_parameters"].get("num_classes", 1)
    if num_classes not in [1, 4]:
        print("Warning: num_classes not in [1, 4]. Defaulting to 1 (summed output).")
        num_classes = 1
        parameters["model_parameters"]["num_classes"] = 1

    # Load the raw data, or reuse a bundle already loaded once for a set of experiments.
    if data_bundle is None:
        print("Loading met, fp AND BACKGROUND data for model", model_name)
        data_bundle = load_dual_data(parameters, verbose=verbose)
    else:
        print("Reusing pre-loaded shared data bundle for model", model_name)
    (train_fp_data, train_inputs, train_bgs, train_aux_cams_data,
     test_fp_data, test_inputs, test_bgs, test_aux_cams_data) = data_bundle

    # Select the background output variables (DataArray with a num_classes dim)
    if num_classes == 1:
        train_bgs = train_bgs[["summed"]].to_dataarray()
        test_bgs = test_bgs[["summed"]].to_dataarray()
        class_names = ["summed"]
    else:
        train_bgs = train_bgs[["north", "south", "east", "west"]].to_dataarray()
        test_bgs = test_bgs[["north", "south", "east", "west"]].to_dataarray()
        class_names = ["north", "south", "east", "west"]

    # to_dataarray adds a leading 'variable' dim; move it to the end as the class dim (time, num_classes)
    train_bgs = train_bgs.transpose("time", "variable")
    test_bgs = test_bgs.transpose("time", "variable")

    use_auxiliary_bc = background_params["use_auxiliary_bc"]
    if use_auxiliary_bc:
        train_aux_cams_data = format_aux_data(train_aux_cams_data, time_coord=train_fp_data.time)
        test_aux_cams_data = format_aux_data(test_aux_cams_data, time_coord=test_fp_data.time)

    # Nawid - normalize the data
    norm_train_bgs, norm_train_aux_data, norm_vals = normalize_boundary_data(train_bgs, aux_data=train_aux_cams_data)
    norm_test_bgs, norm_test_aux_data, norm_vals = normalize_boundary_data(test_bgs, aux_data=test_aux_cams_data, norm_vals=norm_vals)

    save_object(norm_vals, "norm_vals", paths_ctx.training_outputs_path, model_name,
                file_type="json", description=f"Background normalisation values for model {model_name}", use_wandb=use_wandb)

    num_features = train_inputs.shape[-1]
    feature_dim = num_features
    if use_auxiliary_bc:
        aux_dim = train_aux_cams_data.aux.shape[0]
        num_features = num_features + aux_dim
    else:
        aux_dim = 0
    parameters["num_features"] = num_features
    print("using num_features =", num_features, "out of which aux_dim =", aux_dim)

    train_loader, test_loader, fp_labels, test_scaled_fp, scalers = gates_training_dual.setup_dual_dataloaders(
        parameters, train_inputs, train_fp_data, norm_train_bgs,
        test_inputs, test_fp_data, norm_test_bgs,
        norm_train_aux_data, norm_test_aux_data,
    )

    save_object(scalers, "scalers", paths_ctx.training_outputs_path, model_name,
                description=f"Input and footprint scaler objects used in model {model_name}", use_wandb=use_wandb)

    image_plots = random.sample(list(range(len(test_inputs))), k=4)
    image_dates = np.datetime_as_string(test_fp_data.time.values[sorted(image_plots)])
    parameters["plotted_dates"] = image_dates.tolist()

    save_object(parameters, "training_settings", paths_ctx.training_outputs_path, model_name,
                file_type="json", description=f"Training settings for model {model_name}", use_wandb=use_wandb)

    grid, _ = get_grid(train_fp_data, parameters.get("grid_reference_fp"))
    save_object(grid, "grid", paths_ctx.training_outputs_path, model_name,
                description="Grid object used during training", use_wandb=use_wandb)

    training_ctx = BoundaryTrainingContext(
        parameters, device, use_wandb, image_dates, image_plots, grid,
        fp_labels, scalers, feature_dim, len(train_fp_data.lat.values), aux_dim,
    )

    print("Successfully set up dual dataloaders!")
    model, model_ctx = gates_training_dual.setup_dual_model(parameters, training_ctx, paths_ctx)

    if use_wandb:
        wandb.watch(model, log="all", log_freq=100)
#
    losses = gates_training_dual.initialise_dual_losses()

    if torch.cuda.is_available():
        model.cuda()

    run_full_training(
        model, model_ctx, training_ctx, paths_ctx, train_loader, test_loader,
        test_scaled_fp, losses, output_norm=norm_vals["outputs"], epoch_so_far=0,
        bg_detrended=background_params["detrend"],
    )

    if use_wandb:
        wandb.finish()


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the dual-head model. Example: python train_dual_model.py parameters.json")
    parser.add_argument("file_name", help="Parameter file name")
    parser.add_argument("--file_path", help="Parameter file path. Defaults to config.yml's parameter_files_dir.", default=None)

    args = parser.parse_args()

    cfg = gates.config.get_config()
    file_path = args.file_path if args.file_path is not None else cfg.parameter_files_dir
    parameter_path = Path(file_path) / args.file_name

    parameters = load_parameter_file(parameter_path)
    if parameters is None:
        print("Error loading parameters. Exiting.")
        sys.exit(1)

    print("PARAMETERS:")
    print(parameters)

    if parameters.get("use_wandb", False):
        wandb.login()

    if parameters.get("model_save_dir", None) is None:
        model_saving_dir = cfg.save_models_dir
    else:
        model_saving_dir = Path(parameters["model_save_dir"])

    train_and_save_model(parameters, model_save_dir=model_saving_dir)
