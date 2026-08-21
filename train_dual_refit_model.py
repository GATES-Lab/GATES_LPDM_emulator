"""Train the dual-head GATES model with a bg-head freeze / refit schedule.

This is a sibling of ``train_dual_model.py`` (which is unchanged and remains the standard
entry point). It adds a phased schedule for the background head, configured via a
``bg_schedule`` parameter block, to test whether the shared trunk still encodes the
background after continued footprint training:

  Phase 1 "joint"   epochs [0, freeze_epoch):        normal joint training, as before.
  Phase 2 "frozen"  epochs [freeze_epoch, refit_start_epoch):  the bg head is frozen and its
                    loss dropped from the joint loss (identical to the existing
                    ``bg_head.freeze_patience`` behaviour, but at a FIXED epoch so parallel
                    experiment arms share the same trajectory up to the refit).
  Phase 3 "refit"   epochs [refit_start_epoch, end):  the bg head is unfrozen and re-fitted
                    with a fresh AdamW optimizer, in one of three modes:
                      - "bg_only": encoder + processor + fp decoder are frozen; only the bg
                        decoder trains, against the bg criterion alone. The fp head cannot
                        move, so any bg recovery is attributable to head re-alignment on
                        fixed trunk features.
                      - "bg_trunk": the fp decoder alone is frozen; the trunk AND the bg
                        decoder train against the bg criterion alone (the fp loss is
                        dropped, mirroring how the bg freeze drops the bg loss). The fp
                        test loss then measures how much bg-directed trunk movement costs
                        the fixed fp head.
                      - "joint":  full joint training resumes ((1-w)*fp + w*bg); the trunk
                        receives bg gradients again, so the fp test loss shows whether
                        re-fitting the bg head makes the fp head drift.

At the refit transition an optional linear probe first fits a scalar gain/offset a*y+b on
the frozen head's TRAIN-set predictions and reports the calibrated TEST bg loss — the
cheap "does amplitude calibration alone recover the gap?" check.

The ``bg_schedule`` block (all keys optional; with no block this script reduces to
``train_dual_model.py`` behaviour)::

    "bg_schedule": {
        "freeze_epoch": 75,          # freeze the bg head at the START of this epoch
                                     # (null = fall back to bg_head.freeze_patience)
        "refit_start_epoch": 120,    # unfreeze + refit from the start of this epoch
                                     # (null = never refit)
        "refit_mode": "bg_only",     # "bg_only", "bg_trunk" or "joint" (see above)
        "refit_lr_scale": 1.0,       # bg-decoder LR during refit = learning_rate * this
        "linear_probe": true         # fit a*y+b (train) and log calibrated test loss
    }

The refit always starts the bg decoder's optimizer from scratch (fresh AdamW state); in
"joint" mode the trunk/fp parameters keep their original optimizer and its state, so the
trunk's dynamics stay comparable to a no-refit run.

Setting ``"freeze_epoch": 0`` turns the schedule into FP-FIRST pretraining: the bg head
never trains (it stays at its random initialisation) until ``refit_start_epoch``, so with
``refit_mode: "joint"`` the run is "train fp alone, then bring the bg head online and
train both jointly" (see ``experiments_dual_fp_first.json``).

Extra checkpoints/metrics on top of the usual ones: ``*_best_bg_refit.pt`` (best bg test
loss WITHIN the refit phase, even if it never beats the pre-freeze best) and, under
``refit/`` in W&B: the refit-phase bg best, the fp drift since the refit started, and the
linear-probe results.

Data loading, validation and the footprint metric pipeline are imported unchanged from
``train_dual_model``; the shared-data driver runs this script's ``train_and_save_model``
via ``run_dual_experiments_shared_data.py --trainer train_dual_refit_model``.
"""

import os
import sys
import argparse
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import wandb

import gates
import gates.training.training as gates_training
import gates.training.training_dual as gates_training_dual
import gates.training.dual_analysis as dual_analysis
from gates.data.load_data import get_grid
from gates.training.training_background import format_aux_data, normalize_boundary_data, denormalize
from gates.training.training_dataclasses import PathContext, BoundaryTrainingContext
from gates.training.training_helperfuns import (
    load_parameter_file, save_object, write_to_file, save_training_plots,
    save_bg_timeseries_plots, export_results_to_netcdf, set_reproducibility,
    HeadCheckpoint, save_wandb_artifact,
)

# Reused unchanged from the standard dual trainer (kept there so both scripts share one
# implementation; the shared-data driver also imports load_dual_data from this module).
from train_dual_model import (
    _fp_true_values, validate_and_predict, load_dual_data, resolve_background_params,
    log_data_loading_summary_run, DualDataBundle,
)


# ---------------------------------------------------------------
# Background-head schedule (freeze -> refit)
# ---------------------------------------------------------------

PHASE_CODES = {"joint": 0, "frozen": 1, "refit_bg_only": 2, "refit_joint": 3,
               "refit_bg_trunk": 4}


class BgSchedule:
    """Configuration + runtime state of the bg-head freeze/refit schedule.

    Parsed from ``parameters["bg_schedule"]`` (see module docstring). ``phase`` is one of
    ``PHASE_CODES``; the training loop reads it every epoch and this class flips it at the
    configured epochs (``maybe_freeze`` / ``maybe_start_refit``).
    """

    def __init__(self, parameters):
        cfg = parameters.get("bg_schedule", {})
        self.freeze_epoch = cfg.get("freeze_epoch", None)
        self.refit_start_epoch = cfg.get("refit_start_epoch", None)
        self.refit_mode = cfg.get("refit_mode", "bg_only")
        self.refit_lr_scale = cfg.get("refit_lr_scale", 1.0)
        self.linear_probe = cfg.get("linear_probe", True)

        if self.refit_mode not in ("bg_only", "bg_trunk", "joint"):
            raise ValueError("bg_schedule.refit_mode must be 'bg_only', 'bg_trunk' or "
                             f"'joint', got {self.refit_mode!r}")
        if (self.freeze_epoch is not None and self.refit_start_epoch is not None
                and self.refit_start_epoch <= self.freeze_epoch):
            raise ValueError("bg_schedule.refit_start_epoch must be > freeze_epoch "
                             f"(got {self.refit_start_epoch} <= {self.freeze_epoch})")

        # runtime state
        self.phase = "joint"
        self.bg_frozen_epoch = None
        self.refit_started_epoch = None
        self.bg_optimizer = None          # fresh AdamW over bg_decoder params, refit only
        self.refit_ckpt = None            # HeadCheckpoint for the best bg loss WITHIN the refit
        self.pre_refit_bg_best = None     # bg test-loss best at the moment the refit starts
        self.fp_test_at_refit_start = None
        self.probe_metrics = {}

    @property
    def bg_head_active(self):
        """Whether the bg head is currently being trained (its loss reaches its params)."""
        return self.phase in ("joint", "refit_bg_only", "refit_joint")

    def freeze(self, model, model_ctx, epoch, paths_ctx, reason):
        gates_training_dual.freeze_bg_head(model)
        model_ctx.bg_frozen = True
        self.phase = "frozen"
        self.bg_frozen_epoch = epoch
        msg = f"Freezing bg head at epoch {epoch} ({reason})"
        print(msg)
        write_to_file(msg, paths_ctx.updates_path)

    def maybe_freeze(self, model, model_ctx, epoch, paths_ctx):
        """Fixed-epoch freeze, applied at the START of ``freeze_epoch``."""
        if self.phase == "joint" and self.freeze_epoch is not None and epoch >= self.freeze_epoch:
            self.freeze(model, model_ctx, epoch, paths_ctx, f"fixed bg_schedule.freeze_epoch={self.freeze_epoch}")

    def maybe_start_refit(self, model, model_ctx, epoch, paths_ctx, bg_ckpt, losses,
                          train_loader, test_loader):
        if self.refit_start_epoch is None or epoch < self.refit_start_epoch:
            return
        if self.phase not in ("joint", "frozen"):
            return  # already refitting
        if self.phase == "joint":
            msg = (f"Warning: refit starting at epoch {epoch} but the bg head was never frozen "
                   "(no freeze_epoch hit and no patience freeze) — refitting from a live head.")
            print(msg)
            write_to_file(msg, paths_ctx.updates_path)

        # Cheap calibration check BEFORE any refit training: fit a*y+b on the train set and
        # report the calibrated test bg loss of the otherwise-unchanged head.
        if self.linear_probe:
            self.probe_metrics = run_linear_probe(model, model_ctx, train_loader, test_loader)
            probe_msg = ("Linear probe (a*y+b fitted on train): "
                         f"a={self.probe_metrics['refit/linear_probe_a']:.4f}, "
                         f"b={self.probe_metrics['refit/linear_probe_b']:.4f}, "
                         f"calibrated test bg loss {self.probe_metrics['refit/linear_probe_test_bg_loss']:.6f} "
                         f"(uncalibrated {self.probe_metrics['refit/linear_probe_test_bg_loss_uncalibrated']:.6f})")
            print(probe_msg)
            write_to_file(probe_msg, paths_ctx.updates_path)

        # Unfreeze the bg decoder and give it a FRESH optimizer (a refit starts from clean
        # AdamW state; the stale pre-freeze moments are ~45 epochs old).
        for p in model.bg_decoder.parameters():
            p.requires_grad_(True)

        # Remove the bg-decoder params from the main optimizer so the two optimizers never
        # both step the same tensors, and drop their stale state.
        bg_param_ids = {id(p) for p in model.bg_decoder.parameters()}
        for group in model_ctx.optimizer.param_groups:
            group["params"] = [p for p in group["params"] if id(p) not in bg_param_ids]
        for p in model.bg_decoder.parameters():
            model_ctx.optimizer.state.pop(p, None)
        self.bg_optimizer = optim.AdamW(model.bg_decoder.parameters(),
                                        lr=model_ctx.lr * self.refit_lr_scale)

        if self.refit_mode == "bg_only":
            # Freeze the whole trunk + fp head: the fp predictions are constant from here
            # on, so any bg improvement is head re-alignment on fixed features.
            for module in (model.encoder, model.processor, model.fp_decoder):
                for p in module.parameters():
                    p.requires_grad_(False)
            self.phase = "refit_bg_only"
        elif self.refit_mode == "bg_trunk":
            # Freeze only the fp head: the trunk re-adapts to the bg objective and the
            # fixed fp head measures what that movement costs the footprint.
            for p in model.fp_decoder.parameters():
                p.requires_grad_(False)
            self.phase = "refit_bg_trunk"
        else:
            self.phase = "refit_joint"

        model_ctx.bg_frozen = False
        self.refit_started_epoch = epoch
        self.pre_refit_bg_best = bg_ckpt.best_loss
        self.fp_test_at_refit_start = losses["test_fp"][-1] if losses["test_fp"] else None
        self.refit_ckpt = HeadCheckpoint(
            "bg_refit", paths_ctx.model_path / f"{paths_ctx.model_name}_best_bg_refit.pt",
            verbose=True)

        msg = (f"Starting bg refit at epoch {epoch} (mode={self.refit_mode}, "
               f"lr={model_ctx.lr * self.refit_lr_scale:g}, fresh bg optimizer): "
               f"pre-refit bg best {self.pre_refit_bg_best:.6f} at epoch {bg_ckpt.best_epoch}, "
               f"fp test loss at refit start {self.fp_test_at_refit_start}")
        print(msg)
        write_to_file(msg, paths_ctx.updates_path)


@torch.no_grad()
def run_linear_probe(model, model_ctx, train_loader, test_loader):
    """Fit a scalar gain/offset a*y+b on the bg head's TRAIN predictions, evaluate on TEST.

    The fit uses only training data (no test leakage); the reported metric is the test-set
    bg criterion applied to the calibrated predictions ``a*y_hat + b``, alongside the
    uncalibrated test loss for reference. All in normalised bg space, num_classes pooled.
    """
    model.eval()

    def collect(loader):
        preds, trues = [], []
        for batch in loader:
            features = batch[0].to(model_ctx.device)
            bg_batch = batch[2].to(model_ctx.device)
            bg_pred = model(features)["background"]
            preds.append(bg_pred.reshape(-1).cpu())
            trues.append(bg_batch.reshape(-1).cpu())
        return torch.cat(preds), torch.cat(trues)

    train_pred, train_true = collect(train_loader)
    var = torch.var(train_pred, unbiased=False)
    if var.item() == 0.0:
        a, b = 1.0, 0.0  # degenerate constant predictor; calibration undefined
    else:
        cov = torch.mean((train_pred - train_pred.mean()) * (train_true - train_true.mean()))
        a = (cov / var).item()
        b = (train_true.mean() - a * train_pred.mean()).item()

    test_pred, test_true = collect(test_loader)
    calibrated_loss = model_ctx.bg_criterion_test(a * test_pred + b, test_true).item()
    uncalibrated_loss = model_ctx.bg_criterion_test(test_pred, test_true).item()

    return {
        "refit/linear_probe_a": a,
        "refit/linear_probe_b": b,
        "refit/linear_probe_test_bg_loss": calibrated_loss,
        "refit/linear_probe_test_bg_loss_uncalibrated": uncalibrated_loss,
    }


# ---------------------------------------------------------------
# Training loop (phase-aware variant of train_dual_model.train_one_epoch)
# ---------------------------------------------------------------

def train_one_epoch(model, loader, model_ctx, sched, epoch, paths_ctx=None, analyser=None):
    """One training epoch; the loss composition and which optimizers step depend on
    ``sched.phase`` (see module docstring). Both per-head losses are always logged.
    """
    phase = sched.phase
    model.train()
    if phase == "frozen":
        # keep the frozen head's dropout / norm statistics fixed
        model.bg_decoder.eval()
    elif phase == "refit_bg_only":
        # trunk + fp head are frozen: keep their dropout / norm statistics fixed too
        model.encoder.eval()
        model.processor.eval()
        model.fp_decoder.eval()
    elif phase == "refit_bg_trunk":
        # only the fp head is frozen; the trunk keeps training (on the bg objective)
        model.fp_decoder.eval()

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
        if sched.bg_optimizer is not None:
            sched.bg_optimizer.zero_grad()
        outputs = model(features)
        fp_pred = outputs["footprint"]
        bg_pred = outputs["background"]

        if phase in ("joint", "refit_joint"):
            fp_loss = model_ctx.fp_criterion(fp_pred, true_values, fp_batch)
            bg_loss = model_ctx.bg_criterion(bg_pred, bg_batch)
            loss = (1 - model_ctx.bg_loss_weight) * fp_loss + model_ctx.bg_loss_weight * bg_loss
        elif phase == "frozen":
            # As in train_dual_model: bg loss logged on detached predictions and dropped
            # from the joint loss; the fp term keeps its (1 - w) scaling.
            fp_loss = model_ctx.fp_criterion(fp_pred, true_values, fp_batch)
            bg_loss = model_ctx.bg_criterion(bg_pred.detach(), bg_batch)
            loss = (1 - model_ctx.bg_loss_weight) * fp_loss
        elif phase in ("refit_bg_only", "refit_bg_trunk"):
            # The fp head is frozen and its loss dropped (mirroring the bg freeze); only
            # the bg criterion trains, unweighted — the weight only rescales the single
            # term. In "refit_bg_trunk" the bg gradients also reach the (live) trunk;
            # fp_loss on detached predictions is logging only in both modes.
            fp_loss = model_ctx.fp_criterion(fp_pred.detach(), true_values, fp_batch)
            bg_loss = model_ctx.bg_criterion(bg_pred, bg_batch)
            loss = bg_loss
        else:
            raise RuntimeError(f"unknown schedule phase {phase!r}")

        # ANALYSIS ONLY (gates/training/dual_analysis.py), as in train_dual_model.
        if analyser is not None and analyser.should_measure(i):
            analyser.measure(fp_loss, bg_loss, model_ctx.bg_loss_weight,
                             bg_frozen=(phase == "frozen"))

        loss.backward()
        if phase != "refit_bg_only":
            # in refit_bg_only every main-optimizer param is frozen; skip the no-op step
            # (in refit_bg_trunk this step updates the trunk from the bg gradients)
            model_ctx.optimizer.step()
        if sched.bg_optimizer is not None and phase in ("refit_bg_only", "refit_bg_trunk", "refit_joint"):
            sched.bg_optimizer.step()

        with torch.no_grad():
            running_total += loss.item()
            running_fp += fp_loss.item()
            running_bg += bg_loss.item()
        n_batches += 1

        if i % 50 == 0:
            print(f"[{epoch}, {i:5d}] ({phase}) total: {running_total/(i+1):.4f} "
                  f"fp: {running_fp/(i+1):.4f} bg: {running_bg/(i+1):.4f}")
            if paths_ctx is not None:
                write_to_file(f"[{epoch}, {i:5d}] ({phase}) total {running_total/(i+1):.4f} "
                              f"fp {running_fp/(i+1):.4f} bg {running_bg/(i+1):.4f}", paths_ctx.updates_path)

    denom = max(n_batches, 1)
    return running_total / denom, running_fp / denom, running_bg / denom


def run_full_training(model, model_ctx, training_ctx, paths_ctx, train_loader, test_loader,
                      test_fp_dataset, losses, output_norm=None, epoch_so_far=0,
                      bg_detrended=True):
    """Phase-aware joint training loop; mirrors train_dual_model.run_full_training with the
    bg freeze/refit schedule applied at the start of each epoch and refit metrics logged
    under the ``refit/`` prefix.
    """
    write_to_file("starting dual refit training loop", paths_ctx.updates_path)

    sched = BgSchedule(training_ctx.parameters)
    if (sched.refit_start_epoch is not None
            and sched.refit_start_epoch >= epoch_so_far + model_ctx.epochs_num):
        print(f"Warning: bg_schedule.refit_start_epoch={sched.refit_start_epoch} is beyond the "
              f"last training epoch ({epoch_so_far + model_ctx.epochs_num - 1}) — no refit will run.")

    bg_true_ppb = bg_pred_ppb = None
    bg_plot_params = training_ctx.parameters.get("bg_timeseries_plot", {})
    bg_plot_n_windows = bg_plot_params.get("n_windows", 4)

    objective_bg_weight = training_ctx.parameters.get("sweep", {}).get("objective_bg_weight", 1.0)
    best_test_fp = float("inf")
    best_test_bg = float("inf")

    verbose = training_ctx.parameters.get("verbose", True)
    fp_ckpt = HeadCheckpoint("fp", paths_ctx.model_path / f"{paths_ctx.model_name}_best_fp.pt",
                             verbose=verbose)
    bg_ckpt = HeadCheckpoint("bg", paths_ctx.model_path / f"{paths_ctx.model_name}_best_bg.pt",
                             delta=model_ctx.bg_freeze_min_delta, verbose=verbose)

    analysis_cfg = training_ctx.parameters.get("dual_analysis", {})
    analyser = None
    if analysis_cfg.get("enabled", False):
        analyser = dual_analysis.TrunkGradAnalyser(model, every=analysis_cfg.get("grad_norm_every", 50))
        print("dual_analysis enabled: logging loss ratios and trunk grad norms under 'analysis/'")

    for epoch_idx in range(model_ctx.epochs_num):
        epoch = epoch_idx + epoch_so_far
        print(f"\n--- Start Epoch: {epoch} (phase: {sched.phase}) ---")

        # --- Schedule transitions, applied BEFORE the epoch trains ---
        sched.maybe_freeze(model, model_ctx, epoch, paths_ctx)
        sched.maybe_start_refit(model, model_ctx, epoch, paths_ctx, bg_ckpt, losses,
                                train_loader, test_loader)

        avg_train_total, avg_train_fp, avg_train_bg = train_one_epoch(
            model, train_loader, model_ctx, sched, epoch, paths_ctx=paths_ctx, analyser=analyser
        )
        avg_test_total, avg_test_fp, avg_test_bg, bg_mae_denorm, fp_test_out, bg_test_out, bg_test_true = validate_and_predict(
            model, test_loader, model_ctx, output_norm=output_norm
        )

        analysis_metrics = {}
        if analyser is not None:
            analysis_metrics = dual_analysis.loss_ratio_metrics(
                avg_train_fp, avg_train_bg, avg_test_fp, avg_test_bg, model_ctx.bg_loss_weight)
            analysis_metrics.update(analyser.epoch_means())
            analyser.reset()

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
        if sched.refit_ckpt is not None:
            sched.refit_ckpt.step(avg_test_bg, model, epoch)
        if bg_mae_denorm is not None:
            losses["test_bg_mae_denorm"].append(bg_mae_denorm)

        # --- Footprint-head evaluation (standard GATES metric pipeline) ---
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
                "bg/frozen": int(sched.phase == "frozen"),
                "bg/phase": PHASE_CODES[sched.phase],
                **analysis_metrics,
                **list_of_metrics,
            }
            if bg_mae_denorm is not None:
                logging_dict["test/bg_mae_denorm"] = bg_mae_denorm
            if sched.refit_started_epoch is not None:
                logging_dict["refit/best_test_loss_bg"] = sched.refit_ckpt.best_loss
                logging_dict["refit/bg_gap_vs_pre_refit_best"] = (
                    sched.refit_ckpt.best_loss - sched.pre_refit_bg_best)
                if sched.fp_test_at_refit_start is not None:
                    logging_dict["refit/fp_drift"] = avg_test_fp - sched.fp_test_at_refit_start
                logging_dict.update(sched.probe_metrics)
            wandb.log(logging_dict, step=epoch)

        model_ctx.early_stopping(avg_test_total, model)
        if model_ctx.early_stopping.early_stop:
            print("Early stopping triggered. Ending training.")
            break

        # Patience-based freeze (as in train_dual_model) — only used when no fixed
        # freeze_epoch is configured and the schedule is still in the joint phase.
        if (sched.freeze_epoch is None and sched.phase == "joint"
                and model_ctx.bg_freeze_patience is not None
                and bg_ckpt.counter >= model_ctx.bg_freeze_patience):
            sched.freeze(model, model_ctx, epoch, paths_ctx,
                         f"no improvement for {bg_ckpt.counter} epochs "
                         f"(best {bg_ckpt.best_loss:.6f} at epoch {bg_ckpt.best_epoch})")

        log_text = (f"Epoch {epoch} ({sched.phase}), total {avg_train_total:.4f}/{avg_test_total:.4f}, "
                    f"fp {avg_train_fp:.4f}/{avg_test_fp:.4f}, bg {avg_train_bg:.4f}/{avg_test_bg:.4f}")
        if bg_mae_denorm is not None:
            log_text += f", bg MAE(denorm) {bg_mae_denorm:.4e}"
        write_to_file(log_text, paths_ctx.updates_path)

        if output_norm is not None and bg_test_out.shape[1] == 1:
            mean, std = output_norm
            bg_true_ppb = denormalize(bg_test_true[:, 0], mean, std) * 1e9
            bg_pred_ppb = denormalize(bg_test_out[:, 0], mean, std) * 1e9

        if epoch % model_ctx.epochs_visualise == 0:
            img_save_path = save_training_plots(epoch, test_fp_dataset, training_ctx, paths_ctx.model_path, paths_ctx.model_name)
            if model_ctx.use_wandb:
                wandb.log({f"fps_epoch_{epoch}": wandb.Image(img_save_path)}, step=epoch)

            if bg_true_ppb is not None:
                bg_img_path = save_bg_timeseries_plots(
                    epoch, bg_true_ppb, bg_pred_ppb,
                    paths_ctx.model_path, paths_ctx.model_name, detrended=bg_detrended,
                    n_windows=bg_plot_n_windows,
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

    if bg_true_ppb is not None:
        test_fp_dataset["bg_true_ppb"] = (("time",), bg_true_ppb)
        test_fp_dataset["bg_pred_ppb"] = (("time",), bg_pred_ppb)
        test_fp_dataset["bg_true_ppb"].attrs["units"] = "ppb"
        test_fp_dataset["bg_pred_ppb"].attrs["units"] = "ppb"
        if bg_detrended:
            note = "detrended: relative to the south-boundary midpoint baseline"
            test_fp_dataset["bg_true_ppb"].attrs["note"] = note
            test_fp_dataset["bg_pred_ppb"].attrs["note"] = note

    # --- Run summary: per-head bests plus the schedule/refit outcome ---
    summary_msg = (f"Per-head bests: fp {fp_ckpt.best_loss:.6f} at epoch {fp_ckpt.best_epoch} "
                   f"-> {Path(fp_ckpt.path).name}; bg {bg_ckpt.best_loss:.6f} at epoch "
                   f"{bg_ckpt.best_epoch} -> {Path(bg_ckpt.path).name}")
    if sched.bg_frozen_epoch is not None:
        summary_msg += f"; bg head frozen at epoch {sched.bg_frozen_epoch}"
    if sched.refit_started_epoch is not None:
        summary_msg += (f"; refit ({sched.refit_mode}) from epoch {sched.refit_started_epoch}: "
                        f"pre-refit bg best {sched.pre_refit_bg_best:.6f}, refit bg best "
                        f"{sched.refit_ckpt.best_loss:.6f} at epoch {sched.refit_ckpt.best_epoch} "
                        f"-> {Path(sched.refit_ckpt.path).name}")
        if sched.fp_test_at_refit_start is not None and losses["test_fp"]:
            summary_msg += (f"; fp test loss {sched.fp_test_at_refit_start:.6f} (refit start) -> "
                            f"{losses['test_fp'][-1]:.6f} (final)")
        if sched.probe_metrics:
            summary_msg += (f"; linear probe calibrated test bg loss "
                            f"{sched.probe_metrics['refit/linear_probe_test_bg_loss']:.6f}")
    print(summary_msg)
    write_to_file(summary_msg, paths_ctx.updates_path)

    if model_ctx.use_wandb:
        head_ckpts = [fp_ckpt, bg_ckpt]
        if sched.refit_ckpt is not None:
            head_ckpts.append(sched.refit_ckpt)
        for head_ckpt in head_ckpts:
            if head_ckpt.best_epoch is not None:
                save_wandb_artifact(
                    model_ctx.model_name, f"best_{head_ckpt.head_name}", "model",
                    f"Best {head_ckpt.head_name}-head checkpoint (test {head_ckpt.head_name} "
                    f"loss {head_ckpt.best_loss:.6f} at epoch {head_ckpt.best_epoch})",
                    str(head_ckpt.path),
                )
        summary_update = {
            "best/epoch_fp": fp_ckpt.best_epoch,
            "best/epoch_bg": bg_ckpt.best_epoch,
            "bg/frozen_epoch": sched.bg_frozen_epoch,
            "refit/started_epoch": sched.refit_started_epoch,
            "refit/mode": sched.refit_mode if sched.refit_started_epoch is not None else None,
        }
        if sched.refit_started_epoch is not None:
            summary_update.update({
                "refit/best_test_loss_bg": sched.refit_ckpt.best_loss,
                "refit/best_epoch_bg": sched.refit_ckpt.best_epoch,
                "refit/pre_refit_best_test_loss_bg": sched.pre_refit_bg_best,
                "refit/fp_test_loss_at_start": sched.fp_test_at_refit_start,
                "refit/fp_test_loss_final": losses["test_fp"][-1] if losses["test_fp"] else None,
                **sched.probe_metrics,
            })
        wandb.run.summary.update(summary_update)

    export_results_to_netcdf(test_fp_dataset, paths_ctx.model_path, model_ctx.model_name, use_wandb=model_ctx.use_wandb)
    print("Finished Training.")


# ---------------------------------------------------------------
# Top-level training entry point (same setup as train_dual_model, different loop)
# ---------------------------------------------------------------

def train_and_save_model(parameters, model_save_dir, wandb_name=None, data_bundle=None):
    """Full dual-head training run with the bg freeze/refit schedule.

    Identical setup to ``train_dual_model.train_and_save_model`` (same data pipeline,
    dataloaders, model and optimizer construction) — only the training loop differs
    (this module's :func:`run_full_training`). See that function for the argument docs.
    """
    use_wandb = parameters.get('use_wandb', True)
    cfg = gates.config.get_config()
    verbose = parameters.get("verbose", True)

    # Validate the schedule up-front so a bad config fails before the expensive load.
    BgSchedule(parameters)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{parameters['model_name']}_{timestamp}"
    model_path = Path(model_save_dir) / model_name
    print(f"Initialising dual refit run for model_name: {model_name}")

    paths_ctx = PathContext(model_save_dir=model_save_dir, model_name=model_name, model_path=model_path)
    paths_ctx.make_dirs()

    set_reproducibility(parameters.get("seed", 34))

    if use_wandb and wandb.run is not None:
        wandb.config.update(parameters, allow_val_change=True)
    elif use_wandb:
        wandb_project = parameters.get("wandb", {}).get("project", None)
        wandb_entity = parameters.get("wandb", {}).get("entity", None)
        wandb_tags = parameters.get("wandb", {}).get("tags", [])
        wandb_group = parameters.get("wandb", {}).get("group", None)
        # Free-text description of what this run is for, shown in the W&B run's Notes
        # field; set per experiment via a "wandb": {"notes": ...} override.
        wandb_notes = parameters.get("wandb", {}).get("notes", None)
        if wandb_project is None or wandb_entity is None:
            print("Warning: use_wandb is True but no wandb.project/entity specified. W&B disabled.")
            use_wandb = False
            parameters["use_wandb"] = False
        if use_wandb:
            job_id = os.environ.get("SLURM_JOB_ID", "local")
            job_name = os.environ.get("SLURM_JOB_NAME", "run")
            if wandb_name:
                suffix = wandb_name
            else:
                bg_loss_weight = parameters.get("loss_functions", {}).get("bg_loss_weight")
                suffix = f"bg{bg_loss_weight}"
            run_name = f"{job_id}_{job_name}_{suffix}"
            wandb.init(entity=wandb_entity, project=wandb_project, config=parameters, tags=wandb_tags, name=run_name, group=wandb_group, notes=wandb_notes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_to_file(f"using device {device}, starting at " + datetime.now().strftime("%d/%m/%y %H:%M:%S"), paths_ctx.updates_path)

    background_params = resolve_background_params(parameters)
    parameters["background_setup"] = background_params

    num_classes = parameters["model_parameters"].get("num_classes", 1)
    if num_classes not in [1, 4]:
        print("Warning: num_classes not in [1, 4]. Defaulting to 1 (summed output).")
        num_classes = 1
        parameters["model_parameters"]["num_classes"] = 1

    if data_bundle is None:
        print("Loading met, fp AND BACKGROUND data for model", model_name)
        data_bundle = load_dual_data(parameters, verbose=verbose)
    else:
        print("Reusing pre-loaded shared data bundle for model", model_name)
    (train_fp_data, train_inputs, train_bgs, train_aux_cams_data,
     test_fp_data, test_inputs, test_bgs, test_aux_cams_data) = data_bundle

    if num_classes == 1:
        train_bgs = train_bgs[["summed"]].to_dataarray()
        test_bgs = test_bgs[["summed"]].to_dataarray()
    else:
        train_bgs = train_bgs[["north", "south", "east", "west"]].to_dataarray()
        test_bgs = test_bgs[["north", "south", "east", "west"]].to_dataarray()

    train_bgs = train_bgs.transpose("time", "variable")
    test_bgs = test_bgs.transpose("time", "variable")

    use_auxiliary_bc = background_params["use_auxiliary_bc"]
    if use_auxiliary_bc:
        train_aux_cams_data = format_aux_data(train_aux_cams_data, time_coord=train_fp_data.time)
        test_aux_cams_data = format_aux_data(test_aux_cams_data, time_coord=test_fp_data.time)

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
    parser = argparse.ArgumentParser(
        description="Train the dual-head model with a bg freeze/refit schedule. "
                    "Example: python train_dual_refit_model.py parameters.json")
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
