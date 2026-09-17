"""Train the dual-head GATES model with PER-HEAD LEARNING RATES (+ the bg freeze/refit schedule).

This is a copy of ``train_dual_refit_model.py`` (unchanged, still the reference for the
freeze/refit schedule) that additionally lets the footprint head, the background head and
the shared trunk train with DIFFERENT learning rates. Everything else — data pipeline,
schedule, checkpoints, logging — is identical to the refit trainer.

The learning rates come from an optional ``head_learning_rates`` block; any key that is
absent falls back to the top-level ``learning_rate`` (so with no block this script is the
refit trainer)::

    "learning_rate": 5e-5,            # default for every parameter group
    "head_learning_rates": {
        "trunk": 5e-5,                # encoder + processor (everything not in a decoder head)
        "fp": 5e-5,                   # fp_decoder (footprint head)
        "bg": 5e-4                    # bg_decoder (background / boundary-condition head)
    }

The optimizer is one AdamW with three parameter groups (trunk / fp / bg), so a single
``optimizer.step()`` still updates the whole model and ``bg_head.weight_decay`` is still
honoured on the bg group. ``bg_head.lr_scale`` is NOT combined with this block: setting
both is rejected (use ``head_learning_rates.bg`` instead). During a refit the fresh bg
optimizer uses ``head_learning_rates.bg * bg_schedule.refit_lr_scale``.

The resolved rates are written to ``training_settings_*.json`` under
``head_learning_rates_resolved`` and, with W&B, logged once as ``lr/trunk``, ``lr/fp``,
``lr/bg``.

Optionally the bg head can be WARMED UP so it only reaches full speed once the trunk
features have settled (all keys optional; no block or ``epochs`` 0/null = off)::

    "bg_warmup": {
        "epochs": 40,            # ramp over the first N epochs (full speed from epoch N on)
        "mode": "lr",            # "lr":     bg-group LR   = head_learning_rates.bg * scale
                                 # "weight": bg loss term  = w * scale * bg_loss (the fp
                                 #           coefficient stays (1 - w); w itself is unchanged)
        "start_scale": 0.0,      # scale at epoch 0 (0 = bg head does not move at all)
        "shape": "linear"        # "linear" or "cosine" (half-cosine from start_scale to 1)
    }

The scale at epoch ``e`` (< epochs) is ``start + (1 - start) * f(e / epochs)`` with
``f`` the identity (linear) or ``(1 - cos(pi x)) / 2`` (cosine); it is applied at the START
of each epoch and logged as ``bg/warmup_scale`` (and ``lr/bg_current`` for mode "lr").
The warm-up only makes sense in the joint phase: a ``bg_schedule.freeze_epoch`` inside the
warm-up window is rejected.

Below is the freeze/refit schedule documentation, unchanged from the refit trainer:

  Phase 1 "joint"   epochs [0, freeze_epoch):        normal joint training, as before.
  Phase 2 "frozen"  epochs [freeze_epoch, refit_start_epoch):  the bg head is frozen and its
                    loss dropped from the joint loss (identical to the existing
                    ``bg_head.freeze_patience`` behaviour, but at a FIXED epoch so parallel
                    experiment arms share the same trajectory up to the refit).
  Phase 3 "refit"   epochs [refit_start_epoch, end):  the bg head is unfrozen and re-fitted
                    (with a fresh AdamW optimizer for the bg decoder). The refit ALWAYS
                    trains the trunk (encoder + processor) together with the bg decoder —
                    a head-only refit cannot recover a drifted trunk — in one of two modes:
                      - "bg_trunk": the fp decoder alone is frozen; the trunk AND the bg
                        decoder train against the bg criterion alone (the fp loss is
                        dropped, mirroring how the bg freeze drops the bg loss). The fp
                        test loss then measures how much bg-directed trunk movement costs
                        the fixed fp head.
                      - "joint":  full joint training resumes ((1-w)*fp + w*bg); the trunk
                        receives bg gradients again, so the fp test loss shows whether
                        re-fitting the bg head makes the fp head drift.

The ``bg_schedule`` block (all keys optional; with no block this script reduces to
``train_dual_model.py`` behaviour)::

    "bg_schedule": {
        "freeze_epoch": 75,          # freeze the bg head at the START of this epoch
                                     # (null = fall back to bg_head.freeze_patience)
        "refit_start_epoch": 120,    # unfreeze + refit from the start of this epoch
                                     # (null = never refit)
        "refit_mode": "bg_trunk",    # "bg_trunk" or "joint" (see above)
        "refit_lr_scale": 1.0        # bg-decoder LR during refit = learning_rate * this
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
``refit/`` in W&B: the refit-phase bg best and the fp drift since the refit started.

Data loading, validation and the footprint metric pipeline are imported unchanged from
``train_dual_model``; the shared-data driver runs this script's ``train_and_save_model``
via ``run_dual_experiments_shared_data.py --trainer train_dual_headlr_model`` (the
launcher exposes this as ``TRAINER=train_dual_headlr_model``).
"""

import os
import sys
import math
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
import gates.training.distributed as gates_distributed  # multi-GPU (no-ops on one GPU)
from gates.data.load_data import get_grid
from gates.training.training_background import format_aux_data, normalize_boundary_data, denormalize
from gates.training.training_dataclasses import PathContext, BoundaryTrainingContext
from gates.training.training_helperfuns import (
    load_parameter_file, save_object, write_to_file, save_training_plots,
    save_bg_timeseries_plots, export_results_to_netcdf, set_reproducibility,
    enable_deterministic_algorithms,
    HeadCheckpoint, save_wandb_artifact,
)

# Reused unchanged from the standard dual trainer (kept there so both scripts share one
# implementation; the shared-data driver also imports load_dual_data from this module).
from train_dual_model import (
    _fp_true_values, validate_and_predict, load_dual_data, resolve_background_params,
    log_data_loading_summary_run, DualDataBundle,
)


# ---------------------------------------------------------------
# Per-head learning rates
# ---------------------------------------------------------------

HEAD_GROUPS = ("trunk", "fp", "bg")


def resolve_head_learning_rates(parameters):
    """Return ``{"trunk": lr, "fp": lr, "bg": lr}`` from ``parameters`` (see module docstring).

    Missing keys default to the top-level ``learning_rate``. Raises ``ValueError`` on
    unknown keys, non-positive rates, or when ``bg_head.lr_scale`` is also set (the two
    would silently compound).
    """
    base_lr = parameters["learning_rate"]
    cfg = parameters.get("head_learning_rates", None) or {}
    unknown = set(cfg) - set(HEAD_GROUPS)
    if unknown:
        raise ValueError(f"head_learning_rates: unknown keys {sorted(unknown)}; "
                         f"allowed: {list(HEAD_GROUPS)}")
    lrs = {name: (base_lr if cfg.get(name) is None else float(cfg[name])) for name in HEAD_GROUPS}
    bad = {k: v for k, v in lrs.items() if not v > 0}
    if bad:
        raise ValueError(f"head_learning_rates must be > 0, got {bad}")
    bg_lr_scale = parameters.get("bg_head", {}).get("lr_scale", 1.0)
    if cfg and bg_lr_scale != 1.0:
        raise ValueError("head_learning_rates and bg_head.lr_scale are both set; use "
                         "head_learning_rates.bg alone (bg_head.lr_scale would compound with it)")
    return lrs


def build_head_lr_optimizer(model, parameters, lrs):
    """One AdamW with a parameter group per head: trunk (everything that is not a decoder
    head), ``fp_decoder`` and ``bg_decoder``, each at its own learning rate.

    ``bg_head.weight_decay`` (if set) applies to the bg group only, as in
    ``setup_dual_model``. Every model parameter lands in exactly one group.
    """
    fp_params = list(model.fp_decoder.parameters())
    bg_params = list(model.bg_decoder.parameters())
    head_ids = {id(p) for p in fp_params} | {id(p) for p in bg_params}
    trunk_params = [p for p in model.parameters() if id(p) not in head_ids]

    groups = [
        {"params": trunk_params, "lr": lrs["trunk"], "name": "trunk"},
        {"params": fp_params, "lr": lrs["fp"], "name": "fp"},
        {"params": bg_params, "lr": lrs["bg"], "name": "bg"},
    ]
    bg_weight_decay = parameters.get("bg_head", {}).get("weight_decay", None)
    if bg_weight_decay is not None:
        groups[2]["weight_decay"] = bg_weight_decay

    n_total = sum(1 for _ in model.parameters())
    n_grouped = sum(len(g["params"]) for g in groups)
    assert n_grouped == n_total, f"param groups cover {n_grouped} of {n_total} tensors"

    optimizer = optim.AdamW(groups, lr=parameters["learning_rate"])
    for g in groups:
        n_el = sum(p.numel() for p in g["params"])
        print(f"optimizer group {g['name']:5s}: lr={g['lr']:g}, "
              f"weight_decay={g.get('weight_decay', 'default')}, params={n_el}")
    return optimizer


# ---------------------------------------------------------------
# Background-head schedule (freeze -> refit)
# ---------------------------------------------------------------

PHASE_CODES = {"joint": 0, "frozen": 1, "refit_joint": 3, "refit_bg_trunk": 4}


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
        self.refit_mode = cfg.get("refit_mode", "bg_trunk")
        self.refit_lr_scale = cfg.get("refit_lr_scale", 1.0)

        # --- bg warm-up (see module docstring) ---
        wcfg = parameters.get("bg_warmup", None) or {}
        self.warmup_epochs = int(wcfg.get("epochs") or 0)
        self.warmup_mode = wcfg.get("mode", "lr")
        self.warmup_start = float(wcfg.get("start_scale", 0.0))
        self.warmup_shape = wcfg.get("shape", "linear")
        if self.warmup_epochs < 0:
            raise ValueError(f"bg_warmup.epochs must be >= 0, got {self.warmup_epochs}")
        if self.warmup_mode not in ("lr", "weight"):
            raise ValueError(f"bg_warmup.mode must be 'lr' or 'weight', got {self.warmup_mode!r}")
        if self.warmup_shape not in ("linear", "cosine"):
            raise ValueError(f"bg_warmup.shape must be 'linear' or 'cosine', got {self.warmup_shape!r}")
        if not 0.0 <= self.warmup_start <= 1.0:
            raise ValueError(f"bg_warmup.start_scale must be in [0, 1], got {self.warmup_start}")
        if (self.warmup_epochs and self.freeze_epoch is not None
                and self.freeze_epoch < self.warmup_epochs):
            raise ValueError("bg_schedule.freeze_epoch must not fall inside the bg_warmup window "
                             f"(freeze_epoch {self.freeze_epoch} < bg_warmup.epochs {self.warmup_epochs})")
        # multiplier on the bg loss TERM in the joint loss (mode "weight"); 1.0 = no warm-up
        self.bg_loss_scale = 1.0
        self.warmup_scale = 1.0

        if self.refit_mode not in ("bg_trunk", "joint"):
            raise ValueError("bg_schedule.refit_mode must be 'bg_trunk' or 'joint', "
                             f"got {self.refit_mode!r}")
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

    @property
    def bg_head_active(self):
        """Whether the bg head is currently being trained (its loss reaches its params)."""
        return self.phase in ("joint", "refit_bg_trunk", "refit_joint")

    def warmup_scale_at(self, epoch):
        """Warm-up multiplier for ``epoch`` (1.0 once the warm-up is over or disabled)."""
        if not self.warmup_epochs or epoch >= self.warmup_epochs:
            return 1.0
        x = epoch / self.warmup_epochs
        f = x if self.warmup_shape == "linear" else (1.0 - math.cos(math.pi * x)) / 2.0
        return self.warmup_start + (1.0 - self.warmup_start) * f

    def apply_warmup(self, model_ctx, epoch, paths_ctx=None):
        """Apply the bg warm-up for ``epoch`` (call at the START of every epoch).

        mode "lr":     sets the LR of the optimizer group named "bg" to bg_lr * scale
                       (bg_lr = head_learning_rates.bg); restores the full rate after the window.
        mode "weight": sets ``self.bg_loss_scale`` (used by ``train_one_epoch``); LR untouched.
        """
        scale = self.warmup_scale_at(epoch)
        self.warmup_scale = scale
        if not self.warmup_epochs:
            return scale
        if self.warmup_mode == "lr":
            bg_lr = getattr(model_ctx, "head_lrs", {}).get("bg", model_ctx.lr)
            groups = [g for g in model_ctx.optimizer.param_groups if g.get("name") == "bg"]
            if not groups:
                raise RuntimeError("bg_warmup mode 'lr' needs an optimizer group named 'bg' "
                                   "(built by build_head_lr_optimizer)")
            for g in groups:
                g["lr"] = bg_lr * scale
        else:
            self.bg_loss_scale = scale
        if epoch <= self.warmup_epochs and paths_ctx is not None and (
                epoch % 10 == 0 or epoch == self.warmup_epochs):
            msg = (f"bg warm-up ({self.warmup_mode}, {self.warmup_shape}) epoch {epoch}: "
                   f"scale {scale:.3f}" + (" (warm-up complete)" if epoch == self.warmup_epochs else ""))
            print(msg)
            write_to_file(msg, paths_ctx.updates_path)
        return scale

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

    def maybe_start_refit(self, model, model_ctx, epoch, paths_ctx, bg_ckpt, losses):
        if self.refit_start_epoch is None or epoch < self.refit_start_epoch:
            return
        if self.phase not in ("joint", "frozen"):
            return  # already refitting
        if self.phase == "joint":
            msg = (f"Warning: refit starting at epoch {epoch} but the bg head was never frozen "
                   "(no freeze_epoch hit and no patience freeze) — refitting from a live head.")
            print(msg)
            write_to_file(msg, paths_ctx.updates_path)

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
        # The refit LR is the bg HEAD's rate (not the top-level learning_rate) times the scale.
        bg_lr = getattr(model_ctx, "head_lrs", {}).get("bg", model_ctx.lr)
        refit_lr = bg_lr * self.refit_lr_scale
        self.bg_optimizer = optim.AdamW(model.bg_decoder.parameters(), lr=refit_lr)

        if self.refit_mode == "bg_trunk":
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
               f"lr={refit_lr:g} = bg lr {bg_lr:g} x {self.refit_lr_scale:g}, fresh bg optimizer): "
               f"pre-refit bg best {self.pre_refit_bg_best:.6f} at epoch {bg_ckpt.best_epoch}, "
               f"fp test loss at refit start {self.fp_test_at_refit_start}")
        print(msg)
        write_to_file(msg, paths_ctx.updates_path)


# ---------------------------------------------------------------
# Training loop (phase-aware variant of train_dual_model.train_one_epoch)
# ---------------------------------------------------------------

def train_one_epoch(model, loader, model_ctx, sched, epoch, paths_ctx=None):
    """One training epoch; the loss composition and which optimizers step depend on
    ``sched.phase`` (see module docstring). Both per-head losses are always logged.
    """
    phase = sched.phase
    model.train()
    if phase == "frozen":
        # keep the frozen head's dropout / norm statistics fixed
        model.bg_decoder.eval()
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
            # sched.bg_loss_scale is 1 unless a "weight"-mode bg warm-up is ramping
            if sched.bg_loss_scale > 0:
                bg_loss = model_ctx.bg_criterion(bg_pred, bg_batch)
                loss = ((1 - model_ctx.bg_loss_weight) * fp_loss
                        + model_ctx.bg_loss_weight * sched.bg_loss_scale * bg_loss)
            else:
                # Scale exactly 0: detach so the bg head gets NO grad at all (a zero-coefficient
                # term would still leave zero grads, and AdamW's decoupled weight decay would
                # then shrink the head). Logged bg loss only, as in the frozen phase.
                bg_loss = model_ctx.bg_criterion(bg_pred.detach(), bg_batch)
                loss = (1 - model_ctx.bg_loss_weight) * fp_loss
        elif phase == "frozen":
            # As in train_dual_model: bg loss logged on detached predictions and dropped
            # from the joint loss; the fp term keeps its (1 - w) scaling.
            fp_loss = model_ctx.fp_criterion(fp_pred, true_values, fp_batch)
            bg_loss = model_ctx.bg_criterion(bg_pred.detach(), bg_batch)
            loss = (1 - model_ctx.bg_loss_weight) * fp_loss
        elif phase == "refit_bg_trunk":
            # The fp head is frozen and its loss dropped (mirroring the bg freeze); the
            # bg criterion alone trains the trunk + bg decoder, unweighted — the weight
            # only rescales the single term. fp_loss on detached predictions is logging only.
            fp_loss = model_ctx.fp_criterion(fp_pred.detach(), true_values, fp_batch)
            bg_loss = model_ctx.bg_criterion(bg_pred, bg_batch)
            loss = bg_loss
        else:
            raise RuntimeError(f"unknown schedule phase {phase!r}")

        loss.backward()
        gates_distributed.average_gradients(model)  # multi-GPU: mean gradient over all ranks
        model_ctx.optimizer.step()  # trunk (+ fp head unless frozen); bg params only before the refit
        if sched.bg_optimizer is not None and phase in ("refit_bg_trunk", "refit_joint"):
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
    # multi-GPU: epoch means over all ranks' shards (the progress lines above are rank 0 only)
    return gates_distributed.all_reduce_mean(
        (running_total / denom, running_fp / denom, running_bg / denom), model_ctx.device)


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

    # Multi-GPU: every rank runs this loop in lockstep (identical losses -> identical
    # decisions), but only rank 0 evaluates metrics, logs, plots and writes files; the other
    # ranks get test_fp_dataset=None. Always True on a single GPU.
    is_main = gates_distributed.is_main_process()

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

    for epoch_idx in range(model_ctx.epochs_num):
        epoch = epoch_idx + epoch_so_far
        print(f"\n--- Start Epoch: {epoch} (phase: {sched.phase}) ---")

        # --- Schedule transitions, applied BEFORE the epoch trains ---
        sched.apply_warmup(model_ctx, epoch, paths_ctx)
        sched.maybe_freeze(model, model_ctx, epoch, paths_ctx)
        sched.maybe_start_refit(model, model_ctx, epoch, paths_ctx, bg_ckpt, losses)

        avg_train_total, avg_train_fp, avg_train_bg = train_one_epoch(
            model, train_loader, model_ctx, sched, epoch, paths_ctx=paths_ctx
        )
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
        if sched.refit_ckpt is not None:
            sched.refit_ckpt.step(avg_test_bg, model, epoch)
        if bg_mae_denorm is not None:
            losses["test_bg_mae_denorm"].append(bg_mae_denorm)

        list_of_metrics = {}
        if is_main:
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
                "bg/warmup_scale": sched.warmup_scale,
                "lr/bg_current": next((g["lr"] for g in model_ctx.optimizer.param_groups
                                       if g.get("name") == "bg"), model_ctx.lr),
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

        if is_main and epoch % model_ctx.epochs_visualise == 0:
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

        if is_main and epoch % model_ctx.epochs_save == 0:
            checkpoint_path = paths_ctx.model_path / f"{paths_ctx.model_name}_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model_ctx.optimizer.state_dict(),
                'loss': losses,
                'learning_rate': model_ctx.lr,
                'head_learning_rates': getattr(model_ctx, "head_lrs", None),
            }, checkpoint_path)

    if is_main and bg_true_ppb is not None:
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
    if sched.warmup_epochs:
        summary_msg += (f"; bg warm-up ({sched.warmup_mode}, {sched.warmup_shape}, "
                        f"start {sched.warmup_start:g}) over epochs 0-{sched.warmup_epochs}")
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
            "bg/warmup_epochs": sched.warmup_epochs,
            "bg/warmup_mode": sched.warmup_mode if sched.warmup_epochs else None,
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
            })
        wandb.run.summary.update(summary_update)

    if is_main:
        export_results_to_netcdf(test_fp_dataset, paths_ctx.model_path, model_ctx.model_name, use_wandb=model_ctx.use_wandb)
    print("Finished Training.")


def build_model_and_context(parameters, training_ctx, paths_ctx):
    """Model + context exactly as this trainer trains them: ``setup_dual_model`` with its
    optimizer replaced by the per-head-LR one. Also called by the multi-GPU worker processes
    (``gates/training/distributed.py``), which must build an identical model/optimizer.
    """
    head_lrs = resolve_head_learning_rates(parameters)
    model, model_ctx = gates_training_dual.setup_dual_model(parameters, training_ctx, paths_ctx)
    model_ctx.optimizer = build_head_lr_optimizer(model, parameters, head_lrs)
    model_ctx.head_lrs = head_lrs
    return model, model_ctx


# ---------------------------------------------------------------
# Top-level training entry point (same setup as train_dual_model, different loop)
# ---------------------------------------------------------------

def train_and_save_model(parameters, model_save_dir, wandb_name=None, data_bundle=None):
    """Full dual-head training run with per-head learning rates and the bg freeze/refit schedule.

    Identical setup to ``train_dual_model.train_and_save_model`` (same data pipeline,
    dataloaders and model construction); the optimizer built by ``setup_dual_model`` is
    replaced by the three-group AdamW from :func:`build_head_lr_optimizer`, and the
    training loop is this module's :func:`run_full_training`. See ``train_dual_model``
    for the argument docs.
    """
    use_wandb = parameters.get('use_wandb', True)
    cfg = gates.config.get_config()
    verbose = parameters.get("verbose", True)

    # Validate the schedule and the per-head learning rates up-front so a bad config fails
    # before the expensive load.
    BgSchedule(parameters)
    head_lrs = resolve_head_learning_rates(parameters)
    parameters["head_learning_rates_resolved"] = dict(head_lrs)
    print(f"Per-head learning rates: {head_lrs}")

    # Validate the multi-GPU config up-front too, and record the resolved per-GPU / effective
    # batch sizes with the training settings.
    num_gpus = gates_distributed.resolve_num_gpus(parameters)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{parameters['model_name']}_{timestamp}"
    model_path = Path(model_save_dir) / model_name
    print(f"Initialising dual head-LR run for model_name: {model_name}")

    paths_ctx = PathContext(model_save_dir=model_save_dir, model_name=model_name, model_path=model_path)
    paths_ctx.make_dirs()

    set_reproducibility(parameters.get("seed", 34))
    enable_deterministic_algorithms(parameters)

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

    # Multi-GPU (parameters["distributed"], see gates/training/distributed.py): hand the
    # batches to one worker process per extra GPU; this process carries on as rank 0. On a
    # single GPU this returns the loaders untouched (dist_run is None).
    run_kwargs = dict(output_norm=norm_vals["outputs"], epoch_so_far=0,
                      bg_detrended=background_params["detrend"])
    dist_run, train_loader, test_loader = gates_distributed.setup_training_loaders(
        Path(__file__).stem, num_gpus, parameters, training_ctx, paths_ctx,
        train_loader, test_loader, run_kwargs)

    try:
        # Swap the single-LR optimizer from setup_dual_model for the per-head one (inside
        # build_model_and_context). Done before any training step, so no optimizer state is
        # lost; model_ctx.lr stays the top-level default.
        model, model_ctx = build_model_and_context(parameters, training_ctx, paths_ctx)
        gates_distributed.sync_model_from_main(model)
        write_to_file(f"per-head learning rates: {head_lrs}", paths_ctx.updates_path)

        if use_wandb:
            wandb.watch(model, log="all", log_freq=100)
            wandb.log({f"lr/{k}": v for k, v in head_lrs.items()}, step=0)

        losses = gates_training_dual.initialise_dual_losses()

        run_full_training(
            model, model_ctx, training_ctx, paths_ctx, train_loader, test_loader,
            test_scaled_fp, losses, **run_kwargs,
        )
    except BaseException:
        if dist_run is not None:
            dist_run.abort()
        raise
    if dist_run is not None:
        dist_run.finish()

    if use_wandb:
        wandb.finish()


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the dual-head model with per-head learning rates (+ bg freeze/refit "
                    "schedule). Example: python train_dual_headlr_model.py parameters.json")
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
