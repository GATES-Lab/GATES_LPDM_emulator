"""Tests for the ANALYSIS-ONLY dual-model diagnostics (gates/training/dual_analysis.py).

Covers:
  - loss_ratio_metrics: raw and contribution ratios, NaN guards (e.g. bg_loss_weight = 0).
  - TrunkGradAnalyser.measure: positive trunk norms for both heads, cosine in [-1, 1],
    no side effects on param.grad, and the training graph still usable afterwards.
  - The analysis-only guarantee: train_one_epoch with and without an analyser produces
    bitwise-identical parameters.
  - Frozen bg head: its trunk-gradient norm is reported as 0 and the cosine as NaN.

Run via SLURM (never on the login node): ``sbatch launch_test_bg_freeze.sh``
"""

import copy
import math
import os
import sys

# repo root (parent of tests/) so the local `model` and `gates` packages import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, "/user/work/yl18410/new_graphnet")
sys.path.insert(2, "/user/work/yl18410/new_graphnet/graphnet_LPDM_emulator")

import numpy as np
import pytest
import torch

import gates.training.dual_analysis as dual_analysis
import gates.training.training_dual as gates_training_dual
from train_dual_model import train_one_epoch, _fp_true_values
from tests.test_bg_freeze_checkpoint import small_dual_model, make_ctx, make_loader


# ---------------------------------------------------------------
# loss_ratio_metrics
# ---------------------------------------------------------------

def test_loss_ratio_metrics_values():
    m = dual_analysis.loss_ratio_metrics(2.0, 0.5, 1.0, 0.25, bg_loss_weight=0.2)
    assert m["analysis/train_loss_ratio_fp_over_bg"] == pytest.approx(4.0)
    assert m["analysis/test_loss_ratio_fp_over_bg"] == pytest.approx(4.0)
    assert m["analysis/train_loss_contribution_fp_over_bg"] == pytest.approx((0.8 * 2.0) / (0.2 * 0.5))


def test_loss_ratio_metrics_nan_guards():
    # w = 0: the bg contribution is zero, so the contribution ratio is undefined (NaN),
    # not inf; a zero bg loss likewise must not divide by zero.
    m = dual_analysis.loss_ratio_metrics(2.0, 0.5, 1.0, 0.25, bg_loss_weight=0.0)
    assert math.isnan(m["analysis/train_loss_contribution_fp_over_bg"])
    m = dual_analysis.loss_ratio_metrics(2.0, 0.0, 1.0, 0.0, bg_loss_weight=0.2)
    assert math.isnan(m["analysis/train_loss_ratio_fp_over_bg"])
    assert math.isnan(m["analysis/test_loss_ratio_fp_over_bg"])


# ---------------------------------------------------------------
# TrunkGradAnalyser
# ---------------------------------------------------------------

def _manual_losses(model, batch):
    features, fps, bgs = batch
    outputs = model(features)
    fp_loss = torch.nn.functional.mse_loss(outputs["footprint"], _fp_true_values(fps))
    bg_loss = torch.nn.functional.mse_loss(outputs["background"], bgs)
    return fp_loss, bg_loss


def test_measure_reports_norms_without_touching_grads():
    torch.manual_seed(0)
    model = small_dual_model()
    analyser = dual_analysis.TrunkGradAnalyser(model, every=1)
    fp_loss, bg_loss = _manual_losses(model, make_loader(1)[0])

    analyser.measure(fp_loss, bg_loss, bg_loss_weight=0.2)

    # analysis must not write into param.grad anywhere in the model
    assert all(p.grad is None for p in model.parameters())

    means = analyser.epoch_means()
    assert means["analysis/trunk_grad_norm_fp"] > 0
    assert means["analysis/trunk_grad_norm_bg"] > 0
    assert -1.0 - 1e-6 <= means["analysis/trunk_grad_cosine"] <= 1.0 + 1e-6
    assert np.isfinite(means["analysis/trunk_grad_contribution_fp_over_bg"])

    # the training graph was retained: the usual joint backward still works
    (0.8 * fp_loss + 0.2 * bg_loss).backward()
    assert any(p.grad is not None for p in model.encoder.parameters())

    analyser.reset()
    assert analyser.epoch_means() == {}


def test_analysis_does_not_change_training():
    torch.manual_seed(0)
    model_a = small_dual_model()
    model_b = copy.deepcopy(model_a)
    loader = make_loader()  # same tensors for both runs
    ctx_a, ctx_b = make_ctx(model_a), make_ctx(model_b)
    analyser = dual_analysis.TrunkGradAnalyser(model_a, every=1)

    totals_a = train_one_epoch(model_a, loader, ctx_a, epoch=0, analyser=analyser)
    totals_b = train_one_epoch(model_b, loader, ctx_b, epoch=0)

    assert analyser.epoch_means(), "analyser should have measured every batch"
    assert totals_a == totals_b
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(pa, pb), "analysis changed the training trajectory"


def test_frozen_bg_head_reports_zero_grad_and_nan_cosine():
    torch.manual_seed(0)
    model = small_dual_model()
    model_ctx = make_ctx(model)
    gates_training_dual.freeze_bg_head(model)
    model_ctx.bg_frozen = True
    analyser = dual_analysis.TrunkGradAnalyser(model, every=1)

    train_one_epoch(model, make_loader(), model_ctx, epoch=0, analyser=analyser)

    means = analyser.epoch_means()
    assert means["analysis/trunk_grad_norm_fp"] > 0
    assert means["analysis/trunk_grad_norm_bg"] == 0.0
    assert math.isnan(means["analysis/trunk_grad_cosine"])
