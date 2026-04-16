import pytest
import torch

from gates.evaluation.loss_functions import (
    MSELoss,
    ThresholdedMSELoss,
    PixelWeightedMSELoss,
    SumWeightedMSELoss,
    MSEPlusSumLoss,
    scale,
    power,
    normalize_by_mean,
)

"""
Tests for the loss functions in gates.evaluation.loss_functions. These tests focus on:
- Basic functionality: returns scalar, finite loss for valid inputs.
- Behavior with nan masks: loss changes when mask applied, and matches between explicit vs. fp_batch mask.
- Weighting behavior: loss changes when weights applied, and matches expected math for simple cases (e.g. alpha=0 recovers MSE, manual thresholding matches ThresholdedMSELoss output).
- Error handling: raises appropriate exceptions for invalid inputs (e.g. unknown weight_label, mismatched alpha length).
"""
# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

FP_LABELS = ['fp_original', 'flux', 'fp_nan_mask', 'fp']
B, H, W = 3, 10, 10


@pytest.fixture(scope="module")
def loss_data():
    """Shared tensors used across all loss-function tests.

    pred / target : (3, 10, 10), values in [-1, 2]
    nan_mask      : (3, 10, 10) bool, ~15 % of pixels marked invalid
    flux          : (3, 10, 10), Uniform[0, 3e-6]
    fp_original   : (3, 10, 10), Exponential, values in [0, ~1e-2]
    fp_batch      : (3, 10, 10, 4), last dim = [fp_original, flux, fp_nan_mask, fp=target]
    """
    torch.manual_seed(42)
    pred      = torch.rand(B, H, W) * 3 - 1          # Uniform[-1, 2]
    target    = torch.rand(B, H, W) * 3 - 1
    nan_mask  = (torch.rand(B, H, W) < 0.15)          # ~15 % masked
    flux      = torch.rand(B, H, W) * 3e-6            # Uniform[0, 3e-6]
    fp_original = torch.FloatTensor(B, H, W).exponential_() * 1e-3  # Exponential, O(1e-3)

    fp_batch = torch.stack(
        [fp_original, flux, nan_mask.float(), target], dim=-1
    )  # (3, 10, 10, 4)

    return {
        "pred": pred,
        "target": target,
        "nan_mask": nan_mask,
        "flux": flux,
        "fp_original": fp_original,
        "fp_batch": fp_batch,
    }


# ---------------------------------------------------------------------------
# MSELoss
# ---------------------------------------------------------------------------

class TestMSELoss:

    def test_basic_returns_scalar(self, loss_data):
        criterion = MSELoss()
        loss = criterion(loss_data["pred"], loss_data["target"])
        assert loss.shape == ()
        assert loss.isfinite()

    def test_identical_inputs_give_zero_loss(self, loss_data):
        criterion = MSELoss()
        loss = criterion(loss_data["pred"], loss_data["pred"])
        assert torch.allclose(loss, torch.tensor(0.0))

    def test_nan_mask_explicit_changes_loss(self, loss_data):
        criterion = MSELoss()
        loss_unmasked = criterion(loss_data["pred"], loss_data["target"])
        loss_masked   = criterion(loss_data["pred"], loss_data["target"],
                                  nan_mask=loss_data["nan_mask"])
        assert not torch.allclose(loss_unmasked, loss_masked)

    def test_nan_mask_from_fp_batch_equals_explicit(self, loss_data):
        criterion_explicit = MSELoss()
        criterion_label    = MSELoss(FP_LABELS, nan_mask_label='fp_nan_mask')

        loss_explicit = criterion_explicit(
            loss_data["pred"], loss_data["target"],
            nan_mask=loss_data["nan_mask"]
        )
        loss_label = criterion_label(
            loss_data["pred"], loss_data["target"],
            loss_data["fp_batch"]
        )
        assert torch.allclose(loss_explicit, loss_label)

    def test_no_fp_batch_no_nan_mask_runs(self, loss_data):
        criterion = MSELoss()
        loss = criterion(loss_data["pred"], loss_data["target"],
                         fp_batch=None, nan_mask=None)
        assert loss.isfinite()


# ---------------------------------------------------------------------------
# ThresholdedMSELoss
# ---------------------------------------------------------------------------

class TestThresholdedMSELoss:

    def test_single_threshold_returns_scalar(self, loss_data):
        criterion = ThresholdedMSELoss(FP_LABELS, weight_label='fp_original', threshold=0)
        loss = criterion(loss_data["pred"], loss_data["target"], loss_data["fp_batch"])
        assert loss.shape == ()
        assert loss.isfinite()

    def test_multiple_thresholds(self, loss_data):
        criterion = ThresholdedMSELoss(
            FP_LABELS, weight_label='fp_original',
            threshold=[0, 1e-3], alpha=[1.0, 2.0, 3.0]
        )
        loss = criterion(loss_data["pred"], loss_data["target"], loss_data["fp_batch"])
        assert loss.shape == ()
        assert loss.isfinite()

    def test_alpha_scalar_expands_to_list(self):
        # alpha=2.0 with one threshold → bins [(-inf,0], (0,inf)]; alphas [1.0, 2.0]
        criterion = ThresholdedMSELoss(
            FP_LABELS, weight_label='fp_original', threshold=0, alpha=2.0
        )
        assert criterion.alpha == [1.0, 2.0]

    def test_alpha_list_wrong_length_raises(self):
        with pytest.raises(ValueError):
            ThresholdedMSELoss(FP_LABELS, weight_label='fp_original',
                               threshold=[0], alpha=[1.0])  # need length 2

    def test_nan_mask_applied(self, loss_data):
        criterion_no_mask = ThresholdedMSELoss(
            FP_LABELS, weight_label='fp_original', threshold=0
        )
        criterion_mask = ThresholdedMSELoss(
            FP_LABELS, weight_label='fp_original', threshold=0,
            nan_mask_label='fp_nan_mask'
        )
        loss_no_mask = criterion_no_mask(
            loss_data["pred"], loss_data["target"], loss_data["fp_batch"]
        )
        loss_mask = criterion_mask(
            loss_data["pred"], loss_data["target"], loss_data["fp_batch"]
        )
        assert not torch.allclose(loss_no_mask, loss_mask)

    def test_math_single_threshold(self, loss_data):
        """Manual two-bin MSE matches criterion output."""
        threshold = 5e-4
        alpha = [1.0, 2.0]
        criterion = ThresholdedMSELoss(
            FP_LABELS, weight_label='fp_original',
            threshold=threshold, alpha=alpha
        )

        pred, target = loss_data["pred"], loss_data["target"]
        w = loss_data["fp_original"]

        mask_low  = w <= threshold
        mask_high = w > threshold

        mse_low  = ((pred[mask_low]  - target[mask_low])  ** 2).mean()
        mse_high = ((pred[mask_high] - target[mask_high]) ** 2).mean()
        expected = alpha[0] * mse_low + alpha[1] * mse_high

        loss = criterion(pred, target, loss_data["fp_batch"])
        assert torch.allclose(loss, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# PixelWeightedMSELoss
# ---------------------------------------------------------------------------

class TestPixelWeightedMSELoss:

    def test_basic_returns_scalar(self, loss_data):
        criterion = PixelWeightedMSELoss(FP_LABELS, weight_label='fp_original')
        loss = criterion(loss_data["pred"], loss_data["target"], loss_data["fp_batch"])
        assert loss.shape == ()
        assert loss.isfinite()

    def test_unit_weights_equals_mse(self, loss_data):
        """When every weight pixel is 1, PixelWeightedMSE == plain MSE."""
        # Build fp_batch where fp_original channel is all-ones
        fp_batch_ones = loss_data["fp_batch"].clone()
        fp_batch_ones[..., FP_LABELS.index('fp_original')] = 1.0

        criterion_weighted = PixelWeightedMSELoss(FP_LABELS, weight_label='fp_original')
        criterion_plain    = MSELoss()

        loss_weighted = criterion_weighted(
            loss_data["pred"], loss_data["target"], fp_batch_ones
        )
        loss_plain = criterion_plain(loss_data["pred"], loss_data["target"])
        assert torch.allclose(loss_weighted, loss_plain, atol=1e-6)

    def test_transform_scale(self, loss_data):
        c1 = PixelWeightedMSELoss(FP_LABELS, weight_label='fp_original',
                                   transform_fn=scale, w=1.0)
        c2 = PixelWeightedMSELoss(FP_LABELS, weight_label='fp_original',
                                   transform_fn=scale, w=500.0)
        l1 = c1(loss_data["pred"], loss_data["target"], loss_data["fp_batch"])
        l2 = c2(loss_data["pred"], loss_data["target"], loss_data["fp_batch"])
        assert torch.allclose(l2, 500.0 * l1, rtol=1e-5)

    def test_transform_power(self, loss_data):
        criterion = PixelWeightedMSELoss(FP_LABELS, weight_label='fp_original',
                                          transform_fn=power, p=2.0)
        loss = criterion(loss_data["pred"], loss_data["target"], loss_data["fp_batch"])
        assert loss.isfinite()

    def test_nan_mask_explicit(self, loss_data):
        criterion = PixelWeightedMSELoss(FP_LABELS, weight_label='fp_original')
        loss_no_mask = criterion(loss_data["pred"], loss_data["target"],
                                  loss_data["fp_batch"])
        loss_masked  = criterion(loss_data["pred"], loss_data["target"],
                                  loss_data["fp_batch"], nan_mask=loss_data["nan_mask"])
        assert not torch.allclose(loss_no_mask, loss_masked)

    def test_nan_mask_from_fp_batch_equals_explicit(self, loss_data):
        c_explicit = PixelWeightedMSELoss(FP_LABELS, weight_label='fp_original')
        c_label    = PixelWeightedMSELoss(FP_LABELS, weight_label='fp_original',
                                           nan_mask_label='fp_nan_mask')
        l_explicit = c_explicit(loss_data["pred"], loss_data["target"],
                                 loss_data["fp_batch"], nan_mask=loss_data["nan_mask"])
        l_label    = c_label(loss_data["pred"], loss_data["target"],
                              loss_data["fp_batch"])
        assert torch.allclose(l_explicit, l_label)

    def test_unknown_weight_label_raises(self):
        with pytest.raises(ValueError, match="not found"):
            PixelWeightedMSELoss(FP_LABELS, weight_label='nonexistent')


# ---------------------------------------------------------------------------
# SumWeightedMSELoss
# ---------------------------------------------------------------------------

class TestSumWeightedMSELoss:

    def test_basic_returns_scalar(self, loss_data):
        criterion = SumWeightedMSELoss(FP_LABELS, weight_label='flux')
        loss = criterion(loss_data["pred"], loss_data["target"], loss_data["fp_batch"])
        assert loss.shape == ()
        assert loss.isfinite()

    def test_normalize_batch(self, loss_data):
        criterion = SumWeightedMSELoss(FP_LABELS, weight_label='flux',
                                        normalize_fn='batch')
        loss = criterion(loss_data["pred"], loss_data["target"], loss_data["fp_batch"])
        assert loss.isfinite()

    def test_normalize_by_mean(self, loss_data):
        # Compute mean flux sum over the fixture batch
        flux_sum_mean = loss_data["flux"].sum(dim=(1, 2)).mean()
        criterion = SumWeightedMSELoss(
            FP_LABELS, weight_label='flux',
            normalize_fn=normalize_by_mean(flux_sum_mean.item())
        )
        loss = criterion(loss_data["pred"], loss_data["target"], loss_data["fp_batch"])
        assert loss.isfinite()

    def test_normalize_fn_unknown_string_raises(self):
        with pytest.raises(ValueError, match="Unknown normalize_fn"):
            SumWeightedMSELoss(FP_LABELS, weight_label='flux',
                                normalize_fn='unknown')

    def test_transform_scale(self, loss_data):
        criterion = SumWeightedMSELoss(FP_LABELS, weight_label='flux',
                                        transform_fn=scale, w=1e-9)
        loss = criterion(loss_data["pred"], loss_data["target"], loss_data["fp_batch"])
        assert loss.isfinite()

    def test_nan_mask_applied(self, loss_data):
        c_no_mask = SumWeightedMSELoss(FP_LABELS, weight_label='flux')
        c_mask    = SumWeightedMSELoss(FP_LABELS, weight_label='flux',
                                        nan_mask_label='fp_nan_mask')
        l_no_mask = c_no_mask(loss_data["pred"], loss_data["target"],
                               loss_data["fp_batch"])
        l_mask    = c_mask(loss_data["pred"], loss_data["target"],
                            loss_data["fp_batch"])
        assert not torch.allclose(l_no_mask, l_mask)


# ---------------------------------------------------------------------------
# MSEPlusSumLoss
# ---------------------------------------------------------------------------

class TestMSEPlusSumLoss:

    def test_default_ones_no_fp_batch(self, loss_data):
        """weight_label='ones' requires no fp_batch."""
        criterion = MSEPlusSumLoss(alpha=1.0)
        loss = criterion(loss_data["pred"], loss_data["target"])
        assert loss.shape == ()
        assert loss.isfinite()

    def test_alpha_zero_recovers_mse(self, loss_data):
        """alpha=0 collapses the integral term → identical to plain MSELoss."""
        criterion_combo = MSEPlusSumLoss(alpha=0.0)
        criterion_plain = MSELoss()
        loss_combo = criterion_combo(loss_data["pred"], loss_data["target"])
        loss_plain = criterion_plain(loss_data["pred"], loss_data["target"])
        assert torch.allclose(loss_combo, loss_plain, atol=1e-6)

    def test_alpha_scaling(self, loss_data):
        """Integral-error term scales linearly with alpha."""
        c1 = MSEPlusSumLoss(alpha=1.0)
        c2 = MSEPlusSumLoss(alpha=2.0)
        c0 = MSEPlusSumLoss(alpha=0.0)

        l1 = c1(loss_data["pred"], loss_data["target"])
        l2 = c2(loss_data["pred"], loss_data["target"])
        l0 = c0(loss_data["pred"], loss_data["target"])  # pure MSE

        # integral_mse = l1 - l0; at alpha=2 it should be 2*(l1-l0)
        assert torch.allclose(l2 - l0, 2 * (l1 - l0), rtol=1e-5)

    def test_flux_weight_label(self, loss_data):
        criterion = MSEPlusSumLoss(FP_LABELS, weight_label='flux', alpha=1.0)
        loss = criterion(loss_data["pred"], loss_data["target"], loss_data["fp_batch"])
        assert loss.isfinite()

    def test_normalize_batch(self, loss_data):
        criterion = MSEPlusSumLoss(FP_LABELS, weight_label='flux',
                                    normalize_fn='batch', alpha=1.0)
        loss = criterion(loss_data["pred"], loss_data["target"], loss_data["fp_batch"])
        assert loss.isfinite()

    def test_normalize_by_mean(self, loss_data):
        flux_integral_mean = loss_data["flux"].sum(dim=(1, 2)).mean()
        criterion = MSEPlusSumLoss(
            FP_LABELS, weight_label='flux',
            normalize_fn=normalize_by_mean(flux_integral_mean.item()),
            alpha=1.0
        )
        loss = criterion(loss_data["pred"], loss_data["target"], loss_data["fp_batch"])
        assert loss.isfinite()

    def test_nan_mask_explicit(self, loss_data):
        criterion = MSEPlusSumLoss(FP_LABELS, weight_label='flux', alpha=1.0)
        l_no_mask = criterion(loss_data["pred"], loss_data["target"],
                               loss_data["fp_batch"])
        l_masked  = criterion(loss_data["pred"], loss_data["target"],
                               loss_data["fp_batch"],
                               nan_mask=loss_data["nan_mask"])
        assert not torch.allclose(l_no_mask, l_masked)

    def test_nan_mask_from_fp_batch_equals_explicit(self, loss_data):
        c_explicit = MSEPlusSumLoss(FP_LABELS, weight_label='flux', alpha=1.0)
        c_label    = MSEPlusSumLoss(FP_LABELS, weight_label='flux', alpha=1.0,
                                     nan_mask_label='fp_nan_mask')
        l_explicit = c_explicit(loss_data["pred"], loss_data["target"],
                                 loss_data["fp_batch"],
                                 nan_mask=loss_data["nan_mask"])
        l_label    = c_label(loss_data["pred"], loss_data["target"],
                              loss_data["fp_batch"])
        assert torch.allclose(l_explicit, l_label)

    def test_math_integral_term(self, loss_data):
        """Manual integral computation matches criterion output."""
        pred, target = loss_data["pred"], loss_data["target"]
        flux = loss_data["flux"]

        criterion = MSEPlusSumLoss(FP_LABELS, weight_label='flux', alpha=1.0)
        loss = criterion(pred, target, loss_data["fp_batch"])

        # pixel MSE
        mse = ((pred - target) ** 2).mean()

        # per-sample flux-weighted integrals
        integral_pred   = (pred   * flux).sum(dim=(1, 2))  # (B,)
        integral_target = (target * flux).sum(dim=(1, 2))  # (B,)
        integral_mse    = ((integral_pred - integral_target) ** 2).mean()

        expected = mse + integral_mse
        assert torch.allclose(loss, expected, atol=1e-6)