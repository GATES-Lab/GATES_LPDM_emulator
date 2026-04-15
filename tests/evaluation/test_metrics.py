"""Tests for gates.evaluation.metrics, all using dummy data

Shared fixtures
---------------
true_arr  (3, 10, 10)  — one 4×4 blob of 1.0 per sample on a zero background:
    sample 0: rows 1-4, cols 1-4  (top-left)
    sample 1: rows 3-6, cols 3-6  (centre)
    sample 2: rows 5-8, cols 5-8  (bottom-right)

pred_arr  (3, 10, 10)  — predictions with simple, computable errors:
    sample 0: identical to true[0]               → zero error
    sample 1: blob shifted 1 row down (rows 4-7) → 8 mismatched cells
    sample 2: same blob, value 2.0               → amplitude error +1

ignore_mask  (3, 10, 10)  — one large excluded block per sample (True = ignore):
    sample 0: rows 7-9, all cols  (30 cells — below blob)
    sample 1: all rows, cols 7-9  (30 cells — right of blob)
    sample 2: rows 7-9, all cols  (30 cells — partially overlaps blob rows 7-8)

Analytical reference values (no mask, 100 cells per sample)
------------------------------------------------------------
            bias        mae         mse         nmae
sample 0    0           0           0           0/16   = 0
sample 1    0           8/100=0.08  0.08        8/16   = 0.5
sample 2    16/100=0.16 16/100=0.16 0.16        16/16  = 1.0
mean        0.16/3      0.08        0.08        0.5

IoU (threshold=0):
    sample 0: I=16, U=16  → 1.0
    sample 1: I=12, U=20  → 0.6   (overlap rows 4-6 cols 3-6 = 12 cells)
    sample 2: I=16, U=16  → 1.0   (pred=2>0, same location)
    mean: 2.6/3
"""
import numpy as np
import pytest

from gates.evaluation.metrics import (
    bias,
    corrcoef,
    iou,
    mae,
    mse,
    nmae,
    compute_footprint_metrics,
    compute_metrics_by_threshold,
    compute_mfs_metrics,
    calculate_mfs,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N, H, W = 3, 10, 10
BLOB_CELLS = 16          # cells active per blob (4×4)
TOTAL_CELLS = H * W      # 100
NO_CORR_METRICS = ["iou", "mse", "mae", "nmae", "bias"]


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def true_arr():
    """(3, 10, 10) — 4×4 footprint blobs of value 1.0, zeros elsewhere."""
    arr = np.zeros((N, H, W), dtype=np.float64)
    arr[0, 1:5, 1:5] = 1.0   # top-left blob
    arr[1, 3:7, 3:7] = 1.0   # centre blob
    arr[2, 5:9, 5:9] = 1.0   # bottom-right blob
    return arr


@pytest.fixture(scope="module")
def pred_arr():
    """(3, 10, 10) — predictions with analytically verifiable errors.

    sample 0: identical to true[0] → zero error everywhere.
    sample 1: blob at rows 4-7 cols 3-6 (1 row below true[1]) → 8 mismatched cells.
    sample 2: blob at same location as true[2] with value 2.0 → amplitude error +1.
    """
    arr = np.zeros((N, H, W), dtype=np.float64)
    arr[0, 1:5, 1:5] = 1.0
    arr[1, 4:8, 3:7] = 1.0
    arr[2, 5:9, 5:9] = 2.0
    return arr


@pytest.fixture(scope="module")
def ignore_mask():
    """(3, 10, 10) ignore mask — one large contiguous block per sample.

    sample 0: rows 7-9  (30 cells, below blob at rows 1-4).
    sample 1: cols 7-9  (30 cells, right of blob at cols 3-6).
    sample 2: rows 7-9  (30 cells, partially overlaps blob at rows 5-8 → rows 7-8 excluded).
    """
    mask = np.zeros((N, H, W), dtype=bool)
    mask[0, 7:, :] = True
    mask[1, :, 7:] = True
    mask[2, 7:, :] = True
    return mask


""""
For each metric function, we test:
- known values for the dummy true/pred arrays, with and without the ignore mask
- reduce="mean" gives the expected average of the per-sample scores
- reduce=None gives an array of shape (N,) with the per-sample scores
- perfect match (true=pred) gives zero error for bias, mse, mae, nmae, and IoU=1.0
- symmetry of mae (swapping true and pred gives the same result)
"""
# ---------------------------------------------------------------------------
# bias
# ---------------------------------------------------------------------------

class TestBias:

    def test_known_values_per_sample(self, true_arr, pred_arr):
        """Per-sample biases match analytical expectations."""
        scores = bias(true_arr, pred_arr, reduce=None)
        assert scores[0] == pytest.approx(0.0, abs=1e-9)
        assert scores[1] == pytest.approx(0.0, abs=1e-9)
        assert scores[2] == pytest.approx(BLOB_CELLS / TOTAL_CELLS, rel=1e-9)

    def test_reduce_mean(self, true_arr, pred_arr):
        expected = (0.0 + 0.0 + BLOB_CELLS / TOTAL_CELLS) / N
        assert bias(true_arr, pred_arr) == pytest.approx(expected, rel=1e-6)

    def test_reduce_none_shape(self, true_arr, pred_arr):
        assert bias(true_arr, pred_arr, reduce=None).shape == (N,)

    def test_perfect_match(self, true_arr):
        assert bias(true_arr, true_arr) == pytest.approx(0.0, abs=1e-9)

    def test_reduce_none_is_consistent_with_mean(self, true_arr, pred_arr):
        scores = bias(true_arr, pred_arr, reduce=None)
        assert bias(true_arr, pred_arr) == pytest.approx(float(np.mean(scores)), rel=1e-9)

    def test_with_ignore_mask(self, true_arr, pred_arr, ignore_mask):
        """Masking rows 7-9 of sample 2 halves the visible blob error."""
        # sample 2 blob spans rows 5-8; after masking rows 7-9, only rows 5-6
        # remain visible (8 cells), over 70 valid cells total.
        visible_blob_cells = 8   # rows 5-6 of the 4×4 blob
        valid_cells = TOTAL_CELLS - 30

        scores = bias(true_arr, pred_arr, ignore_mask=ignore_mask, reduce=None)
        assert scores[0] == pytest.approx(0.0, abs=1e-9)
        assert scores[1] == pytest.approx(0.0, abs=1e-9)
        assert scores[2] == pytest.approx(visible_blob_cells / valid_cells, rel=1e-6)

    def test_nonzero_excludes_zero_cells(self):
        """nonzero=True should drop cells where either array is zero."""
        t = np.array([[[0.0, 1.0, 1.0, 2.0]]])   # (1, 1, 4)
        p = np.array([[[1.0, 2.0, 0.0, 3.0]]])
        # Without nonzero: 4 cells → mean(1, 1, -1, 1) = 0.5
        # With nonzero   : cells 0 (t=0) and 2 (p=0) dropped → cells 1,3 → mean(1, 1) = 1.0
        assert bias(t, p, nonzero=False) == pytest.approx(0.5, abs=1e-9)
        assert bias(t, p, nonzero=True) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# mse
# ---------------------------------------------------------------------------

class TestMse:

    def test_known_values_per_sample(self, true_arr, pred_arr):
        """Squared errors are 0, 1 (binary blobs), so MSE = fraction of
        mismatched cells."""
        scores = mse(true_arr, pred_arr, reduce=None)
        assert scores[0] == pytest.approx(0.0, abs=1e-9)
        assert scores[1] == pytest.approx(8 / TOTAL_CELLS, rel=1e-6)
        assert scores[2] == pytest.approx(BLOB_CELLS / TOTAL_CELLS, rel=1e-6)

    def test_reduce_mean(self, true_arr, pred_arr):
        expected = (0.0 + 8 / TOTAL_CELLS + BLOB_CELLS / TOTAL_CELLS) / N
        assert mse(true_arr, pred_arr) == pytest.approx(expected, rel=1e-6)

    def test_perfect_match(self, true_arr):
        assert mse(true_arr, true_arr) == pytest.approx(0.0, abs=1e-9)

    def test_reduce_none_shape(self, true_arr, pred_arr):
        assert mse(true_arr, pred_arr, reduce=None).shape == (N,)

    def test_log_transform(self):
        """log-MSE: error = (log10(p) - log10(t))^2 = (log10(2))^2 when p = 2t."""
        t = np.ones((1, H, W))
        p = np.full((1, H, W), 2.0)
        expected = np.log10(2.0) ** 2
        assert mse(t, p, log_transform=True, nonzero=True) == pytest.approx(expected, rel=1e-6)

    def test_with_ignore_mask(self, true_arr, pred_arr, ignore_mask):
        """Mask reduces the visible error for sample 2."""
        score_no_mask = mse(true_arr, pred_arr)
        score_with_mask = mse(true_arr, pred_arr, ignore_mask=ignore_mask)
        assert score_with_mask < score_no_mask


# ---------------------------------------------------------------------------
# mae
# ---------------------------------------------------------------------------

class TestMae:

    def test_known_values_per_sample(self, true_arr, pred_arr):
        scores = mae(true_arr, pred_arr, reduce=None)
        assert scores[0] == pytest.approx(0.0, abs=1e-9)
        assert scores[1] == pytest.approx(8 / TOTAL_CELLS, rel=1e-6)
        assert scores[2] == pytest.approx(BLOB_CELLS / TOTAL_CELLS, rel=1e-6)

    def test_reduce_mean(self, true_arr, pred_arr):
        expected = (0.0 + 8 / TOTAL_CELLS + BLOB_CELLS / TOTAL_CELLS) / N
        assert mae(true_arr, pred_arr) == pytest.approx(expected, rel=1e-6)

    def test_perfect_match(self, true_arr):
        assert mae(true_arr, true_arr) == pytest.approx(0.0, abs=1e-9)

    def test_reduce_none_shape(self, true_arr, pred_arr):
        assert mae(true_arr, pred_arr, reduce=None).shape == (N,)

    def test_with_ignore_mask(self, true_arr, pred_arr, ignore_mask):
        score_no_mask = mae(true_arr, pred_arr)
        score_with_mask = mae(true_arr, pred_arr, ignore_mask=ignore_mask)
        assert score_with_mask < score_no_mask

    def test_symmetry(self, true_arr, pred_arr):
        """MAE is symmetric: swapping true and pred gives the same result."""
        assert mae(true_arr, pred_arr) == pytest.approx(mae(pred_arr, true_arr), rel=1e-9)


# ---------------------------------------------------------------------------
# nmae
# ---------------------------------------------------------------------------

class TestNmae:

    def test_known_values_per_sample(self, true_arr, pred_arr):
        """nmae = sum(|pred - true|) / sum(true).

        sample 0: 0  / 16 = 0
        sample 1: 8  / 16 = 0.5   (8 cells with error 1)
        sample 2: 16 / 16 = 1.0   (16 cells with error 1; true_sum=16)
        """
        scores = nmae(true_arr, pred_arr, reduce=None)
        assert scores[0] == pytest.approx(0.0, abs=1e-9)
        assert scores[1] == pytest.approx(0.5, rel=1e-6)
        assert scores[2] == pytest.approx(1.0, rel=1e-6)

    def test_reduce_mean(self, true_arr, pred_arr):
        assert nmae(true_arr, pred_arr) == pytest.approx(0.5, rel=1e-6)

    def test_perfect_match(self, true_arr):
        assert nmae(true_arr, true_arr) == pytest.approx(0.0, abs=1e-9)

    def test_reduce_none_shape(self, true_arr, pred_arr):
        assert nmae(true_arr, pred_arr, reduce=None).shape == (N,)

    def test_all_zero_true_returns_nan(self):
        """Division by zero when sum(true)=0 should produce NaN, not a crash."""
        t = np.zeros((1, H, W))
        p = np.ones((1, H, W))
        score = nmae(t, p, reduce=None)
        assert np.isnan(score[0])


# ---------------------------------------------------------------------------
# corrcoef
# ---------------------------------------------------------------------------

class TestCorrcoef:

    def test_identical_arrays_give_perfect_correlation(self, true_arr):
        """corrcoef(x, x) = 1 for any array with non-zero variance."""
        assert corrcoef(true_arr, true_arr) == pytest.approx(1.0, abs=1e-9)

    def test_amplitude_scaled_pred_gives_perfect_correlation(self, true_arr, pred_arr):
        """sample 2: pred = 2 * true (same support) → corrcoef = 1."""
        scores = corrcoef(true_arr, pred_arr, reduce=None)
        assert scores[2] == pytest.approx(1.0, abs=1e-9)

    def test_reduce_none_shape(self, true_arr, pred_arr):
        assert corrcoef(true_arr, pred_arr, reduce=None).shape == (N,)

    def test_all_samples_in_range(self, true_arr, pred_arr):
        scores = corrcoef(true_arr, pred_arr, reduce=None)
        finite = scores[np.isfinite(scores)]
        assert np.all(finite >= -1.0 - 1e-9)
        assert np.all(finite <=  1.0 + 1e-9)

    def test_anticorrelated(self):
        t = np.linspace(0.0, 1.0, TOTAL_CELLS).reshape(1, H, W)
        p = np.linspace(1.0, 0.0, TOTAL_CELLS).reshape(1, H, W)
        assert corrcoef(t, p) == pytest.approx(-1.0, abs=1e-9)

    def test_log_transform_preserves_rank(self):
        """log-corrcoef = 1 when pred = k * true for k > 0 (same linear rank in log space)."""
        t = np.logspace(0.0, 1.0, TOTAL_CELLS).reshape(1, H, W)
        p = t * 3.0
        assert corrcoef(t, p, log_transform=True) == pytest.approx(1.0, abs=1e-9)

    def test_reduce_none_consistent_with_mean(self, true_arr, pred_arr):
        scores = corrcoef(true_arr, pred_arr, reduce=None)
        mean_val = float(np.nanmean(scores))
        assert corrcoef(true_arr, pred_arr) == pytest.approx(mean_val, rel=1e-9)


# ---------------------------------------------------------------------------
# iou
# ---------------------------------------------------------------------------

class TestIou:

    def test_known_values_per_sample(self, true_arr, pred_arr):
        """
        sample 0: perfect overlap (identical blobs)       → IoU = 1.0
        sample 1: 12-cell intersection, 20-cell union     → IoU = 12/20 = 0.6
        sample 2: pred blob same location, value 2 > 0   → IoU = 1.0
        """
        scores = iou(true_arr, pred_arr, reduce=None)
        assert scores[0] == pytest.approx(1.0, abs=1e-9)
        assert scores[1] == pytest.approx(12 / 20, rel=1e-6)
        assert scores[2] == pytest.approx(1.0, abs=1e-9)

    def test_reduce_mean(self, true_arr, pred_arr):
        expected = (1.0 + 12 / 20 + 1.0) / N
        assert iou(true_arr, pred_arr) == pytest.approx(expected, rel=1e-6)

    def test_reduce_none_shape(self, true_arr, pred_arr):
        assert iou(true_arr, pred_arr, reduce=None).shape == (N,)

    def test_no_overlap(self):
        """Disjoint active regions → IoU = 0."""
        t = np.zeros((1, H, W))
        p = np.zeros((1, H, W))
        t[0, :, :5] = 1.0   # left half
        p[0, :, 5:] = 1.0   # right half
        assert iou(t, p) == pytest.approx(0.0, abs=1e-9)

    def test_empty_both_arrays_returns_nan(self):
        """Union = 0 → IoU is undefined → NaN, not a crash."""
        t = np.zeros((1, H, W))
        p = np.zeros((1, H, W))
        score = iou(t, p, reduce=None)
        assert np.isnan(score[0])

    def test_threshold_boundary(self):
        """Cells at exactly the threshold are NOT counted as active."""
        t = np.full((1, H, W), 0.5)
        p = np.full((1, H, W), 0.5)
        # threshold=0.5 → 0.5 is not > 0.5 → both empty → NaN
        assert np.isnan(iou(t, p, threshold=0.5, reduce=None)[0])
        # threshold=0.4 → 0.5 > 0.4 → both fully active → IoU = 1.0
        assert iou(t, p, threshold=0.4) == pytest.approx(1.0, abs=1e-9)

    def test_ignore_mask_changes_iou(self):
        """Masking the pred-only region raises IoU from 0.5 to 1.0."""
        t = np.zeros((1, H, W))
        p = np.ones((1, H, W))
        t[0, :5, :] = 1.0   # top half active in true; pred = all ones

        # Without mask: I=50, U=100 → IoU = 0.5
        assert iou(t, p) == pytest.approx(0.5, abs=1e-9)

        mask = np.zeros((H, W), dtype=bool)
        mask[5:, :] = True  # hide the pred-only bottom half
        # After mask: only top half visible; both true and pred active → IoU = 1.0
        assert iou(t, p, ignore_mask=mask) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# compute_footprint_metrics
# ---------------------------------------------------------------------------

class TestComputeFootprintMetrics:

    def test_returns_all_default_keys(self, true_arr, pred_arr):
        result = compute_footprint_metrics(true_arr, pred_arr, metrics=NO_CORR_METRICS)
        expected_keys = set(NO_CORR_METRICS)
        assert set(result.keys()) == expected_keys

    def test_values_match_individual_functions(self, true_arr, pred_arr):
        result = compute_footprint_metrics(true_arr, pred_arr, metrics=NO_CORR_METRICS)
        assert result["bias"] == pytest.approx(bias(true_arr, pred_arr), rel=1e-6)
        assert result["mse"] == pytest.approx(mse(true_arr, pred_arr), rel=1e-6)
        assert result["mae"] == pytest.approx(mae(true_arr, pred_arr), rel=1e-6)
        assert result["nmae"] == pytest.approx(nmae(true_arr, pred_arr), rel=1e-6)
        assert result["iou"] == pytest.approx(iou(true_arr, pred_arr), rel=1e-6)
        
        # the dummy arrays do not have correlation and therefore corrcoeff returns nan 
        # assert result["corrcoef"] == pytest.approx(
        #     corrcoef(true_arr, pred_arr, log_transform=False), rel=1e-6
        # )
        # assert result["corrcoef_log"] == pytest.approx(
        #     corrcoef(true_arr, pred_arr, log_transform=True), rel=1e-6
        # )

    def test_subset_of_metrics(self, true_arr, pred_arr):
        result = compute_footprint_metrics(true_arr, pred_arr, metrics=["mse", "mae"])
        assert set(result.keys()) == {"mse", "mae"}

    def test_invalid_metric_raises(self, true_arr, pred_arr):
        with pytest.raises(ValueError, match="Unsupported metric"):
            compute_footprint_metrics(true_arr, pred_arr, metrics=["not_a_metric"])

    def test_with_ignore_mask(self, true_arr, pred_arr, ignore_mask):
        unmasked = compute_footprint_metrics(true_arr, pred_arr, metrics=NO_CORR_METRICS)
        masked = compute_footprint_metrics(true_arr, pred_arr, ignore_mask=ignore_mask, metrics=NO_CORR_METRICS)
        # All error metrics should be lower or equal when masking partial errors
        assert masked["mae"] <= unmasked["mae"] + 1e-9
        assert masked["mse"] <= unmasked["mse"] + 1e-9


# ---------------------------------------------------------------------------
# compute_metrics_by_threshold
# ---------------------------------------------------------------------------

class TestComputeMetricsByThreshold:

    def test_bin_keys_from_threshold_list(self, true_arr, pred_arr):
        result = compute_metrics_by_threshold(true_arr, pred_arr, thresholds=[0.5, 1.5], metrics=NO_CORR_METRICS)
        assert set(result.keys()) == {"-inf_to_0.5", "0.5_to_1.5", "1.5_to_inf"}

    def test_single_scalar_threshold(self, true_arr, pred_arr):
        result = compute_metrics_by_threshold(true_arr, pred_arr, thresholds=0.5, metrics=NO_CORR_METRICS)
        assert set(result.keys()) == {"-inf_to_0.5", "0.5_to_inf"}

    def test_each_bin_contains_required_keys(self, true_arr, pred_arr):
        result = compute_metrics_by_threshold(true_arr, pred_arr, thresholds=[0.5], metrics=NO_CORR_METRICS)
        for bin_scores in result.values():
            for key in ("mse", "mae", "nmae", "iou", "threshold_range"):
                assert key in bin_scores

    def test_threshold_range_values(self, true_arr, pred_arr):
        result = compute_metrics_by_threshold(true_arr, pred_arr, thresholds=[0.5, 1.5], metrics=NO_CORR_METRICS)
        assert result["-inf_to_0.5"]["threshold_range"] == (-float("inf"), 0.5)
        assert result["0.5_to_1.5"]["threshold_range"] == (0.5, 1.5)
        assert result["1.5_to_inf"]["threshold_range"] == (1.5, float("inf"))

    def test_invalid_thresholds_raise(self, true_arr, pred_arr):
        with pytest.raises(ValueError):
            compute_metrics_by_threshold(true_arr, pred_arr, thresholds="bad")


# ---------------------------------------------------------------------------
# compute_mfs_metrics
# ---------------------------------------------------------------------------

class TestComputeMfsMetrics:

    def test_perfect_match(self):
        ts = np.linspace(1.0, 3.0, 50)
        result = compute_mfs_metrics(ts, ts)
        assert result["corrcoef"] == pytest.approx(1.0, abs=1e-9)
        assert result["mean_absolute_error"] == pytest.approx(0.0, abs=1e-9)
        assert result["mean_bias"] == pytest.approx(0.0, abs=1e-9)
        assert result["r2_score"] == pytest.approx(1.0, abs=1e-9)
        assert result["true_mean"] == pytest.approx(np.mean(ts), rel=1e-9)
        assert result["predicted_mean"] == pytest.approx(np.mean(ts), rel=1e-9)

    def test_constant_offset(self):
        """Pred = true + 0.5: bias and MAE are both 0.5; r2 < 1."""
        ts = np.array([1.0, 2.0, 3.0, 4.0])
        ps = ts + 0.5
        result = compute_mfs_metrics(ts, ps)
        assert result["mean_bias"] == pytest.approx(0.5, abs=1e-9)
        assert result["mean_absolute_error"] == pytest.approx(0.5, abs=1e-9)
        assert result["true_mean"] == pytest.approx(2.5, abs=1e-9)
        assert result["predicted_mean"] == pytest.approx(3.0, abs=1e-9)
        assert result["corrcoef"] == pytest.approx(1.0, abs=1e-9)

    def test_anticorrelated(self):
        ts = np.linspace(0.0, 1.0, 50)
        ps = np.linspace(1.0, 0.0, 50)
        result = compute_mfs_metrics(ts, ps)
        assert result["corrcoef"] == pytest.approx(-1.0, abs=1e-9)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="[Ss]hape"):
            compute_mfs_metrics(np.ones(5), np.ones(6))

    def test_2d_array_raises(self):
        with pytest.raises(ValueError):
            compute_mfs_metrics(np.ones((5, 5)), np.ones((5, 5)))

    def test_returns_all_expected_keys(self):
        ts = np.linspace(1.0, 2.0, 20)
        result = compute_mfs_metrics(ts, ts)
        expected_keys = {
            "corrcoef", "mean_absolute_error", "mean_bias",
            "true_mean", "predicted_mean", "r2_score",
        }
        assert set(result.keys()) == expected_keys


# ---------------------------------------------------------------------------
# calculate_mfs
# ---------------------------------------------------------------------------

class TestCalculateMfs:

    def test_uniform_flux_sums_to_fp_total(self):
        """fp * uniform(1) summed = sum(fp) per sample."""
        fp = np.ones((N, H, W)) * 2.0
        fluxes = np.ones((N, H, W))
        result = calculate_mfs(fp, fluxes)
        expected = H * W * 2.0
        assert result == pytest.approx([expected] * N, rel=1e-9)

    def test_zero_flux(self):
        fp = np.ones((N, H, W))
        fluxes = np.zeros((N, H, W))
        result = calculate_mfs(fp, fluxes)
        assert result == pytest.approx([0.0] * N, abs=1e-9)

    def test_transform_factor_scales_output(self):
        fp = np.ones((1, H, W))
        fluxes = np.ones((1, H, W))
        result = calculate_mfs(fp, fluxes, transform_factor=2.0)
        assert result == pytest.approx([2.0 * H * W], rel=1e-9)

    def test_mfs_with_blob_footprints(self, true_arr):
        """sum(blob * uniform_flux) = number of active blob cells."""
        fluxes = np.ones((N, H, W))
        result = calculate_mfs(true_arr, fluxes)
        # Each sample has exactly BLOB_CELLS active cells, all value 1.0
        assert result == pytest.approx([float(BLOB_CELLS)] * N, rel=1e-9)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            calculate_mfs(np.ones((2, H, W)), np.ones((3, H, W)))