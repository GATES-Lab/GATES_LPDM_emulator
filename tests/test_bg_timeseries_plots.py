"""Tests for save_bg_timeseries_plots (single-class background index-series figure)."""

import numpy as np

from gates.training.training_helperfuns import save_bg_timeseries_plots


def _make_series(n_times, seed=0):
    rng = np.random.default_rng(seed)
    true = rng.normal(size=n_times)
    pred = true + 0.1 * rng.normal(size=n_times)
    return true, pred


def test_single_class_plot(tmp_path):
    true, pred = _make_series(200)
    out = save_bg_timeseries_plots(0, true, pred, tmp_path, "testmodel")
    assert out.exists()
    assert out.name == "testmodel_bg_timeseries_0.png"


def test_custom_window_settings(tmp_path):
    """n_windows is honoured (e.g. a single window over all samples)."""
    true, pred = _make_series(300)
    out = save_bg_timeseries_plots(1, true, pred, tmp_path, "testmodel", n_windows=1)
    assert out.exists()


def test_fewer_samples_than_windows_does_not_crash(tmp_path):
    """More windows than samples still plots (empty windows are hidden)."""
    true, pred = _make_series(3)
    out = save_bg_timeseries_plots(2, true, pred, tmp_path, "testmodel", n_windows=4)
    assert out.exists()


def test_2d_column_inputs_are_flattened(tmp_path):
    """(N, 1)-shaped arrays (as produced by the model) are accepted."""
    true, pred = _make_series(50)
    out = save_bg_timeseries_plots(3, true.reshape(-1, 1), pred.reshape(-1, 1),
                                   tmp_path, "testmodel")
    assert out.exists()
