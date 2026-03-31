import os
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from model.data.load_data import remove_duplicates

SAMPLE_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data")


# ── Test 1: no files, pure synthetic data ────────────────────────────────────

def test_remove_duplicates_synthetic():
    """remove_duplicates drops repeated coordinate values along a dimension."""
    times = pd.date_range("2016-01-01", periods=5, freq="6h")
    times_with_dup = times.append(times[:2])          # 5 + 2 duplicate timestamps
    data = np.random.rand(len(times_with_dup), 4, 4).astype("float32")
    ds = xr.Dataset(
        {"fp": (["time", "lat", "lon"], data)},
        coords={
            "time": times_with_dup,
            "lat": np.linspace(-10.0, 0.0, 4),
            "lon": np.linspace(-50.0, -40.0, 4),
        },
    )

    result = remove_duplicates(ds, dim="time")

    assert len(result.time) == 5, "Expected duplicates to be removed, leaving 5 unique timestamps"


# ── Test 2: with sample files ─────────────────────────────────────────────────

def test_fp_sample_file_structure():
    """fp_sample.nc loads with the expected dimensions and variables."""
    fp_path = os.path.join(SAMPLE_DATA_DIR, "fp_sample.nc")
    if not os.path.exists(fp_path):
        pytest.skip("fp_sample.nc not found — run tests/sample_data/generate_sample_data.py first")

    ds = xr.open_dataset(fp_path)

    assert "time" in ds.dims, "Expected 'time' dimension"
    assert "lat" in ds.dims, "Expected 'lat' dimension"
    assert "lon" in ds.dims, "Expected 'lon' dimension"
    assert "fp" in ds.data_vars, "Expected 'fp' variable"
    #assert ds["fp"].dtype == np.float32, "Expected fp to be float32"
