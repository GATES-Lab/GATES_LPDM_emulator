# tests/test_data_summary.py
"""Tests for summarize_loaded_data (gates/data/load_data_helper_funs.py).

The summary is the transparency record of what the (shared) data loader produced: it is
written next to the run outputs and pushed into the W&B run config by train_dual_model.
It must therefore be JSON-serialisable, and it must never read lazy (dask-backed) data
unless statistics are explicitly forced.
"""

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from gates.data.load_data_helper_funs import summarize_loaded_data

N_TIME, N_LAT, N_LON, N_VARS = 6, 4, 5, 3


def _make_split(lazy=False):
    """Tiny synthetic (fp, inputs, bgs, aux) matching load_GATES_data_with_bg's returns."""
    times = pd.date_range("2016-01-01", periods=N_TIME, freq="6h")
    lats = np.linspace(-15.0, 15.0, N_LAT)
    lons = np.linspace(-40.0, 30.0, N_LON)

    fp_values = np.random.rand(N_TIME, N_LAT, N_LON)
    fp_values[0, 0, 0] = np.nan
    fp = xr.Dataset(
        {"fp": (["time", "lat", "lon"], fp_values),
         "release_lat": (["time"], np.random.uniform(-10.0, 10.0, N_TIME)),
         "release_lon": (["time"], np.random.uniform(-30.0, 20.0, N_TIME))},
        coords={"time": times, "lat": lats, "lon": lons},
    )

    inputs = xr.DataArray(
        np.random.rand(N_TIME, N_LAT, N_LON, N_VARS),
        dims=["fp_time", "lat", "lon", "variable"],
        coords={"fp_time": times, "lat": lats, "lon": lons},
        name="inputs",
    )

    bgs = xr.Dataset(
        {name: (["time"], np.random.rand(N_TIME))
         for name in ["summed", "north", "south", "east", "west"]},
        coords={"time": times},
    )

    aux = xr.Dataset(
        {"aux": (["height", "time"], np.random.rand(2, N_TIME))},
        coords={"time": times},
    )

    if lazy:
        fp = fp.chunk({"time": 2})
        inputs = inputs.chunk({"fp_time": 2})
        bgs = bgs.chunk({"time": 2})
        aux = aux.chunk({"time": 2})
    return fp, inputs, bgs, aux


def test_summary_in_memory_has_stats_and_is_json_serialisable():
    fp, inputs, bgs, aux = _make_split()
    summary = summarize_loaded_data(fp, inputs, bgs, aux)

    assert summary["n_times"] == N_TIME
    assert summary["time_range"] == ["2016-01-01T00:00:00", "2016-01-02T06:00:00"]
    assert summary["footprints"]["dims"] == {"time": N_TIME, "lat": N_LAT, "lon": N_LON}
    assert summary["footprints"]["lazy"] is False
    assert summary["footprints"]["stats"]["fp"]["nan_count"] == 1
    assert summary["met_inputs"]["dims"]["variable"] == N_VARS
    assert summary["met_inputs"]["stats"]["nan_count"] == 0
    assert set(summary["backgrounds"]["data_vars"]) == {"summed", "north", "south", "east", "west"}
    assert 0.0 <= summary["backgrounds"]["stats"]["summed"]["min"] <= 1.0
    assert summary["aux_cams"]["dims"] == {"height": 2, "time": N_TIME}

    json.dumps(summary)  # must not raise


def test_summary_lazy_data_not_read_by_default():
    pytest.importorskip("dask")
    fp, inputs, bgs, aux = _make_split(lazy=True)
    summary = summarize_loaded_data(fp, inputs, bgs, aux)

    for component in ["footprints", "met_inputs", "backgrounds", "aux_cams"]:
        assert summary[component]["lazy"] is True
        assert summary[component]["stats"] == "not computed"
    # metadata-only fields are still recorded
    assert summary["n_times"] == N_TIME
    assert summary["footprints"]["dims"]["time"] == N_TIME
    json.dumps(summary)


def test_summary_lazy_stats_can_be_forced():
    pytest.importorskip("dask")
    fp, inputs, bgs, aux = _make_split(lazy=True)
    summary = summarize_loaded_data(fp, inputs, bgs, aux, compute_stats=True)

    assert summary["footprints"]["stats"]["fp"]["nan_count"] == 1
    assert summary["backgrounds"]["stats"]["north"]["nan_count"] == 0
    json.dumps(summary)


def test_summary_stats_can_be_disabled():
    fp, inputs, bgs, aux = _make_split()
    summary = summarize_loaded_data(fp, inputs, bgs, aux, compute_stats=False)
    assert summary["footprints"]["stats"] == "not computed"
    assert summary["met_inputs"]["stats"] == "not computed"


def test_summary_handles_missing_aux():
    fp, inputs, bgs, _ = _make_split()
    summary = summarize_loaded_data(fp, inputs, bgs, None)
    assert summary["aux_cams"] is None
    json.dumps(summary)
