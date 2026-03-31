import os
import numpy as np
import pandas as pd
import xarray as xr

from model.data.load_data import remove_duplicates, _wrap_longitudes, _get_release_idxs

SAMPLE_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data")


# ── Aux function tests: no file I/O ──────────────────────────────────────────

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


def test_wrap_longitudes_converts_0_360():
    """_wrap_longitudes converts 0-360 lons to -180-180 and sorts the result."""
    lons_0_360 = np.array([0.0, 90.0, 180.0, 270.0, 350.0])
    ds = xr.Dataset(
        {"data": (["lat", "lon"], np.ones((3, 5)))},
        coords={"lat": np.linspace(-10.0, 10.0, 3), "lon": lons_0_360},
    )

    result = _wrap_longitudes(ds)

    assert float(result.lon.max()) <= 180.0, "Max lon should be <= 180 after wrapping"
    assert float(result.lon.min()) >= -180.0, "Min lon should be >= -180 after wrapping"
    # coords must remain monotonically increasing after sortby
    assert np.all(np.diff(result.lon.values) > 0), "Lons should be sorted after wrapping"


def test_wrap_longitudes_no_op_when_already_wrapped():
    """_wrap_longitudes leaves a -180-180 dataset unchanged."""
    lons = np.linspace(-60.0, -40.0, 5)
    ds = xr.Dataset(
        {"data": (["lat", "lon"], np.ones((3, 5)))},
        coords={"lat": np.linspace(-10.0, 10.0, 3), "lon": lons},
    )

    result = _wrap_longitudes(ds)

    np.testing.assert_array_equal(result.lon.values, lons)

def test_get_release_idxs_known_position():
    # release at lat=-5, lon=-50 — we know exactly which grid index that is
    lats = np.linspace(-15.0, 5.0, 21)   # index 10 = -5.0
    lons = np.linspace(-60.0, -40.0, 21)  # index 10 = -50.0
    fp = xr.Dataset(
        {"release_lat": ("time", [-5.0]), "release_lon": ("time", [-50.0])},
        coords={"time": pd.date_range("2016-01-01", periods=1),
                "lat": lats, "lon": lons}
    )
    idxs = _get_release_idxs(fp)
    assert idxs[0, 0] == 10   # lat index
    assert idxs[0, 1] == 10   # lon index

# ── Dataset format tests ──────────────────────────────────────────────────────

class TestDatasetFormat:
    """
    Check that each fixture dataset has the expected dimensions, variables,
    and basic sanity properties. Accepts either lat/lon or latitude/longitude
    names for spatial dimensions, since raw files may not have been renamed yet.
    """

    def _has_dim(self, ds, *names):
        """Return True if at least one of names is a dimension of ds."""
        return any(n in ds.dims for n in names)

    def test_fp_format(self, fp_ds):
        assert self._has_dim(fp_ds, "time"), "Expected time dimension"
        assert self._has_dim(fp_ds, "lat", "latitude"), "Expected lat/latitude dimension"
        assert self._has_dim(fp_ds, "lon", "longitude"), "Expected lon/longitude dimension"
        assert "fp" in fp_ds.data_vars, "Expected 'fp' variable"
        assert "release_lat" in fp_ds.data_vars, "Expected 'release_lat' variable"
        assert "release_lon" in fp_ds.data_vars, "Expected 'release_lon' variable"
        assert float(fp_ds.fp.min()) >= 0, "Footprint values should be non-negative"

    def test_met_format(self, met_ds):
        assert self._has_dim(met_ds, "time"), "Expected time dimension"
        assert self._has_dim(met_ds, "lat", "latitude"), "Expected lat/latitude dimension"
        assert self._has_dim(met_ds, "lon", "longitude"), "Expected lon/longitude dimension"
        assert self._has_dim(met_ds, "model_level_number"), "Expected model_level_number dimension"
        assert "x_wind" in met_ds.data_vars, "Expected 'x_wind' variable"
        assert "y_wind" in met_ds.data_vars, "Expected 'y_wind' variable"

    def test_topog_format(self, topog_ds):
        assert self._has_dim(topog_ds, "lat", "latitude"), "Expected lat/latitude dimension"
        assert self._has_dim(topog_ds, "lon", "longitude"), "Expected lon/longitude dimension"
        assert "surface_altitude" in topog_ds.data_vars, "Expected 'surface_altitude' variable"

    def test_landcover_format(self, landcover_ds):
        assert self._has_dim(landcover_ds, "lat", "latitude"), "Expected lat/latitude dimension"
        assert self._has_dim(landcover_ds, "lon", "longitude"), "Expected lon/longitude dimension"
        assert self._has_dim(landcover_ds, "pseudo_level"), "Expected pseudo_level dimension"
