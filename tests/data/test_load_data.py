import os
import pytest
import numpy as np
import pandas as pd
import xarray as xr
from unittest.mock import patch

from model.data.load_data import (
    remove_duplicates,
    _wrap_longitudes,
    _rename_latlon,
    _get_release_idxs,
    _pad_domain,
    cut_satellite_data,
    cut_topog_data,
    LoadBaseSatelliteData,
    LoadSquareSatelliteData,
)

SAMPLE_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data")
TEST_YEAR = 2016
TEST_MONTH = "01"
TEST_DATE = f"{TEST_YEAR}{TEST_MONTH}"
TEST_START_DATE = f"{TEST_YEAR}-{TEST_MONTH}-01"


# ── Aux function tests: no file I/O ──────────────────────────────────────────

def test_remove_duplicates_synthetic():
    """remove_duplicates drops repeated coordinate values along a dimension."""
    times = pd.date_range(TEST_START_DATE, periods=5, freq="6h")
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
        coords={"time": pd.date_range(TEST_START_DATE, periods=1),
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
        # Accept either the new variable names (expected by cut_topog_data) or the raw file's land_cover
        has_new_vars = all(
            v in landcover_ds.data_vars
            for v in ["landcover_type", "land_binary_mask", "landcover_fraction"]
        )
        has_old_var = "land_cover" in landcover_ds.data_vars
        assert has_new_vars or has_old_var, (
            "Expected landcover variables: either 'land_cover' (raw) or "
            "'landcover_type', 'land_binary_mask', 'landcover_fraction' (processed)"
        )


# ── _pad_domain tests (Tier 1 — always runs on inline synthetic data) ─────────

class TestPadDomain:
    """Tests for _pad_domain: verifies NaN/zero/edge fill and index updates."""

    def _make_fp(self, n_lat, n_lon, release_lat_idx, release_lon_idx, fp_value=1.0):
        """Build a minimal 1-timestep fp Dataset with a release at specified indices."""
        lats = np.linspace(-10.0, 10.0, n_lat)
        lons = np.linspace(-10.0, 10.0, n_lon)
        return xr.Dataset(
            {
                "fp": (["time", "lat", "lon"],
                       np.full((1, n_lat, n_lon), fp_value, dtype="float64")),
                "release_lat": ("time", [lats[release_lat_idx]]),
                "release_lon": ("time", [lons[release_lon_idx]]),
            },
            coords={
                "time": pd.date_range(TEST_START_DATE, periods=1),
                "lat": lats,
                "lon": lons,
            },
        )

    def test_no_padding_when_release_at_centre(self):
        n = 10
        fp = self._make_fp(n, n, n // 2, n // 2)
        idxs = _get_release_idxs(fp)
        result, _, _, _ = _pad_domain(fp, fp, idxs, half=4, pad_mode="nans")
        assert result.lat.size == n, "No lat padding expected when release is at centre"
        assert result.lon.size == n, "No lon padding expected when release is at centre"

    def test_pads_south_nans(self):
        n, half = 8, 4
        fp = self._make_fp(n, n, 0, n // 2)     # release at south edge (idx 0)
        idxs = _get_release_idxs(fp)
        result, _, _, new_idxs = _pad_domain(fp, fp, idxs, half, "nans")
        assert result.lat.size > n, "Expected extra lat rows after south padding"
        assert new_idxs[0, 0] == half, "Release lat index should equal half after padding"
        # The newly added southern rows should be NaN-filled
        assert np.all(np.isnan(result.fp.values[0, :half, :])), \
            "Padded southern rows should be NaN"

    def test_pads_south_zeros(self):
        n, half = 8, 4
        fp = self._make_fp(n, n, 0, n // 2)
        idxs = _get_release_idxs(fp)
        result, _, _, _ = _pad_domain(fp, fp, idxs, half, "zeros")
        assert result.lat.size > n
        assert np.all(result.fp.values[0, :half, :] == 0.0), \
            "Padded southern rows should be zero"

    def test_pads_south_edge(self):
        n, half = 8, 4
        fp_val = 3.14
        fp = self._make_fp(n, n, 0, n // 2, fp_value=fp_val)
        idxs = _get_release_idxs(fp)
        result, _, _, new_idxs = _pad_domain(fp, fp, idxs, half, "edge")
        assert result.lat.size > n
        # Edge-padded rows replicate the boundary row value
        boundary_row = result.fp.values[0, new_idxs[0, 0], :]
        for pad_row in range(new_idxs[0, 0]):
            np.testing.assert_array_equal(
                result.fp.values[0, pad_row, :], boundary_row,
                err_msg=f"Padded row {pad_row} should equal the boundary row",
            )

    def test_pads_east_nans(self):
        n, half = 8, 4
        fp = self._make_fp(n, n, n // 2, n - 1)  # release at east edge (idx n-1)
        idxs = _get_release_idxs(fp)
        result, _, _, new_idxs = _pad_domain(fp, fp, idxs, half, "nans")
        assert result.lon.size > n, "Expected extra lon cols after east padding"
        # The original lon index is still n-1; padded cols are beyond it
        orig_lon_idx = new_idxs[0, 1]
        assert np.all(np.isnan(result.fp.values[0, :, orig_lon_idx + 1:])), \
            "Padded eastern cols should be NaN"

    def test_release_idxs_updated_after_padding(self):
        n, half = 8, 4
        fp = self._make_fp(n, n, 0, n // 2)   # release at south edge
        idxs = _get_release_idxs(fp)
        _, _, _, new_idxs = _pad_domain(fp, fp, idxs, half, "nans")
        assert new_idxs[0, 0] == half, \
            "After padding a south-edge release by 'half', lat index should equal half"

    def test_invalid_pad_mode_raises(self):
        n = 8
        fp = self._make_fp(n, n, 0, n // 2)
        idxs = _get_release_idxs(fp)
        with pytest.raises(ValueError):
            _pad_domain(fp, fp, idxs, 4, "invalid_mode")


# ── LoadBaseSatelliteData tests (Tier 2 — uses conftest fixtures) ─────────────

class TestLoadBase:
    """Tests for LoadBaseSatelliteData using real sample files (if --sample-files) or mocked methods."""

    @pytest.fixture
    def base_obj(self, fp_ds, met_ds, topog_ds, landcover_ds, 
                 test_fp_datadir, test_met_datadir, test_topog_path, test_landcover_path,
                 request):
        """
        Construct a LoadBaseSatelliteData instance.
        - If --sample-files and all paths are available: use real sample files directly
        - Otherwise: use mocked file-loading methods with synthetic data
        """
        # Check if all path fixtures are available (they skip if --sample-files not enabled)
        try:
            paths_available = all([
                test_fp_datadir is not None,
                test_met_datadir is not None,
                test_topog_path is not None,
                test_landcover_path is not None,
            ])
        except Exception:
            paths_available = False
        
        if paths_available:
            # Use real sample files
            return LoadBaseSatelliteData(
                year=TEST_YEAR,
                month=TEST_MONTH,
                fp_datadir=test_fp_datadir[:-9], # remove trailing date and .nc from path to get directory
                met_args={"met_datadir": test_met_datadir[:-9]}, # remove trailing date and .nc from path to get directory
                topog_args={"topog_path": test_topog_path,"landcover_path": test_landcover_path},
                load_everything=True
            )
        else:
            print("Using mocked data for LoadBaseSatelliteData tests (real sample files not available)")
            # Fall back to mocked approach with synthetic data
            def _mock_load_fps(self_, fp_datadir):
                self_.fp_data_full = fp_ds

            def _mock_get_met(self_, met_datadir, lazy_load=True):
                met_file = met_ds
                met_file = met_file.drop_duplicates(dim=["lat", "lon", "time"])
                if "model_level_number" in met_file.dims:
                    met_file = met_file.rename({"model_level_number": "levels"})
                self.met_file = met_file
                return self.met_file
                

            def _mock_load_topog(self_, topog_path="default", landcover_path="default"):
                topog = topog_ds
                if "latitude" in topog.dims:
                    topog = topog.rename({"latitude": "lat", "longitude": "lon"})
                return topog, landcover_ds

            with patch.object(LoadBaseSatelliteData, "_load_footprints", _mock_load_fps), \
                 patch.object(LoadBaseSatelliteData, "_get_meteorology_file", _mock_get_met), \
                 patch.object(LoadBaseSatelliteData, "load_topog", _mock_load_topog):
                return LoadBaseSatelliteData(year=TEST_YEAR, month=TEST_MONTH, load_everything=True)

    def test_required_attributes_exist(self, base_obj):
        assert hasattr(base_obj, "fp_data_full"), "Missing fp_data_full attribute"
        assert hasattr(base_obj, "met_file"),     "Missing met_file attribute"
        assert hasattr(base_obj, "topog_file"),   "Missing topog_file attribute"
        assert hasattr(base_obj, "landcover_file"), "Missing landcover_file attribute"

    def test_year_month_date_stored(self, base_obj):
        assert base_obj.year == TEST_YEAR, f"Expected year={TEST_YEAR}, got {base_obj.year}"
        assert base_obj.month == TEST_MONTH, f"Expected month='{TEST_MONTH}', got {base_obj.month}"
        assert base_obj.date == TEST_DATE, f"Expected date='{TEST_DATE}', got {base_obj.date}"

    def test_fp_time_in_correct_year(self, base_obj):
        years = pd.DatetimeIndex(base_obj.fp_data_full.time.values).year
        assert np.all(years == TEST_YEAR), f"All fp timestamps should be in {TEST_YEAR}"

    def test_met_time_in_correct_year(self, base_obj):
        years = pd.DatetimeIndex(base_obj.met_file.time.values).year
        assert np.all(years == TEST_YEAR), f"All met timestamps should be in {TEST_YEAR}"


# ── LoadSquareSatelliteData tests (Tier 2) ────────────────────────────────────

class TestLoadSquare:
    """Tests for LoadSquareSatelliteData: spatial shape, coordinate convention,
    time consistency across attributes, and NaN-padding for large crops."""

    def _make_square_obj(
        self,
        fp_ds,
        met_ds,
        topog_ds,
        landcover_ds,
        size,
        test_fp_datadir=None,
        test_met_datadir=None,
        test_topog_path=None,
        test_landcover_path=None,
    ):
        """Shared helper: build LoadSquareSatelliteData with real paths or mocked I/O."""

        paths_available = all([
            test_fp_datadir is not None,
            test_met_datadir is not None,
            test_topog_path is not None,
            test_landcover_path is not None,
        ])

        if paths_available:
            return LoadSquareSatelliteData(
                year=TEST_YEAR,
                month=TEST_MONTH,
                size=size,
                fp_datadir=test_fp_datadir[:-9],
                met_args={"met_datadir": test_met_datadir[:-9]},
                topog_args={"topog_path": test_topog_path, "landcover_path": test_landcover_path},
                load_everything=True,
                lazy_load=True,
            )
        
        else:
            print("Using mocked data for LoadSquareSatelliteData tests (real sample files not available)")

            def _mock_load_fps(self_, fp_datadir):
                self_.fp_data_full = fp_ds
                self_._subsample_frequency(**self_.subsample_parameters)

            def _mock_get_met(self_, met_datadir, lazy_load=True):
                met_file = met_ds
                met_file = met_file.drop_duplicates(dim=["lat", "lon", "time"])
                if "model_level_number" in met_file.dims:
                    met_file = met_file.rename({"model_level_number": "levels"})
                self.met_file = met_file
                return self.met_file

            def _mock_load_topog(self_, topog_path="default", landcover_path="default"):
                topog = topog_ds
                landcover = landcover_ds
                print("topog")
                print(topog)
                print("landcover")
                print(landcover)
                
                topog = self_._interp_topog(topog)
                landcover = self_._interp_landcover(landcover)
                return topog, landcover

            with patch.object(LoadBaseSatelliteData, "_load_footprints", _mock_load_fps), \
                patch.object(LoadBaseSatelliteData, "_get_meteorology_file", _mock_get_met), \
                patch.object(LoadBaseSatelliteData, "load_topog", _mock_load_topog):
                return LoadSquareSatelliteData(
                    year=TEST_YEAR, month=TEST_MONTH, size=size,
                    load_everything=True, lazy_load=True,
                )

    @pytest.fixture
    def square_obj_size8(
        self,
        fp_ds,
        met_ds,
        topog_ds,
        landcover_ds,
        test_fp_datadir,
        test_met_datadir,
        test_topog_path,
        test_landcover_path,
    ):
        return self._make_square_obj(
            fp_ds,
            met_ds,
            topog_ds,
            landcover_ds,
            size=8,
            test_fp_datadir=test_fp_datadir,
            test_met_datadir=test_met_datadir,
            test_topog_path=test_topog_path,
            test_landcover_path=test_landcover_path,
        )

    @pytest.fixture
    def square_obj_size20(
        self,
        fp_ds,
        met_ds,
        topog_ds,
        landcover_ds,
        test_fp_datadir,
        test_met_datadir,
        test_topog_path,
        test_landcover_path,
    ):
        return self._make_square_obj(
            fp_ds,
            met_ds,
            topog_ds,
            landcover_ds,
            size=20,
            test_fp_datadir=test_fp_datadir,
            test_met_datadir=test_met_datadir,
            test_topog_path=test_topog_path,
            test_landcover_path=test_landcover_path,
        )

    def test_fp_xr_spatial_shape(self, square_obj_size8):
        fp_xr = square_obj_size8.fp_xr
        print(fp_xr)
        assert fp_xr.fp.dims == ("time", "lat", "lon"), \
            f"Unexpected dims: {fp_xr.fp.dims}"
        assert fp_xr.fp.shape[1] == 8, "Expected lat size 8"
        assert fp_xr.fp.shape[2] == 8, "Expected lon size 8"

    def test_fp_xr_coords_are_artificial(self, square_obj_size8):
        fp_xr = square_obj_size8.fp_xr
        np.testing.assert_array_equal(fp_xr.lat.values, np.arange(8),
                                      err_msg="lat coords should be 0..7")
        np.testing.assert_array_equal(fp_xr.lon.values, np.arange(8),
                                      err_msg="lon coords should be 0..7")

    def test_fp_xr_has_actual_coord_vars(self, square_obj_size8):
        fp_xr = square_obj_size8.fp_xr
        assert "lat_coords" in fp_xr.data_vars, "Expected lat_coords variable"
        assert "lon_coords" in fp_xr.data_vars, "Expected lon_coords variable"

    def test_time_consistent_fp_and_topog(self, square_obj_size8):
        fp_times   = square_obj_size8.fp_xr.time.values
        topog_times = square_obj_size8.topog.time.values
        np.testing.assert_array_equal(fp_times, topog_times,
                                      err_msg="fp_xr and topog time coordinates must match")

    def test_time_consistent_fp_and_met(self, square_obj_size8):
        fp_times  = square_obj_size8.fp_xr.time.values
        met_fp_times = square_obj_size8.met.fp_time.values
        np.testing.assert_array_equal(fp_times, met_fp_times,
                                      err_msg="fp_xr.time and met.fp_time must match")
    """
    def test_large_size_produces_nans(self, square_obj_size200):
        fp_vals = square_obj_size200.fp_xr.fp.values
        assert np.isnan(fp_vals).any(), \
            "Expected NaN values in fp_xr.fp when crop size (200) exceeds domain"
    """

# ── Edge cases (Tier 1 / Tier 2) ─────────────────────────────────────────────

class TestEdgeCases:

    def test_cut_satellite_data_odd_size_raises(self, fp_ds):
        with pytest.raises(ValueError, match="even"):
            cut_satellite_data(fp_ds, size=7)

    def test_cut_topog_data_odd_size_raises(self, fp_ds, topog_ds, landcover_ds):
        topog = topog_ds
        if "latitude" in topog.dims:
            topog = topog.rename({"latitude": "lat", "longitude": "lon"})
        with pytest.raises(ValueError, match="even"):
            cut_topog_data(topog, landcover_ds, fp_ds, size=7)

    def test_get_domain_unknown_region_raises(self):
        # Use __new__ to bypass __init__ (which requires file paths)
        obj = LoadBaseSatelliteData.__new__(LoadBaseSatelliteData)
        with pytest.raises(ValueError):
            obj._get_domain("UNKNOWN_REGION")

    def test_subsample_freq1_no_change(self, fp_ds):
        def _mock_load_fps(self_, fp_datadir):
            self_.fp_data_full = fp_ds
            self_._subsample_frequency(**self_.subsample_parameters)

        with patch.object(LoadBaseSatelliteData, "_load_footprints", _mock_load_fps):
            obj = LoadBaseSatelliteData(year=TEST_YEAR, month=TEST_MONTH, freq=1)

        n_before = len(obj.fp_data_full.time)
        obj._subsample_frequency(freq=1)
        assert len(obj.fp_data_full.time) == n_before, \
            "freq=1 should leave the dataset unchanged"

    def test_subsample_freq2_halves_count(self, fp_ds):
        def _mock_load_fps(self_, fp_datadir):
            self_.fp_data_full = fp_ds
            self_._subsample_frequency(**self_.subsample_parameters)

        with patch.object(LoadBaseSatelliteData, "_load_footprints", _mock_load_fps):
            obj = LoadBaseSatelliteData(year=TEST_YEAR, month=TEST_MONTH, freq=1)

        n_before = len(obj.fp_data_full.time)
        obj._subsample_frequency(freq=2)
        expected = len(fp_ds.time.values[::2])  # same slice as _subsample_frequency uses
        assert len(obj.fp_data_full.time) == expected, \
            f"freq=2 should give {expected} timesteps, got {len(obj.fp_data_full.time)}"


    # ── _check_domain_overlap edge case tests ────────────────────────────────────

    class TestCheckDomainOverlap:
        """Tests for _check_domain_overlap edge conditions."""

        def _make_footprint_ds(self, lat_range, lon_range, release_lat, release_lon):
            lats = np.linspace(lat_range[0], lat_range[1], 5)
            lons = np.linspace(lon_range[0], lon_range[1], 5)
            return xr.Dataset(
                {
                    "fp": (["time", "lat", "lon"], np.ones((1, 5, 5), dtype="float32")),
                    "release_lat": ("time", [release_lat]),
                    "release_lon": ("time", [release_lon]),
                },
                coords={
                    "time": pd.date_range(TEST_START_DATE, periods=1),
                    "lat": lats,
                    "lon": lons,
                },
            )

        def _make_data_ds(self, lat_range, lon_range):
            lats = np.linspace(lat_range[0], lat_range[1], 5)
            lons = np.linspace(lon_range[0], lon_range[1], 5)
            return xr.Dataset(
                {"data": (["lat", "lon"], np.ones((5, 5), dtype="float32"))},
                coords={"lat": lats, "lon": lons},
            )

        def test_no_spatial_overlap_raises_error(self):
            fp = self._make_footprint_ds(
                lat_range=(-10.0, 10.0),
                lon_range=(-50.0, -40.0),
                release_lat=8.0,
                release_lon=-42.0,
            )
            data2 = self._make_data_ds(lat_range=(20.0, 30.0), lon_range=(-20.0, -10.0))

            def _mock_load_fps(self_, fp_datadir):
                self_.fp_data_full = fp

            with patch.object(LoadBaseSatelliteData, "_load_footprints", _mock_load_fps):
                obj = LoadBaseSatelliteData(year=TEST_YEAR, month=TEST_MONTH)

            with pytest.raises(ValueError, match="No spatial overlap"):
                obj._check_domain_overlap(fp, data2, "footprints", "meteorology")

        def test_domain_close_to_release_warns(self):
            fp = self._make_footprint_ds(
                lat_range=(-10.0, 10.0),
                lon_range=(-50.0, -40.0),
                release_lat=9.0,
                release_lon=-41.0,
            )
            data2 = self._make_data_ds(lat_range=(-5.0, 8.0), lon_range=(-48.0, -40.9))

            def _mock_load_fps(self_, fp_datadir):
                self_.fp_data_full = fp

            with patch.object(LoadBaseSatelliteData, "_load_footprints", _mock_load_fps):
                obj = LoadBaseSatelliteData(year=TEST_YEAR, month=TEST_MONTH)

            with pytest.warns(UserWarning, match="does not fully cover the area"):
                obj._check_domain_overlap(fp, data2, "footprints", "meteorology")
