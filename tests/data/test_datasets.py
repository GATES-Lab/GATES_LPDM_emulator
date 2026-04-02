import pytest
import numpy as np

from model.data.datasets import get_square_satellite_inputs
from ..test_helper_funs import make_square_satellite_obj

TEST_YEAR = 2016
TEST_MONTH = "01"


# ── TestGetSquareSatelliteInputs ──────────────────────────────────────────────

class TestGetSquareSatelliteInputs:
    """Tests for get_square_satellite_inputs using a LoadSquareSatelliteData object
    built either from real sample files (--sample-files) or synthetic mocks."""

    MET_PARAMS = {
        "met_variables": {
            "x_wind": [3],
            "y_wind": [3],
            "atmosphere_boundary_layer_thickness": [],
        },
        "static_variables": ["lat_coords", "lon_coords", "topog"],
        "time_deltas": [6],
    }

    # ── shared fixture ────────────────────────────────────────────────────────

    @pytest.fixture
    def inputs_result(
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
        """Build the square data object and call get_square_satellite_inputs once."""
        obj = make_square_satellite_obj(
            fp_ds,
            met_ds,
            topog_ds,
            landcover_ds,
            year=TEST_YEAR,
            month=TEST_MONTH,
            size=8,
            test_fp_datadir=test_fp_datadir,
            test_met_datadir=test_met_datadir,
            test_topog_path=test_topog_path,
            test_landcover_path=test_landcover_path,
        )
        result = get_square_satellite_inputs(
            obj,
            met_variables=self.MET_PARAMS["met_variables"],
            time_deltas=self.MET_PARAMS["time_deltas"],
            static_variables=self.MET_PARAMS["static_variables"],
            verbose=False,
        )
        return result, fp_ds

    # ── tests ─────────────────────────────────────────────────────────────────

    def test_get_square_satellite_inputs_runs(self, inputs_result):
        """get_square_satellite_inputs completes without error and returns a non-empty DataArray."""
        result, _ = inputs_result
        assert result is not None
        assert result.sizes["fp_time"] > 0

    def test_fp_times_are_subset_of_fp_ds(self, inputs_result):
        """Every fp_time in the result should be traceable back to a time in fp_ds."""
        result, fp_ds = inputs_result
        assert np.all(np.isin(result.fp_time.values, fp_ds.time.values)), (
            "Some fp_time values in the result are not present in fp_ds.time"
        )

    def test_variable_names_match_params(self, inputs_result):
        """The variable-name level of the result MultiIndex should contain exactly
        the variables requested via met_variables and static_variables."""
        result, _ = inputs_result
        expected = (
            set(self.MET_PARAMS["met_variables"].keys())
            | set(self.MET_PARAMS["static_variables"])
        )
        actual = set(result.coords["variable"].values)
        assert actual == expected
