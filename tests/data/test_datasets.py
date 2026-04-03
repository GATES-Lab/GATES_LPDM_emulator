import pytest
import numpy as np
import pandas as pd

from model.data.datasets import get_square_satellite_inputs_v2 as get_square_satellite_inputs
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
    def square_obj(
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
        """Build the square data object once for edge-case tests."""
        return make_square_satellite_obj(
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

    @pytest.fixture
    def inputs_result(
        self,
        square_obj,
        fp_ds,
    ):
        """Build the square data object and call get_square_satellite_inputs once."""
        result = get_square_satellite_inputs(
            square_obj,
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

    def test_raises_when_input_is_not_satellite_object(self):
        """Passing a plain object should fail the SatelliteData guard."""
        with pytest.raises(AssertionError, match="SatelliteData"):
            get_square_satellite_inputs(
                object(),
                met_variables={"x_wind": [3]},
                time_deltas=[],
                static_variables=[],
                verbose=False,
            )

    def test_raises_when_met_not_processed(self, square_obj):
        """Function should fail fast if meteorology is not processed."""
        square_obj.met_processed = False
        with pytest.raises(AssertionError, match="loaded and cut the meteorology"):
            get_square_satellite_inputs(
                square_obj,
                met_variables={"x_wind": [3]},
                time_deltas=[],
                static_variables=[],
                verbose=False,
            )

    def test_raises_when_met_variables_not_dict(self, square_obj):
        """met_variables must be a dict of variable -> levels list."""
        with pytest.raises(ValueError, match="met_variables should be a dict"):
            get_square_satellite_inputs(
                square_obj,
                met_variables=["x_wind"],
                time_deltas=[],
                static_variables=[],
                verbose=False,
            )

    def test_missing_level_is_dropped_when_other_levels_exist(self, square_obj):
        """Unknown levels should be dropped while valid requested levels are kept."""
        result = get_square_satellite_inputs(
            square_obj,
            met_variables={
                "x_wind": [3, 999],
                "atmosphere_boundary_layer_thickness": [],
            },
            time_deltas=[],
            static_variables=["lat_coords"],
            verbose=False,
        )

        tuples = list(result.variable_name.values)
        assert ("x_wind", 3, 0) in tuples
        assert ("x_wind", 999, 0) not in tuples

    def test_all_requested_levels_missing_raises(self, square_obj):
        """If all levelled requests are invalid and no other inputs remain, concat should fail."""
        with pytest.raises(ValueError, match="levels"):
            get_square_satellite_inputs(
                square_obj,
                met_variables={"x_wind": [999]},
                time_deltas=[],
                static_variables=[],
                verbose=False,
            )

    def test_missing_met_variable_raises_keyerror(self, square_obj):
        """Requesting a variable that is absent in met should raise a warning during selection/stacking."""
        with pytest.raises(ValueError, match="unavailable"):
            get_square_satellite_inputs(
                square_obj,
                met_variables={"not_a_met_var": [3]},
                time_deltas=[],
                static_variables=[],
                verbose=False,
            )
        with pytest.warns(UserWarning, match="variable"):
            get_square_satellite_inputs(
                square_obj,
                met_variables={"x_wind": [3], "not_a_met_var": [3]},
                time_deltas=[],
                static_variables=[],
                verbose=False,
            )

    def test_unknown_static_variable_is_skipped(self, square_obj):
        """Unknown static variable names should be ignored without breaking output creation."""
        result = get_square_satellite_inputs(
            square_obj,
            met_variables={"x_wind": [3]},
            time_deltas=[],
            static_variables=["lat_coords", "unknown_static_var"],
            verbose=False,
        )
        assert "unknown_static_var" not in set(result.coords["variable"].values)
        assert "lat_coords" in set(result.coords["variable"].values)

    def test_time_deltas_sorted_unique_and_zero_included(self, square_obj):
        """Returned attrs should normalize time_deltas to sorted unique values including t=0."""
        deltas = [12, 6, 6]
        result = get_square_satellite_inputs(
            square_obj,
            met_variables={"x_wind": [3]},
            time_deltas=deltas,
            static_variables=[],
            verbose=False,
        )
        assert result.attrs["time_deltas"] == [0, 6, 12]
        # Current behavior mutates caller list when 0 was missing.
        assert 0 in result.attrs["time_deltas"]

    def test_empty_time_deltas_defaults_to_t0(self, square_obj):
        """An empty time_deltas input should still return only t=0 features."""
        result = get_square_satellite_inputs(
            square_obj,
            met_variables={"x_wind": [3]},
            time_deltas=[],
            static_variables=[],
            verbose=False,
        )
        assert result.attrs["time_deltas"] == [0]
        assert set(pd.unique(result.coords["time_delta"].values)) == {0}
