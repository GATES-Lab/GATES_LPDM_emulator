import pytest
import numpy as np
import pandas as pd
import xarray as xr

from gates.data.datasets import (
    get_square_satellite_inputs,
    make_dataloader,
    InputsDataset,
    XarrayScaler,
    LogAndShiftFpScaler,
    LogAndShiftMeanFpScaler,
    FootprintDataset,
)
from ..test_helper_funs import make_square_satellite_obj
from ..conftest import SAMPLE_YEAR, SAMPLE_MONTH

TEST_YEAR = int(SAMPLE_YEAR)
TEST_MONTH = SAMPLE_MONTH


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
    ):
        """Build the square data object and call get_square_satellite_inputs once."""
        result, data_used = get_square_satellite_inputs(
            square_obj,
            met_variables=self.MET_PARAMS["met_variables"],
            time_deltas=self.MET_PARAMS["time_deltas"],
            static_variables=self.MET_PARAMS["static_variables"],
            verbose=False,
        )
        return result, data_used

    # ── tests ─────────────────────────────────────────────────────────────────

    def test_get_square_satellite_inputs_runs(self, inputs_result):
        """get_square_satellite_inputs completes without error and returns a non-empty DataArray."""
        result, _ = inputs_result
        assert result is not None
        assert result.sizes["fp_time"] > 0

    def test_fp_times_are_subset_of_fp_ds(self, inputs_result):
        """Every fp_time in the result should be traceable back to a time in fp_ds."""
        result, data_used = inputs_result
        assert np.all(np.isin(result.fp_time.values, data_used.fp_data_full.time.values)), (
            "Some fp_time values in the result are not present in fp_ds.time"
        )
        assert np.all(result.fp_time.values == data_used.fp_xr.time.values), (
            "fp_time values in the result do not match fp_xr.time values from the data object, although they should be identical after selection and alignment"
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
        result, _ = get_square_satellite_inputs(
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
        result, _ = get_square_satellite_inputs(
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
        result, _ = get_square_satellite_inputs(
            square_obj,
            met_variables={"x_wind": [3]},
            time_deltas=deltas,
            static_variables=[],
            verbose=False,
        )
        assert result.attrs["time_deltas"] == [0, 6, 12]
        assert 0 in result.attrs["time_deltas"]

    def test_empty_time_deltas_defaults_to_t0(self, square_obj):
        """An empty time_deltas input should still return only t=0 features."""
        result, _ = get_square_satellite_inputs(
            square_obj,
            met_variables={"x_wind": [3]},
            time_deltas=[],
            static_variables=[],
            verbose=False,
        )
        assert result.attrs["time_deltas"] == [0]
        assert set(pd.unique(result.coords["time_delta"].values)) == {0}

    def test_surface_only_met_variables(self, square_obj):
        """Surface-only met variables should be handled without level extraction."""
        result, _ = get_square_satellite_inputs(
            square_obj,
            met_variables={
                "atmosphere_boundary_layer_thickness": [],
            },
            time_deltas=[],
            static_variables=[],
            verbose=False,
        )

        variable_tuples = list(result.variable_name.values)
        assert {"atmosphere_boundary_layer_thickness"} == set(result.coords["variable"].values)
        assert all(level == 0 for _, level, _ in variable_tuples)
        assert all(time_delta == 0 for _, _, time_delta in variable_tuples)

    def test_levelled_variable_passed_as_surface_warns_and_skips(self, square_obj):
        """Levelled variables passed as surface should warn and be skipped."""
        with pytest.warns(UserWarning, match="passed as surface variables"):
            result, _ = get_square_satellite_inputs(
                square_obj,
                met_variables={"x_wind": [], "atmosphere_boundary_layer_thickness": []},
                time_deltas=[],
                static_variables=[],
                verbose=False,
            )
        assert "x_wind" not in set(result.coords["variable"].values)
        assert "atmosphere_boundary_layer_thickness" in set(result.coords["variable"].values)

    def test_all_surface_variables_skipped_still_raises(self, square_obj):
        """If every requested input is skipped, final concatenation should still raise."""
        with pytest.warns(UserWarning, match="passed as surface variables"):
            with pytest.raises(ValueError, match="unavailable"):
                get_square_satellite_inputs(
                    square_obj,
                    met_variables={"x_wind": []},
                    time_deltas=[],
                    static_variables=[],
                    verbose=False,
                )

    def test_level_only_met_variables_without_static_variables(self, square_obj):
        """Level-based met variables should work on their own without static variables."""
        result, _ = get_square_satellite_inputs(
            square_obj,
            met_variables={
                "x_wind": [3],
                "y_wind": [3],
            },
            time_deltas=[6],
            static_variables=[],
            verbose=False,
        )

        variable_tuples = list(result.variable_name.values)
        assert {"x_wind", "y_wind"} == set(result.coords["variable"].values)
        assert ("x_wind", 3, 6) in variable_tuples or ("x_wind", 3, 0) in variable_tuples
        assert ("y_wind", 3, 6) in variable_tuples or ("y_wind", 3, 0) in variable_tuples
        assert result.attrs["time_deltas"] == [0, 6]

    def test_static_variables_only(self, square_obj):
        """Static variables should be extractable without any met variables at all."""
        result, _ = get_square_satellite_inputs(
            square_obj,
            met_variables={},
            time_deltas=[],
            static_variables=["lat_coords", "lon_coords", "topog"],
            verbose=False,
        )

        variable_tuples = list(result.variable_name.values)
        assert set(result.coords["variable"].values) == {"lat_coords", "lon_coords", "topog"}
        assert all(level == 0 for _, level, _ in variable_tuples)
        assert all(time_delta == 0 for _, _, time_delta in variable_tuples)


class TestMakeDataloader:
    """Tests for make_dataloader."""

    MET_PARAMS = {
        "met_variables": {
            "x_wind": [3],
            "y_wind": [3],
            "atmosphere_boundary_layer_thickness": [],
        },
        "static_variables": ["lat_coords", "lon_coords", "topog"],
        "time_deltas": [6],
    }

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
        """Build the square data object for dataloader tests."""
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
    def get_inputs(self, square_obj):
        """Return inputs and an fp DataArray from the returned, potentially-updated data object."""
        inputs, data_used = get_square_satellite_inputs(
            square_obj,
            met_variables=self.MET_PARAMS["met_variables"],
            time_deltas=self.MET_PARAMS["time_deltas"],
            static_variables=self.MET_PARAMS["static_variables"],
            verbose=False,
        )
        fps = data_used.fp_xr.fp
        return inputs, fps

    def test_raises_on_incompatible_time_dimensions(self, get_inputs):
        """Mismatched time lengths between inputs and fps should raise."""
        inputs, fps = get_inputs
        fps_short = fps.isel(time=slice(0, -1))

        with pytest.raises(ValueError, match="Incompatible dimensions"):
            make_dataloader(
                inputs,
                fps_short,
                batch_size=4,
                randomize=False,
                dataloader_params={"num_workers": 0},
            )

    def test_dataarray_fps_returns_expected_shapes(self, get_inputs):
        """DataArray fps path should yield (X, y) with expected dimensionality if passing a dataarray."""
        inputs, fps = get_inputs
        dataloader, fps_labels = make_dataloader(
            inputs,
            fps,
            batch_size=4,
            randomize=False,
            dataloader_params={"num_workers": 0},
        )

        x_batch, y_batch = next(iter(dataloader))
        assert isinstance(fps_labels, list)
        assert x_batch.ndim == 4
        assert y_batch.ndim == 3
        assert x_batch.shape[0] == y_batch.shape[0]
        print(fps_labels)
        assert len(fps_labels) == 1
        assert x_batch.shape[1] == y_batch.shape[1]
        assert x_batch.shape[2] == y_batch.shape[2]

    def test_randomize_is_reproducible_with_fixed_seed(self, get_inputs):
        """Using the same random seed should produce the same shuffled first batch."""
        inputs, fps = get_inputs

        dataloader_a, _ = make_dataloader(
            inputs,
            fps,
            batch_size=4,
            randomize=True,
            random_seed=7,
            dataloader_params={"num_workers": 0},
        )
        dataloader_b, _ = make_dataloader(
            inputs,
            fps,
            batch_size=4,
            randomize=True,
            random_seed=7,
            dataloader_params={"num_workers": 0},
        )

        x_a, y_a = next(iter(dataloader_a))
        x_b, y_b = next(iter(dataloader_b))
        assert np.array_equal(np.asarray(x_a), np.asarray(x_b))
        assert np.array_equal(np.asarray(y_a), np.asarray(y_b))

    def test_custom_dataloader_params_are_accepted(self, get_inputs):
        """Passing explicit dataloader_params should build a working dataloader."""
        inputs, fps = get_inputs
        dataloader, _ = make_dataloader(
            inputs,
            fps,
            batch_size=4,
            randomize=False,
            dataloader_params={"num_workers": 0, "pin_memory": False},
        )

        x_batch, y_batch = next(iter(dataloader))
        assert x_batch.shape[0] > 0
        assert y_batch.shape[0] > 0

    def test_transformed_dataset_fps_has_two_labels_and_extra_y_dimension(self, get_inputs):
        """Passing transformed footprint Dataset should yield two fps labels and 4D y batches."""
        inputs, fps = get_inputs
        fp_dataset = FootprintDataset(fps)
        transformed_fps = fp_dataset.fit_transform()

        dataloader, fps_labels = make_dataloader(
            inputs,
            transformed_fps,
            batch_size=4,
            randomize=False,
            dataloader_params={"num_workers": 0},
        )

        _, y_batch = next(iter(dataloader))
        assert len(fps_labels) == 2
        assert y_batch.ndim == 4
        assert y_batch.shape[-1] == 2


class TestFootprintTransformers:
    """Tests for footprint scalers and FootprintDataset wrapper."""

    @pytest.fixture
    def fp_da(self):
        """Small synthetic footprint DataArray with non-negative values."""
        times = pd.date_range("2016-01-01", periods=4, freq="h")
        lat = np.array([0.0, 1.0])
        lon = np.array([10.0, 11.0, 12.0])
        values = np.array(
            [
                [[0.0, 1e-6, 1e-3], [1.0, 2.0, 5.0]],
                [[0.0, 1e-5, 1e-2], [2.0, 3.0, 6.0]],
                [[0.0, 1e-4, 1e-1], [3.0, 4.0, 7.0]],
                [[0.0, 2e-4, 2e-1], [4.0, 5.0, 8.0]],
            ],
            dtype="float32",
        )
        return xr.DataArray(
            values,
            dims=("time", "lat", "lon"),
            coords={"time": times, "lat": lat, "lon": lon},
            name="fp",
        )

    def test_log_and_shift_fp_scaler_non_negative_transform(self, fp_da):
        """LogAndShiftFpScaler should produce non-negative transformed outputs when configured."""
        scaler = LogAndShiftFpScaler(minimum_oom=5, non_negative=True)
        transformed = scaler.transform(fp_da)
        assert transformed.min().item() >= 0

    def test_log_and_shift_fp_scaler_inverse_roundtrip_for_large_values(self, fp_da):
        """Inverse transform should approximately recover values above threshold."""
        scaler = LogAndShiftFpScaler(minimum_oom=5, non_negative=True)
        transformed = scaler.transform(fp_da)
        recovered = scaler.inverse_transform(transformed)

        mask = fp_da.values > 1e-5
        assert np.allclose(recovered.values[mask], fp_da.values[mask], rtol=1e-5, atol=1e-8)

    def test_log_and_shift_mean_scaler_requires_fit(self, fp_da):
        """LogAndShiftMeanFpScaler.transform should fail if fit has not been called."""
        scaler = LogAndShiftMeanFpScaler(minimum_oom=5)
        with pytest.raises(AttributeError):
            scaler.transform(fp_da)

    def test_log_and_shift_mean_scaler_fit_transform_inverse(self, fp_da):
        """Mean-shift scaler should support fit, transform and inverse_transform coherently."""
        scaler = LogAndShiftMeanFpScaler(minimum_oom=5)
        scaler.fit(fp_da)
        transformed = scaler.transform(fp_da)
        recovered = scaler.inverse_transform(transformed)
        assert np.allclose(recovered.values, fp_da.values, rtol=1e-5, atol=1e-7)

    def test_footprint_dataset_check_fp_format_rejects_invalid(self, fp_da):
        """FootprintDataset should reject inputs that are not DataArray or Dataset with fp."""
        with pytest.raises(ValueError, match="xarray DataSet"):
            FootprintDataset(np.array([1.0, 2.0, 3.0]))

        bad_ds = xr.Dataset({"not_fp": fp_da})
        with pytest.raises(ValueError, match="variable named 'fp'"):
            FootprintDataset(bad_ds)

    def test_footprint_dataset_transform_output_schema(self, fp_da):
        """FootprintDataset.transform should return expected variables and idx coordinate."""
        fp_dataset = FootprintDataset(fp_da)
        fp_dataset.fit()
        transformed_ds = fp_dataset.transform(fp_da)

        assert "fp_transformed" in transformed_ds.data_vars
        assert "fp_original" in transformed_ds.data_vars
        assert "idx" in transformed_ds.coords
        assert transformed_ds.sizes["time"] == transformed_ds.idx.size

    def test_footprint_dataset_inverse_transform_accepts_dataset_and_dataarray(self, fp_da):
        """inverse_transform should accept either full transformed dataset or DataArray."""
        fp_dataset = FootprintDataset(fp_da)
        transformed_ds = fp_dataset.fit_transform()

        recovered_from_ds = fp_dataset.inverse_transform(transformed_ds)
        recovered_from_da = fp_dataset.inverse_transform(transformed_ds["fp_transformed"])

        assert np.allclose(recovered_from_ds.values, recovered_from_da.values, rtol=1e-6, atol=1e-8)


class TestPipeline:
    """End-to-end tests across inputs, footprint transforms, and dataloader creation."""

    MET_PARAMS = {
        "met_variables": {
            "x_wind": [3],
            "y_wind": [3],
            "atmosphere_boundary_layer_thickness": [],
        },
        "static_variables": ["lat_coords", "lon_coords", "topog"],
        "time_deltas": [6],
    }

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
        """Build the square data object for the end-to-end pipeline test."""
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

    def test_full_pipeline_runs(self, square_obj, use_sample_files):
        """End-to-end smoke test: build object, extract inputs, transform both inputs and fps, build dataloader."""
        if not use_sample_files:
            pytest.skip("pipeline smoke test requires --sample-files")

        inputs, data_used = get_square_satellite_inputs(
            square_obj,
            met_variables=self.MET_PARAMS["met_variables"],
            time_deltas=self.MET_PARAMS["time_deltas"],
            static_variables=self.MET_PARAMS["static_variables"],
            verbose=False,
        )

        inputs_dataset = InputsDataset(inputs, scaler_params={"fit_on_subsample": 1})
        inputs_dataset.fit()
        transformed_inputs = inputs_dataset.transform(inputs)

        fp_dataset = FootprintDataset(data_used.fp_xr.fp)
        transformed_fps = fp_dataset.fit_transform()

        dataloader, fps_labels = make_dataloader(
            transformed_inputs,
            transformed_fps,
            batch_size=4,
            randomize=False,
            dataloader_params={"num_workers": 0},
        )

        x_batch, y_batch = next(iter(dataloader))
        assert inputs.sizes["fp_time"] > 0
        assert transformed_inputs.attrs["transformer"] == "DefaultInputsScaler"
        assert len(fps_labels) == 2
        assert x_batch.ndim == 4
        assert y_batch.ndim == 4
        assert y_batch.shape[-1] == 2


class TestInputsDataset:
    """Tests for InputsDataset and its scaler delegation behavior."""

    @pytest.fixture
    def inputs_da(self):
        """Synthetic inputs DataArray with a variable_name MultiIndex."""
        times = pd.date_range("2016-01-01", periods=6, freq="h")
        lat = np.array([-1.0, 0.0])
        lon = np.array([10.0, 11.0])
        variable_tuples = [
            ("x_wind", 3, 0),
            ("topog", 0, 0),
        ]
        values = np.array(
            [
                [[[1.0, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]]],
                [[[2.0, 3.0], [4.0, 5.0]], [[11.0, 21.0], [31.0, 41.0]]],
                [[[3.0, 4.0], [5.0, 6.0]], [[12.0, 22.0], [32.0, 42.0]]],
                [[[4.0, 5.0], [6.0, 7.0]], [[13.0, 23.0], [33.0, 43.0]]],
                [[[5.0, 6.0], [7.0, 8.0]], [[14.0, 24.0], [34.0, 44.0]]],
                [[[6.0, 7.0], [8.0, 9.0]], [[15.0, 25.0], [35.0, 45.0]]],
            ],
            dtype="float32",
        )
        data_array = xr.DataArray(
            values,
            dims=("fp_time", "lat", "lon", "variable_name"),
            coords={
                "fp_time": times,
                "lat": lat,
                "lon": lon,
                "variable_name": pd.MultiIndex.from_tuples(variable_tuples, names=["variable", "levels", "time_delta"]),
            },
            name="inputs",
        )
        return data_array

    def test_constructor_defaults_to_default_scaler(self, inputs_da):
        """InputsDataset without a scaler should wrap DefaultInputsScaler."""
        dataset = InputsDataset(inputs_da, scaler_params={"fit_on_subsample": 1})
        assert dataset.scaler.__class__.__name__ == "DefaultInputsScaler"
        assert dataset.fit_on_subsample == 1

    def test_constructor_accepts_custom_scaler(self, inputs_da):
        """InputsDataset should instantiate a provided scaler class."""
        dataset = InputsDataset(inputs_da, scaler=XarrayScaler, scaler_params={"fit_on_subsample": 1})
        assert isinstance(dataset.scaler, XarrayScaler)

    def test_fit_on_subsample_negative_raises(self, inputs_da):
        """Negative fit_on_subsample should raise a ValueError."""
        dataset = InputsDataset(inputs_da, scaler_params={"fit_on_subsample": -0.1})
        with pytest.raises(ValueError, match="fit_on_subsample should be between 0 and 1"):
            dataset.fit()

    def test_fit_on_subsample_creates_subsampled_inputs(self, inputs_da):
        """Subsample fit should store a reduced set of fp_time samples."""
        np.random.seed(0)
        dataset = InputsDataset(inputs_da, scaler_params={"fit_on_subsample": 0.5})
        dataset.fit()

        assert hasattr(dataset, "subsampled_inputs")
        assert dataset.subsampled_inputs.sizes["fp_time"] == 3
        assert dataset.subsampled_inputs.sizes["fp_time"] < inputs_da.sizes["fp_time"]

    def test_fit_on_full_dataset_uses_all_inputs(self, inputs_da):
        """fit_on_subsample >= 1 should fit on the full input set."""
        dataset = InputsDataset(inputs_da, scaler_params={"fit_on_subsample": 1})
        dataset.fit()

        assert not hasattr(dataset, "subsampled_inputs")
        assert hasattr(dataset.scaler, "fitted_variable_names")

    def test_transform_preserves_structure_with_default_scaler(self, inputs_da):
        """Default scaler transformation should preserve dimensions and variable_name labels."""
        dataset = InputsDataset(inputs_da, scaler_params={"fit_on_subsample": 1})
        dataset.fit()
        transformed = dataset.transform(inputs_da)

        assert transformed.dims == inputs_da.dims
        assert set(map(tuple, transformed.variable_name.values)) == set(map(tuple, inputs_da.variable_name.values))
        assert transformed.attrs["transformer"] == "DefaultInputsScaler"
        assert transformed.name == "stacked_transformed_inputs"

    def test_transform_delegates_to_custom_scaler(self, inputs_da):
        """InputsDataset.transform should delegate to the wrapped scaler instance."""

        class DummyScaler:
            def __init__(self):
                self.fit_called = False
                self.transform_called = False

            def fit(self, inputs):
                self.fit_called = True
                self.fitted_shape = inputs.shape
                return self

            def transform(self, inputs):
                self.transform_called = True
                return inputs + 1

        dataset = InputsDataset(inputs_da, scaler=DummyScaler, scaler_params={"fit_on_subsample": 1})
        dataset.fit()
        transformed = dataset.transform(inputs_da)

        assert dataset.scaler.fit_called is True
        assert dataset.scaler.transform_called is True
        assert np.allclose(transformed.values, inputs_da.values + 1)

    """
    def test_footprint_dataset_save_and_load_scaler_roundtrip(self, fp_da, tmp_path):
        #Saved scaler should load into a new FootprintDataset instance and preserve type.
        fp_dataset = FootprintDataset(fp_da)
        fp_dataset.fit()
        fp_dataset.save_scaler(tmp_path)

        loaded_dataset = FootprintDataset(fp_da)
        loaded_dataset.load_scaler(tmp_path)

        assert loaded_dataset.scaler.__class__.__name__ == fp_dataset.scaler.__class__.__name__
    """