"""Smoke test for the unify_dual_pred_zarr merge.

Checks, in order:
 1. the merged package imports (data loaders, training stacks, both model packages);
 2. LoadSquareSatelliteData reads meteorology from a yearly Zarr store;
 3. get_square_satellite_inputs_v2 builds an input array from the zarr met;
 4. load_GATES_data_v2 (the training-path loader) runs end to end for one month;
 5. the dual-model BC loader still pulls particle_locations_* variables;
 6. a dataloader yields a batch of the expected shape.

Run on a compute node (sbatch launch_smoke_zarr_merge.sh). Uses freq subsampling
so it stays light: only a handful of footprints and 3 met variables x 2 levels.
"""

import sys
import traceback

MET_DATADIR = "/group/chem/acrg/met_archive/zarr_store/SOUTHAMERICA/SOUTHAMERICA_Met_"
YEAR = 2018
MONTH = "01"
REGION = "BRAZIL"
SIZE = 10
FREQ = 200  # keep only ~1/200 footprints: enough to exercise the pipeline

MET_ARGS = {
    "met_datadir": MET_DATADIR,
    "met_variables": ["x_wind", "y_wind", "air_temperature"],
    "met_levels": [3, 9],
}

failures = []


def stage(name):
    def deco(fn):
        def wrapper(*args, **kwargs):
            print(f"\n===== STAGE: {name} =====", flush=True)
            try:
                out = fn(*args, **kwargs)
                print(f"----- PASS: {name}", flush=True)
                return out
            except Exception:
                traceback.print_exc()
                failures.append(name)
                print(f"----- FAIL: {name}", flush=True)
                return None
        return wrapper
    return deco


@stage("imports")
def test_imports():
    import gates  # noqa: F401
    from gates import LoadSquareSatelliteData  # noqa: F401
    from gates.data.datasets import get_square_satellite_inputs_v2  # noqa: F401
    import gates.training.training as gates_training  # noqa: F401
    import gates.training.training_dual  # noqa: F401
    import gates.training.training_background  # noqa: F401
    from gates.model.forecast import GraphSatelliteForecaster  # noqa: F401
    from model.forecast import (  # noqa: F401
        GraphSatelliteDualForecaster,
        GraphSatelliteBackgroundPredictor,
    )
    print("all merged modules import cleanly")
    return True


@stage("zarr met loading via LoadSquareSatelliteData")
def test_zarr_loading():
    from gates import LoadSquareSatelliteData

    data = LoadSquareSatelliteData(
        year=YEAR,
        month=MONTH,
        region=REGION,
        size=SIZE,
        freq=FREQ,
        met_args=MET_ARGS,
        verbose=True,
    )
    met = data.met_file
    assert met is not None, "met_file was not loaded"
    for v in MET_ARGS["met_variables"]:
        assert v in met.data_vars, f"met variable {v} missing from loaded zarr met"
    assert "levels" in met.dims, f"expected 'levels' dim in zarr met, got {met.dims}"
    n_fp = len(data.fp_data_full.time)
    print(f"loaded {n_fp} footprints; met dims: {dict(met.sizes)}")
    src = met.encoding.get("source", "") or str(getattr(met, "_file_obj", ""))
    print(f"met source hint: {src!r}")
    assert n_fp > 0, "no footprints loaded"
    return data


@stage("inputs via get_square_satellite_inputs_v2")
def test_inputs(data):
    from gates.data.datasets import get_square_satellite_inputs_v2

    inputs, data = get_square_satellite_inputs_v2(
        data,
        met_variables=MET_ARGS["met_variables"],
        met_levels=MET_ARGS["met_levels"],
        static_variables=["lat_coords", "lon_coords"],
        time_deltas=[6],
        add_wind_direction=False,
        verbose=True,
        load_into_memory=True,
    )
    assert "fp_time" in inputs.dims and "variable_name" in inputs.dims
    assert inputs.sizes["lat"] == SIZE and inputs.sizes["lon"] == SIZE
    n_nan = int(inputs.isnull().sum())
    print(f"inputs shape: {dict(inputs.sizes)}; NaNs: {n_nan}")
    return inputs, data


@stage("training-path loader load_GATES_data_v2")
def test_load_gates_data_v2():
    import gates.training.training as gates_training

    fp_xr, inputs = gates_training.load_GATES_data_v2(
        {
            "year": YEAR,
            "month": MONTH,
            "region": REGION,
            "size": SIZE,
            "freq": FREQ,
            "met_args": dict(MET_ARGS),
        },
        input_variables={
            "met_variables": MET_ARGS["met_variables"],
            "met_levels": MET_ARGS["met_levels"],
            "static_variables": ["lat_coords", "lon_coords"],
            "time_deltas": [6],
            "add_wind_direction": False,
        },
        verbose=True,
        load_into_memory=True,
    )
    assert len(fp_xr.time) > 0, "no footprints returned"
    assert len(fp_xr.time) == inputs.sizes["fp_time"], (
        f"fp/inputs mismatch: {len(fp_xr.time)} vs {inputs.sizes['fp_time']}"
    )
    print(f"load_GATES_data_v2 returned {len(fp_xr.time)} samples, "
          f"inputs {dict(inputs.sizes)}")
    return fp_xr, inputs


@stage("dual-model BC loading (LoadSquareSatelliteDataWithBCs)")
def test_bc_loading():
    from gates.training.training_background import LoadSquareSatelliteDataWithBCs

    data = LoadSquareSatelliteDataWithBCs(
        year=YEAR,
        month=MONTH,
        region=REGION,
        size=SIZE,
        freq=FREQ,
        met_args=MET_ARGS,
        load_everything=False,
        verbose=True,
    )
    for v in ["particle_locations_n", "particle_locations_s",
              "particle_locations_e", "particle_locations_w"]:
        assert v in data.fp_data_full.data_vars, f"BC variable {v} not loaded"
    print("BC particle_locations_* variables present in fp_data_full")
    return True


@stage("dataloader batch")
def test_dataloader(inputs, data):
    import gates.data.datasets as gates_datasets

    inputs_t, fps_t = gates_datasets.trim_to_batch_size(inputs, data.fp_xr.fp, batch_size=2)
    loader, labels = gates_datasets.make_dataloader(
        inputs_t, fps_t, batch_size=2, dataloader_params={"num_workers": 0}
    )
    x, y = next(iter(loader))
    print(f"first batch: x {tuple(x.shape)}, y {tuple(y.shape)}, labels {labels}")
    assert x.shape[0] == 2
    return True


if __name__ == "__main__":
    ok = test_imports()
    data = test_zarr_loading() if ok else None
    pair = test_inputs(data) if data is not None else None
    test_load_gates_data_v2()
    test_bc_loading()
    if pair is not None:
        test_dataloader(*pair)

    print("\n===== SMOKE TEST SUMMARY =====")
    if failures:
        print(f"FAILED stages: {failures}")
        sys.exit(1)
    print("ALL STAGES PASSED")
