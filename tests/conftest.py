# tests/conftest.py
import pytest, os, sys
import numpy as np
import pandas as pd
import xarray as xr


FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "sample_data")
SAMPLE_YEAR = "2016"
SAMPLE_MONTH = "01"

# ---------- synthetic helpers (hardcoded structure) ----------
def _make_fp_ds(n_time=100, n_lat=50, n_lon=50, sample_year=SAMPLE_YEAR, sample_month=SAMPLE_MONTH):
    """Synthetic footprint dataset matching real data structure."""
    #times = pd.date_range("2016-01-01", periods=n_time, freq="6h")
    # make timestamps n_time randomly sampled to the nearest second from the whole month
    month_start = pd.Timestamp(f"{sample_year}-{sample_month}-01")
    month_end = month_start + pd.offsets.MonthEnd(0)  # get last day of the month
    total_seconds = int((month_end - month_start).total_seconds()) + 1
    second_offsets = np.random.choice(total_seconds, n_time, replace=False)
    times = month_start + pd.to_timedelta(second_offsets, unit="s")

    lats  = np.linspace(-15.0,  15.0, n_lat)
    lons  = np.linspace(-40.0, 30.0, n_lon)
    ds = xr.Dataset(
        {"fp": (["time", "lat", "lon"], np.random.rand(n_time, n_lat, n_lon)),
        "release_lat": (["time"], np.random.uniform(-10.0, 10.0, n_time)), 
        "release_lon": (["time"], np.random.uniform(-30.0, 20.0, n_time))},
        coords={"time": times, "lat": lats, "lon": lons}
    )
    ds = ds.sortby("time")  # ensure time is sorted for _get_release_idxs
    return ds

def _make_met_ds(n_lat=50, n_lon=50, sample_year=SAMPLE_YEAR, sample_month=SAMPLE_MONTH):
    """Synthetic met dataset matching real data structure."""
    # get three-hourly timestamps for the whole month
    month_start = pd.Timestamp(f"{sample_year}-{sample_month}-01")
    month_end = month_start + pd.offsets.MonthEnd(0)
    times = pd.date_range(month_start, month_end, freq="3h")
    n_time = len(times)
    lats   = np.linspace(-15.0,  15.0, n_lat)
    lons   = np.linspace(-40.0, 30.0, n_lon)
    levels = [3, 5, 10]
    n_levels = len(levels)
    return xr.Dataset(
        {"x_wind": (["time","model_level_number","lat","lon"], np.random.rand(n_time, n_levels, n_lat, n_lon)),
         "y_wind": (["time","model_level_number","lat","lon"], np.random.rand(n_time, n_levels, n_lat, n_lon)),
         "atmosphere_boundary_layer_thickness": (["time","lat","lon"], np.random.rand(n_time, n_lat, n_lon))},
        coords={"time": times, "model_level_number": levels, "lat": lats, "lon": lons}
    )

def _make_topog_ds(n_lat=100, n_lon=100):
    """Synthetic topography dataset matching real data structure.
    Uses 'latitude'/'longitude' coords (as in the raw file, before rename)."""
    lats = np.linspace(-20.0, 20.0, n_lat)
    lons = np.linspace(-50.0, 40.0, n_lon)
    return xr.Dataset(
        {"surface_altitude": (["latitude", "longitude"],
                              np.random.rand(n_lat, n_lon))},
        coords={"latitude": lats, "longitude": lons}
    )

def _make_landcover_ds(n_lat=50, n_lon=50, n_pseudo_level=9):
    """Synthetic landcover dataset matching the structure expected by cut_topog_data.
    Uses lat/lon in -180..180 range, covering the fp domain ([-40,30] lon, [-15,15] lat).
    Variables: landcover_type (lat,lon), land_binary_mask (lat,lon),
               landcover_fraction (lat,lon,pseudo_level)."""
    lats          = np.linspace(-20.0, 20.0, n_lat)
    lons          = np.linspace(-45.0, 35.0, n_lon)   # covers fp domain [-40, 30]
    pseudo_levels = np.arange(n_pseudo_level)
    return xr.Dataset(
        {
            "landcover_type": (
                ["lat", "lon"],
                np.random.randint(0, 10, (n_lat, n_lon)).astype("int32"),
            ),
            "land_binary_mask": (
                ["lat", "lon"],
                np.random.randint(0, 2, (n_lat, n_lon)).astype("int32"),
            ),
            "landcover_fraction": (
                ["lat", "lon", "pseudo_level"],
                np.random.rand(n_lat, n_lon, n_pseudo_level).astype("float32"),
            ),
        },
        coords={"lat": lats, "lon": lons, "pseudo_level": pseudo_levels},
    )

# ---------- CLI option: --sample-files to use real .nc files ----------
def pytest_report_header(config):
    if config.getoption("--sample-files"):
        return "dataset mode: real .nc sample files (tests/sample_data/)"
    return "dataset mode: synthetic in-memory data"


def pytest_addoption(parser):
    parser.addoption(
        "--sample-files",
        action="store_true",
        default=False,
        help="Run tests using real .nc sample files in tests/sample_data/ instead of synthetic data",
    )

@pytest.fixture
def use_sample_files(request):
    return request.config.getoption("--sample-files")

# ---------- fixtures: synthetic by default, real files with --sample-files ----------
def _resolve_sample_filename(filename, year=SAMPLE_YEAR, month=SAMPLE_MONTH):
    """Resolve sample filename from template and hardcoded year/month.

    Supported forms:
      - "fp_sample_{ym}.nc"
      - "fp_sample_{year}{month}.nc"
      - "fp_sample_" (legacy prefix form -> appends YYYYMM.nc)
      - "topog_sample.nc" (unchanged)
    """
    ym = f"{year}{month}"
    if "{ym}" in filename or "{year}" in filename or "{month}" in filename:
        # what does filename.format(year=year, month=month, ym=ym) do? 
        return filename.format(year=year, month=month, ym=ym)
    if filename.endswith("_"):
        return f"{filename}{ym}.nc"
    return filename


def _load_or_make(request, filename, make_fn, return_path=False):
    """Load or make dataset, optionally returning path instead of data.
    
    Args:
        request: pytest request object
        filename: name of the file in tests/sample_data/
        make_fn: function to generate synthetic data if not using --sample-files
        return_path: if True, return path to file (only with --sample-files);
                     if False, return loaded dataset or synthetic data
    
    Returns:
        If return_path=True: path to file (str) if --sample-files, else None
        If return_path=False: xr.Dataset (loaded or synthetic)
    """
    if request.config.getoption("--sample-files"):
        resolved_filename = _resolve_sample_filename(filename)
        path = os.path.join(FIXTURE_DIR, resolved_filename)
        if not os.path.exists(path):
            pytest.fail(
                f"{resolved_filename} not found "# in {FIXTURE_DIR} "
                "- run generate_sample_data.py first, or remove the --sample-files option to use synthetic data"
            )
        if return_path:
            return path
        return xr.open_dataset(path)
    if return_path:
        return None
    return make_fn()

@pytest.fixture
def fp_ds(request):
    return _load_or_make(request, "fp_sample_{ym}.nc", _make_fp_ds)

@pytest.fixture
def met_ds(request):
    return _load_or_make(request, "met_sample_{ym}.nc", _make_met_ds)

@pytest.fixture
def topog_ds(request):
    return _load_or_make(request, "topog_sample.nc", _make_topog_ds)

@pytest.fixture
def landcover_ds(request):
    return _load_or_make(request, "landcover_sample.nc", _make_landcover_ds)

# ---------- fixtures: paths to sample files ----------
@pytest.fixture
def test_fp_datadir(request):
    """Path to fp sample file, only available with --sample-files."""
    path = _load_or_make(request, "fp_sample_{ym}.nc", None, return_path=True)
    #if path is None:
    #    pytest.skip("--sample-files not specified; test requires real sample files")
    return path

@pytest.fixture
def test_met_datadir(request):
    """Path to met sample file, only available with --sample-files."""
    path = _load_or_make(request, "met_sample_{ym}.nc", None, return_path=True)
    #if path is None:
    #    pytest.skip("--sample-files not specified; test requires real sample files")
    return path

@pytest.fixture
def test_topog_path(request):
    """Path to topog sample file, only available with --sample-files."""
    path = _load_or_make(request, "topog_sample.nc", None, return_path=True)
    #if path is None:
    #    pytest.skip("--sample-files not specified; test requires real sample files")
    return path

@pytest.fixture
def test_landcover_path(request):
    """Path to landcover sample file, only available with --sample-files."""
    path = _load_or_make(request, "landcover_sample.nc", None, return_path=True)
    #if path is None:
    #    pytest.skip("--sample-files not specified; test requires real sample files")
    return path