# tests/conftest.py
import pytest, os, sys
import numpy as np
import pandas as pd
import xarray as xr

# Make the project root importable so tests can do `from model.data.load_data import ...`
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "sample_data")

# ---------- synthetic helpers (hardcoded structure) ----------
def _make_fp_ds(n_time=10, n_lat=50, n_lon=50):
    """Synthetic footprint dataset matching real data structure."""
    times = pd.date_range("2016-01-01", periods=n_time, freq="6h")
    lats  = np.linspace(-15.0,  15.0, n_lat)
    lons  = np.linspace(-40.0, 30.0, n_lon)
    return xr.Dataset(
        {"fp": (["time", "lat", "lon"], np.random.rand(n_time, n_lat, n_lon)),
        "release_lat": (["time"], np.random.uniform(-10.0, 10.0, n_time)), 
        "release_lon": (["time"], np.random.uniform(-30.0, 20.0, n_time))},
        coords={"time": times, "lat": lats, "lon": lons}
    )

def _make_met_ds(n_time=10, n_lat=50, n_lon=50, n_levels=3):
    """Synthetic met dataset matching real data structure."""
    times  = pd.date_range("2016-01-01", periods=n_time, freq="6h")
    lats   = np.linspace(-15.0,  15.0, n_lat)
    lons   = np.linspace(-40.0, 30.0, n_lon)
    levels = [1, 5, 10]
    return xr.Dataset(
        {"x_wind": (["time","model_level_number","lat","lon"], np.random.rand(n_time, n_levels, n_lat, n_lon)),
         "y_wind": (["time","model_level_number","lat","lon"], np.random.rand(n_time, n_levels, n_lat, n_lon))},
        coords={"time": times, "model_level_number": levels, "lat": lats, "lon": lons}
    )

def _make_topog_ds(n_lat=50, n_lon=50):
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
    """Synthetic landcover dataset matching real data structure.
    Uses 'lat'/'lon'/'pseudo_level' dims; lon in 0-360 range (as raw file)."""
    lats         = np.linspace(-20.0, 20.0, n_lat)
    lons_0_360   = np.linspace(290.0, 330.0, n_lon)   # will be wrapped to -180..180
    pseudo_levels = np.arange(n_pseudo_level)
    return xr.Dataset(
        {"land_cover": (["lat", "lon", "pseudo_level"],
                        np.random.rand(n_lat, n_lon, n_pseudo_level))},
        coords={"lat": lats, "lon": lons_0_360, "pseudo_level": pseudo_levels}
    )

# ---------- CLI option: --sample-files to use real .nc files ----------
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
def _load_or_make(request, filename, make_fn):
    if request.config.getoption("--sample-files"):
        path = os.path.join(FIXTURE_DIR, filename)
        if not os.path.exists(path):
            pytest.skip(f"{filename} not found — run tests/sample_data/generate_sample_data.py first")
        return xr.open_dataset(path)
    return make_fn()

@pytest.fixture
def fp_ds(request):
    return _load_or_make(request, "fp_sample.nc", _make_fp_ds)

@pytest.fixture
def met_ds(request):
    return _load_or_make(request, "met_sample.nc", _make_met_ds)

@pytest.fixture
def topog_ds(request):
    return _load_or_make(request, "topog_sample.nc", _make_topog_ds)

@pytest.fixture
def landcover_ds(request):
    return _load_or_make(request, "landcover_sample.nc", _make_landcover_ds)