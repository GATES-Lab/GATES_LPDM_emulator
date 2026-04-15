# HOW TO TESTS

## Running tests
You can run the tests either with realistic sample data (see #2 below on how to generate it) or with synthetic data. Use the following commands:

- synthetic (default):

```bash
python -m pytest tests/ -v
```

- real sample files:

```bash
python -m pytest tests/ --sample-files
```

To avoid deprecation warnings, add `-Wignore::DeprecationWarning` to the commands above.

The rest of this file describes how test data is currently configured in this repository.

## 1) Expected sample files and coordinate conventions

The pytest fixtures in [tests/conftest.py](tests/conftest.py) switch between:
- real NetCDF files in [tests/sample_data](tests/sample_data) when `--sample-files` is set
- synthetic in-memory datasets otherwise

Current expected sample files are:
- `fp_sample_201601.nc`
- `met_sample_201601.nc`
- `topog_sample.nc`
- `landcover_sample.nc`

Filename resolution is handled by `_resolve_sample_filename(...)` in [tests/conftest.py](tests/conftest.py). The hardcoded defaults are:
- `SAMPLE_YEAR = "2016"`
- `SAMPLE_MONTH = "01"`
which are used across test.
TODO: Make these importable from config!


### Footprints (`fp_sample_YYYYMM.nc`)

Expected structure (as validated/used in tests):
- dims: `time`, `lat`, `lon` (or equivalent lat/lon naming in real files)
- variables: `fp`, `release_lat`, `release_lon`

### Meteorology (`met_sample_YYYYMM.nc`)

Expected structure:
- dims: `time`, `model_level_number`, `lat`, `lon`
- variables: `x_wind`, `y_wind`, `atmosphere_boundary_layer_thickness`
- currently generated from model levels `[3, 15]` by [tests/sample_data/generate_sample_data.py](tests/sample_data/generate_sample_data.py)

Note: the level dimension can also be `levels` (`model_level_number` is renamed to `levels` if it exists)

### Topography (`topog_sample.nc`)

Expected structure:
- variable: `surface_altitude`
- coords can be raw `latitude`/`longitude` or already-renamed `lat`/`lon`


### Landcover (`landcover_sample.nc`)

Expected structure for most processing paths:
- dims: `lat`, `lon`, `pseudo_level`
- variables: `landcover_type`, `land_binary_mask`, `landcover_fraction`

Tests also accept the raw variable form:
- `land_cover`


## 2) How to generate sample files (`generate_sample_data.py`)

Script location: [tests/sample_data/generate_sample_data.py](tests/sample_data/generate_sample_data.py)

From repository root, run:

```bash
python tests/sample_data/generate_sample_data.py
```

What it does:
- finds footprint files matching `FP_GLOB`
- finds met files matching `MET_GLOB`
- loads full topography and landcover files from hardcoded paths
- writes the four sample files into [tests/sample_data](tests/sample_data)

Important details from current script:
- month/year are hardcoded to `2016-01`
- footprint sample keeps every 10th timestep
- met sample keeps variables `x_wind`, `y_wind`, `atmosphere_boundary_layer_thickness`
- met sample keeps levels `[3, 15]`
- source paths are currently hardcoded to ACRG locations

TODO: edit to make importable from config!

If no files are found for glob patterns, the script raises `FileNotFoundError`.

## 3) What the mock (synthetic) files look like when not using sample files

When you run `pytest` without `--sample-files`, fixtures in [tests/conftest.py](tests/conftest.py) return synthetic `xarray.Dataset` objects generated in memory:

- `_make_fp_ds(...)`
  - `fp(time, lat, lon)`
  - `release_lat(time)`
  - `release_lon(time)`
  - random timestamps inside the selected month (Jan 2016 as default), then sorted by time

- `_make_met_ds(...)`
  - `x_wind(time, model_level_number, lat, lon)`
  - `y_wind(time, model_level_number, lat, lon)`
  - `atmosphere_boundary_layer_thickness(time, lat, lon)`
  - 3-hourly timestamps across the selected month
  - default levels `[3, 5, 10]`

- `_make_topog_ds(...)`
  - `surface_altitude(latitude, longitude)`

- `_make_landcover_ds(...)`
  - `landcover_type(lat, lon)`
  - `land_binary_mask(lat, lon)`
  - `landcover_fraction(lat, lon, pseudo_level)`

These synthetic datasets are intentionally shaped to match what the loading/cutting code expects.

## 4) How to use data objects in tests

### Basic fixture usage

Use dataset fixtures directly as test arguments:

```python
def test_fp_format(fp_ds):
    assert "fp" in fp_ds.data_vars
```

Relevant fixtures from [tests/conftest.py](tests/conftest.py):
- dataset objects: `fp_ds`, `met_ds`, `topog_ds`, `landcover_ds`
- sample-file paths (only meaningful with `--sample-files`):
  - `test_fp_datadir`, `test_met_datadir`, `test_topog_path`, `test_landcover_path`
- mode flag: `use_sample_files`

### Build a `LoadSquareSatelliteData` object in either mode

Preferred helper: `make_square_satellite_obj(...)` in [tests/test_helper_funs.py](tests/test_helper_funs.py)

Pattern used in tests:

```python
square_obj = make_square_satellite_obj(
    fp_ds,
    met_ds,
    topog_ds,
    landcover_ds,
    year=2016,
    month="01",
    size=8,
    test_fp_datadir=test_fp_datadir,
    test_met_datadir=test_met_datadir,
    test_topog_path=test_topog_path,
    test_landcover_path=test_landcover_path,
)
```

Behavior:
- if all sample paths are available, it constructs `LoadSquareSatelliteData` from real files
- otherwise, it patches loader methods and uses synthetic fixtures


### Practical tip

Some end-to-end tests are intentionally gated behind `--sample-files` and will skip in synthetic mode. Use `use_sample_files` in tests where behavior should differ by mode.