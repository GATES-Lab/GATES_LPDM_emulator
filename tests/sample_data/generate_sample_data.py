"""Generate small test fixtures from ACRG source data.

Run this once to create NetCDF files in tests/sample_data.
"""

from pathlib import Path
import glob

import xarray as xr


OUT_DIR = Path(__file__).resolve().parent
SAMPLE_YEAR = "2016"
SAMPLE_MONTH = "01"
SAMPLE_YM = f"{SAMPLE_YEAR}{SAMPLE_MONTH}"

# Hardcoded source paths for now. Move these to config later.
FP_GLOB = (
    "/group/chem/acrg/LPDM/fp_NAME_pre20210701/SOUTHAMERICA/"
    f"*BRAZIL*SOUTHAMERICA_{SAMPLE_YM}*.nc"
)
MET_GLOB = (
    "/group/chem/acrg/met_archive/UM/SOUTHAMERICA/"
    f"SOUTHAMERICA_Met_{SAMPLE_YM}*.nc"
)
TOPOG_PATH = "/group/chem/acrg/LPDM/topog_NAME/TopogUMG_Mk8_global.nc"
LANDCOVER_PATH = "/group/chem/acrg/LPDM/topog_NAME/land_cover.nc"


def _require_matches(pattern: str, label: str) -> list[str]:
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No {label} files found for pattern: {pattern}")
    return matches


def generate_fp_sample() -> Path:
    """Create footprint sample, keeping every 10th timestep."""
    fp_files = _require_matches(FP_GLOB, "footprint")
    fp = xr.open_mfdataset(fp_files, combine="by_coords")
    out_path = OUT_DIR / f"fp_sample_{SAMPLE_YM}.nc"
    fp.isel(time=slice(0, None, 10)).to_netcdf(out_path)
    return out_path


def generate_met_sample() -> Path:
    """Create met sample for selected levels and variables."""
    met_files = _require_matches(MET_GLOB, "met")
    met = xr.open_mfdataset(met_files, combine="nested", concat_dim="time")
    out_path = OUT_DIR / f"met_sample_{SAMPLE_YM}.nc"
    met.sel(model_level_number=[3, 15])[
        ["x_wind", "y_wind", "atmosphere_boundary_layer_thickness"]
    ].to_netcdf(out_path)
    return out_path


def generate_topog_landcover_samples() -> tuple[Path, Path]:
    """Create topography and landcover sample files."""
    topog = xr.load_dataset(TOPOG_PATH)
    topog_out = OUT_DIR / "topog_sample.nc"
    topog.to_netcdf(topog_out)

    landcover = xr.load_dataset(LANDCOVER_PATH)
    landcover_out = OUT_DIR / "landcover_sample.nc"
    landcover.to_netcdf(landcover_out)
    return topog_out, landcover_out


def main() -> None:
    fp_out = generate_fp_sample()
    met_out = generate_met_sample()
    topog_out, landcover_out = generate_topog_landcover_samples()

    print("Fixtures written:")
    print(f"- {fp_out}")
    print(f"- {met_out}")
    print(f"- {topog_out}")
    print(f"- {landcover_out}")


if __name__ == "__main__":
    main()