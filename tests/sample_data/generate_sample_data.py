"""
Run once to create small test fixtures from real data.
Requires access to ACRG data paths.
"""
import xarray as xr
import glob, os

OUT_DIR = os.path.dirname(__file__)

# Footprints: load 1 month, keep every 10th timestep
# TODO replace this with paths from config files!

fp = xr.open_mfdataset(sorted(glob.glob(
    "/group/chem/acrg/LPDM/fp_NAME_pre20210701/SOUTHAMERICA/*BRAZIL*SOUTHAMERICA_201601*.nc"
)), combine="by_coords")
fp.isel(time=slice(0, None, 10)).to_netcdf(f"{OUT_DIR}/fp_sample.nc")

# Met: same approach
met = xr.open_mfdataset(sorted(glob.glob(
    "/group/chem/acrg/met_archive/UM/SOUTHAMERICA/SOUTHAMERICA_Met_201601*.nc"
)), combine="nested", concat_dim="time")
met.sel(model_level_number=[3, 15])[["x_wind", "y_wind"]].to_netcdf(f"{OUT_DIR}/met_sample.nc")

# Topog + landcover: just crop to the domain of fp
topog = xr.load_dataset("/group/chem/acrg/LPDM/topog_NAME/TopogUMG_Mk8_global.nc")
topog.sel(latitude=slice(-20, 10), longitude=slice(-65, -35)).to_netcdf(f"{OUT_DIR}/topog_sample.nc")

lc = xr.load_dataset("/group/chem/acrg/LPDM/topog_NAME/land_cover.nc")
lc.sel(lat=slice(-20, 10), lon=slice(295, 325)).to_netcdf(f"{OUT_DIR}/landcover_sample.nc")

print("Fixtures written to", OUT_DIR)