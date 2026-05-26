import xarray as xr
from gates.config import get_config
from glob import glob
import pandas as pd
import numpy as np

def load_cams_data(domain, year=2016, month="*", species="ch4", cfg=None):
    """
    Load CAMS reanalysis data for the given domain, year, month, and species. The data is expected to be in NetCDF format and located in the directory specified by the configuration file. The function will look for files matching the pattern "{bc_datadir}/{domain}/{species}_{domain}_{year}{month}*.nc". If multiple files are found, they will be combined along the time dimension. The function returns an xarray Dataset containing the loaded data.

    Args:
        domain (str): The domain for which to load the data (e.g., "SOUTHAMERICA").
        year (int): The year for which to load the data (default: 2016).
        month (str or int): The month for which to load the data. Can be an integer (1-12) or a string (e.g., "01", "*") (default: "*", which means all months).
        species (str): The species to load (default: "ch4").
        cfg: The configuration object containing the data directory paths. If None, the function will call get_config() to load the configuration.
    Returns:
        xarray.Dataset: The loaded CAMS reanalysis data.
    """
    if cfg is None:
        cfg = get_config()
    
    bc_datadir = cfg.bc_datadir

    # if month is an int, format as a str
    if isinstance(month, int):
        month = f"{month:02d}"

    file_pattern = f"{bc_datadir}/{domain}/{species}_{domain}_{year}{month}*.nc"
    found_files = sorted(glob(file_pattern))
    if len(found_files) == 0:
        raise FileNotFoundError(f"No files found at {file_pattern}")
    
    
    if any("climatology" in file for file in found_files):
        print(f"WARNING: Climatology files found at {file_pattern}. These will be used instead of the regular files.")

    bc_data = xr.open_mfdataset(found_files, combine='nested',concat_dim="time", engine="h5netcdf")
    bc_data = bc_data.sortby("time")

    # append to the attrs the file path and whether it's a climatology file or not
    bc_data.attrs["file_path"] = file_pattern
    bc_data.attrs["is_climatology"] = any("climatology" in file for file in found_files)
    bc_data.attrs["domain"] = domain
    bc_data.attrs["species"] = species

    # return as float32
    bc_data = bc_data.astype(np.float32)

    return bc_data


def calculate_bg(fp, bc_data):
    """
    Calculate the background concentration for each side of the domain (north, south, east, west) by multiplying the particle locations in the footprints with the corresponding boundary condition data from the CAMS reanalysis, and summing the height, latitude, and longitude dimensions to get a single background concentration value for each time point and direction. The result is returned as an xarray Dataset with variables "north", "south", "east", and "west".

    """
    
    if not all(var in fp.variables for var in ["particle_locations_n", "particle_locations_s", "particle_locations_e", "particle_locations_w"]):
        raise ValueError("The input fp dataset must contain the variables 'particle_locations_n', 'particle_locations_s', 'particle_locations_e', and 'particle_locations_w'.")
    

    # --- 1. Time matching ---
    ### NOTE this actually should be done differently! each 
    tol = pd.Timedelta("32D")
    """
    nearest = bc_data.indexes["time"].get_indexer(
        pd.DatetimeIndex(fp.time.values), method="nearest", tolerance=tol)
    nan_idxs = pd.DatetimeIndex(fp.time.values)[nearest == -1]
    
    if len(nan_idxs):
        print(f"Warning: {len(nan_idxs)} time points in fp have no matching time point in bc_data within a tolerance of {tol}. These will be set to NaN in the output.")
    nearest_safe = np.where(nearest != -1, nearest, 0)
    bc_matched = bc_data.isel(time=xr.DataArray(nearest_safe, dims="time"))
    bc_matched["time"] = fp.time.values
    if len(nan_idxs):
        bc_matched = bc_matched.where(
            xr.DataArray(nearest != -1, dims="time"), np.nan)
    """
    bc_matched = bc_data.reindex(
    time=fp.time,
    method="ffill",
    tolerance=pd.Timedelta("32D") )  
    
    # bg is a dataset with variables "bc_n", "bc_s", "bc_e", "bc_w" which is each the multiplication of the respective particle location variable in fp with the matched bc data
    bg = xr.Dataset()
    var_names_dict = {"n":"north", "s":"south", "e":"east", "w":"west"}
    for direction in ["n", "s", "e", "w"]:
        particle_var = f"particle_locations_{direction}"
        bc_var = f"vmr_{direction}"
        bg[var_names_dict[direction]] = fp[particle_var] * bc_matched[bc_var]  # multiply by the first variable in the bc data, which should be the species concentration
    
    bg = bg.sum(dim=["height", "lat", "lon"], skipna=True)
    bg["summed"] = bg.north + bg.south + bg.east + bg.west

    return bg

def calculate_detrending_factor(bc_file, boundary="south", height_index=1):
    """
    Calculate a detrending factor for the boundary condition correction by taking the mean value of the specified boundary direction (e.g., "south") at the specified height index across all time points in the bc_file. This factor can be used to detrend the boundary condition correction by dividing the correction values by this factor, which helps to remove any systematic bias in the boundary condition data.
    """
    var_name = f"vmr_{boundary[0]}"
    mean_dim = "lat" if boundary in ["east", "west"] else "lon"
    detrending_factor = bc_file[var_name].isel(height=height_index).mean(dim=mean_dim)

    return detrending_factor
