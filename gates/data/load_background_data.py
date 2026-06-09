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
        year (int or list): The year(s) for which to load the data (default: 2016).
        month (str or int): The month for which to load the data. Can be an integer (1-12) or a string (e.g., "01", "*") (default: "*", which means all months).
        species (str): The species to load (default: "ch4").
        cfg: The configuration object containing the data directory paths. If None, the function will call get_config() to load the configuration.

    Returns:
        xarray.Dataset: The loaded CAMS reanalysis data.
    """
    if cfg is None:
        cfg = get_config()
    
    bc_datadir = cfg.bc_datadir

    # if year is an int or str, make it a list and concat all 
    if isinstance(year, (int, str)):
        year = [year]

    # if month is an int, format as a str
    if isinstance(month, int):
        month = f"{month:02d}"

    bc_files = []
    for y in year:
        file_pattern = f"{bc_datadir}/{domain}/{species}_{domain}_{y}{month}*.nc"
        found_files = sorted(glob(file_pattern))
        if len(found_files) == 0:
            raise FileNotFoundError(f"No files found at {file_pattern}")
        

        if any("climatology" in file for file in found_files):
            print(f"WARNING: Climatology files found at {file_pattern}. These will be used instead of the regular files.")

        bc_data = xr.open_mfdataset(found_files, combine='nested',concat_dim="time", engine="h5netcdf")
        bc_data = bc_data.sortby("time")
        bc_files.append(bc_data)
    
    bc_data = xr.concat(bc_files, dim="time")

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
    Calculate background concentration for each domain boundary by convolving particle locations from footprints with CAMS boundary condition data. Convolving means doing element-wise multiplication and summing over the spatial dimensions. 

    Args:
        fp (xarray.Dataset): Footprint dataset containing variables 'particle_locations_n/s/e/w'.
        bc_data (xarray.Dataset): CAMS boundary condition dataset containing variables 'vmr_n/s/e/w', aligned to monthly or finer resolution.
    Returns:
        xarray.Dataset: Background concentrations with variables 'north', 'south', 'east', 'west', and 'summed' (their sum), each indexed by time.
    """
    
    if not all(var in fp.variables for var in ["particle_locations_n", "particle_locations_s", "particle_locations_e", "particle_locations_w"]):
        raise ValueError("The input fp dataset must contain the variables 'particle_locations_n', 'particle_locations_s', 'particle_locations_e', and 'particle_locations_w'.")
    

    # --- 1. Time matching ---
    tol = pd.Timedelta("32D")

    bc_matched = bc_data.reindex(
    time=fp.time,
    method="ffill",
    tolerance=tol )  
    
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

