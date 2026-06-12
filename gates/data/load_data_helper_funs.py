import numpy as np
import xarray as xr
import pandas as pd
import glob
import dask
import sys
import os

import numpy as np

def haversine(lat1, lon1, lat2, lon2, radius=6371.0, degrees=False):
    """
    Compute great-circle distance using the haversine formula.

    Parameters
    ----------
    lat1, lon1, lat2, lon2 : array-like or scalar
        Coordinates of the two points.
        Interpreted as degrees if degrees=True, otherwise radians.
    radius : float, default=6371.0
        Sphere radius (Earth radius in km by default).
    degrees : bool, default=False
        If True, inputs are assumed to be in degrees and are converted
        to radians internally.

    Returns
    -------
    distance : array-like or scalar
        Great-circle distance in the same units as `radius`.

    Supports NumPy arrays, xarray DataArrays, and PyTorch tensors (including CUDA).
    """
    import torch
    _is_tensor = isinstance(lat1, torch.Tensor) or isinstance(lon1, torch.Tensor)

    if _is_tensor:
        math = torch
        deg2rad = lambda x: x * (torch.pi / 180.0)
    else:
        math = np
        deg2rad = np.deg2rad

    if degrees:
        lat1, lon1, lat2, lon2 = deg2rad(lat1), deg2rad(lon1), deg2rad(lat2), deg2rad(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    )
    c = 2 * math.asin(math.sqrt(a)) if _is_tensor else 2 * np.arcsin(np.sqrt(a))

    return radius * c


def select_met_levels(met, levels=None):
    # subset the right levels and variables, as specified in the inputs
    if levels is not None and len(levels)>0 and "levels" in met.coords:
        for lev in levels:
            if lev not in met.levels.values: 
                print("level ", lev, "cannot be found in the met file")
                levels.remove(lev)

        met = met.sel(levels=levels)


    return met

def select_met_variables(met, variables=None):
    protected_variables = ["fp_time", "lat_coords", "lon_coords"]
    protected_coords = ["levels", "time", "time_delta", "lat", "lon"]
    if variables is not None:
        for v in variables:
            if v not in met.data_vars and v != "wind_speed" and v != "wind_angle":
                print("variable ", v, " not found in met file")
                variables.remove(v)
        vars_to_drop = list(set(list(met.data_vars))- set(variables) - set(protected_variables))
        met = met.drop_vars(vars_to_drop)

        met = met.drop_vars(list(set(list(met.coords))- set(protected_coords)))
    return met

def _static_var_topog(topog_ds, coordinate_ds):
    if "time" not in topog_ds.coords:
        topog = topog_ds.topog.broadcast_like(coordinate_ds, exclude=["lat", "lon"])
        coordinate_ds = coordinate_ds.assign({"topog":topog})
        
    else:
        coordinate_ds = coordinate_ds.assign({"topog":topog_ds.topog.rename({"time":"fp_time"})})

    return coordinate_ds

def _static_var_landcover(topog_ds, coordinate_ds):
    coordinate_ds = coordinate_ds.assign({"landcover":topog_ds.landcover.rename({"time":"fp_time"})})
    return coordinate_ds

def _static_var_landcover_disaggregated(topog_ds, coordinate_ds):
    n_landcover_types=10
    # extract the ten types of landcover as separate 2D inputs
    for landcover_type in range(n_landcover_types):
        coordinate_ds = coordinate_ds.assign({f"landcover_type_{landcover_type}":topog_ds.disaggregated_landcover.sel(landcover_level=landcover_type).rename({"time":"fp_time"})})
    if "landcover_level" in coordinate_ds.variables:
        coordinate_ds = coordinate_ds.drop_vars("landcover_level")
    return coordinate_ds

def _static_var_sin_lat_coords(coordinate_ds):
    deg_to_rad = np.pi/180
    coordinate_ds = coordinate_ds.assign({"sin_lat_coords":np.sin(coordinate_ds.lat_coords * deg_to_rad)})
    return coordinate_ds

def _static_var_sin_lon_coords(coordinate_ds):
    deg_to_rad = np.pi/180
    coordinate_ds = coordinate_ds.assign({"sin_lon_coords":np.sin(coordinate_ds.lon_coords * deg_to_rad)})
    return coordinate_ds

def _static_var_cos_lat_coords(coordinate_ds):
    deg_to_rad = np.pi/180
    coordinate_ds = coordinate_ds.assign({"cos_lat_coords":np.cos(coordinate_ds.lat_coords * deg_to_rad)})
    return coordinate_ds

def _static_var_cos_lon_coords(coordinate_ds):
    deg_to_rad = np.pi/180
    coordinate_ds = coordinate_ds.assign({"cos_lon_coords":np.cos(coordinate_ds.lon_coords * deg_to_rad)})
    return coordinate_ds

def _static_var_x_coords(coordinate_ds):
    # create a mesh with [0,0] at the release point, in the x coordinate (longitude)
    grid_coords = np.meshgrid(np.arange(coordinate_ds.lon.size)-int(coordinate_ds.lon.size/2), np.arange(coordinate_ds.lat.size) -int(coordinate_ds.lat.size/2))

    coord = np.dstack([grid_coords[0]]*coordinate_ds.fp_time.size).transpose([2,0,1])
    coordinate_ds = coordinate_ds.assign({"y_coords":(("fp_time", "lat", "lon"), coord)})

    return coordinate_ds 

def _static_var_y_coords(coordinate_ds):
    # create a mesh with [0,0] at the release point, in the y coordinate (latitude)
    grid_coords = np.meshgrid(np.arange(coordinate_ds.lon.size)-int(coordinate_ds.lon.size/2), np.arange(coordinate_ds.lat.size) -int(coordinate_ds.lat.size/2))

    coord = np.dstack([grid_coords[1]]*coordinate_ds.fp_time.size).transpose([2,0,1])
    coordinate_ds = coordinate_ds.assign({"x_coords":(("fp_time", "lat", "lon"), coord)})

    return coordinate_ds 

def _domain_distance_release(fp_data, coordinate_ds):
    # calculates the haversine distance from each grid point to the release point for each fp
    lat_vals = np.radians(fp_data.lat.values)  # shape (n_lat,)
    lon_vals = np.radians(fp_data.lon.values)  # shape (n_lon,)
    release_lat = np.radians(fp_data.release_lat.values)  # shape (n_time,)
    release_lon = np.radians(fp_data.release_lon.values)  # shape (n_time,)

    lat_grid, lon_grid = np.meshgrid(lat_vals, lon_vals, indexing='ij')  # shape (n_lat, n_lon)

    # Broadcast all to 3d
    lat_grid_3d = lat_grid[:, :, np.newaxis]  
    lon_grid_3d = lon_grid[:, :, np.newaxis]  

    release_lat_3d = release_lat[np.newaxis, np.newaxis, :]  
    release_lon_3d = release_lon[np.newaxis, np.newaxis, :] 

    distances = haversine(lat_grid_3d, lon_grid_3d, release_lat_3d, release_lon_3d)

    coordinate_ds = coordinate_ds.assign({"distance_release":(("fp_time", "lat", "lon"), distances.transpose([2,0,1]))})

    return coordinate_ds

def _domain_binary_release(fp_data, coordinate_ds):
    # provides a grid the size of the domain filled with zeros, except 1 at the release point

    lat_vals = fp_data.lat.values  # shape (n_lat,)
    lon_vals = fp_data.lon.values  # shape (n_lon,)
    release_lat = fp_data.release_lat.values  # shape (n_time,)
    release_lon = fp_data.release_lon.values  # shape (n_time,)

    lat_grid, lon_grid = np.meshgrid(lat_vals, lon_vals, indexing='ij')  # shape (n_lat, n_lon)

    # Broadcast all to 3d
    lat_grid_3d = lat_grid[:, :, np.newaxis]  
    lon_grid_3d = lon_grid[:, :, np.newaxis]  

    release_lat_3d = release_lat[np.newaxis, np.newaxis, :]  
    release_lon_3d = release_lon[np.newaxis, np.newaxis, :] 


    # Compute squared distances to all grid points for all times
    dist2 = (lat_grid_3d - release_lat_3d)**2 + (lon_grid_3d - release_lon_3d)**2  
    min_indices = np.argmin(dist2.reshape(len(lat_vals)*len(lon_vals), -1), axis=0)  
    lat_indices, lon_indices = np.unravel_index(min_indices, (len(lat_vals), len(lon_vals)))

    marker = np.zeros((len(lat_vals), len(lon_vals), len(release_lat)), dtype=np.uint8)

    marker[lat_indices, lon_indices, np.arange(len(release_lat))] = 1

    coordinate_ds = coordinate_ds.assign({"binary_release":(("fp_time", "lat", "lon"), marker.transpose([2,0,1]))})
    
    return coordinate_ds




def _earth_distance_centre(fp_data, coordinate_ds):
    # Haversine distance from each grid cell (using real lat/lon coords) to the release point
    lat_coords = np.radians(fp_data.lat_coords.values)           # (n_time, n_lat)
    lon_coords = np.radians(fp_data.lon_coords.values)           # (n_time, n_lon)
    release_lat = np.radians(fp_data.release_lat.values)         # (n_time,)
    release_lon = np.radians(fp_data.release_lon.values)         # (n_time,)

    lat_3d = lat_coords[:, :, np.newaxis]                        # (n_time, n_lat, 1)
    lon_3d = lon_coords[:, np.newaxis, :]                        # (n_time, 1, n_lon)
    release_lat_3d = release_lat[:, np.newaxis, np.newaxis]      # (n_time, 1, 1)
    release_lon_3d = release_lon[:, np.newaxis, np.newaxis]      # (n_time, 1, 1)

    distances = haversine(lat_3d, lon_3d, release_lat_3d, release_lon_3d)  # (n_time, n_lat, n_lon)

    coordinate_ds = coordinate_ds.assign({"earth_distance_centre":(("fp_time", "lat", "lon"), distances)})

    return coordinate_ds



def _binary_centre(coordinate_ds):
    assert coordinate_ds.lat.size == coordinate_ds.lon.size, "_binary_centre function only works for square datasets!"

    centre = int(coordinate_ds.lat.size/2)
    grid = np.zeros((coordinate_ds.lat.size, coordinate_ds.lon.size)) -1
    grid[centre, centre] = 1
    coord = np.dstack([grid]*coordinate_ds.fp_time.size).transpose([2,0,1])
    coordinate_ds = coordinate_ds.assign({"binary_centre":(("fp_time", "lat", "lon"), coord)})

    return coordinate_ds

def _xy_distance_centre(coordinate_ds):
    assert coordinate_ds.lat.size == coordinate_ds.lon.size, "_xy_distance_centre function only works for square datasets!"

    centre = int(coordinate_ds.lat.size/2)
    grid_coords = np.meshgrid(np.arange(coordinate_ds.lat.size), np.arange(coordinate_ds.lon.size)) 
    distance = np.sqrt(np.abs(grid_coords[0]-centre)**2 + np.abs(grid_coords[1]-centre)**2)

    coord = np.dstack([distance]*coordinate_ds.fp_time.size).transpose([2,0,1])
    coordinate_ds = coordinate_ds.assign({"xy_distance_centre":(("fp_time", "lat", "lon"), coord)})

    return coordinate_ds

    
def get_static_variables_functions():
    # dict of arguments and the functions that they return. can probably be made dynamic, eg imported, or more functions could be passed in an optional arg
    static_variables_functions = {"lat_coords":
                                NotImplemented,
                                "lon_coords":
                                NotImplemented,
                                "sin_lat_coords":
                                _static_var_sin_lat_coords,
                                "sin_lon_coords":
                                _static_var_sin_lon_coords,
                                "cos_lat_coords":
                                _static_var_cos_lat_coords,
                                "cos_lon_coords":
                                _static_var_cos_lon_coords,
                                "x_coords":
                                _static_var_x_coords,
                                "y_coords":
                                _static_var_y_coords,
                                "binary_centre":
                                _binary_centre,
                                "xy_distance_centre":
                                _xy_distance_centre, 
                                "earth_distance_centre":
                                _earth_distance_centre,
                                "lat_degrees_distance":
                                NotImplemented,
                                "lon_degrees_distance":
                                NotImplemented,
                                "topog":
                                _static_var_topog, 
                                "sea_mask":
                                NotImplemented, 
                                "landcover":
                                _static_var_landcover, 
                                "landcover_disaggregated":
                                _static_var_landcover_disaggregated,  
                                "land_cover_disaggregated_binary":
                                NotImplemented,
                                "domain_binary_release":
                                _domain_binary_release,
                                "domain_distance_release":
                                _domain_distance_release,
    }    
            
    return static_variables_functions

