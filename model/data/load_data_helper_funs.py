import numpy as np
import xarray as xr
import pandas as pd
import glob
import dask
import sys
import os

    
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
            if v not in met.data_vars:
                print("variable ", v, " not found in met file")
                variables.remove(v)
        vars_to_drop = list(set(list(met.data_vars))- set(variables) - set(protected_variables))
        met = met.drop_vars(vars_to_drop)

        met = met.drop_vars(list(set(list(met.coords))- set(protected_coords)))
    return met

def _static_var_topog(topog_ds, coordinate_ds):
    print(topog_ds.coords)
    if "time" not in topog_ds.coords:
        print(">!>!>!")
        topog = topog_ds.topog.broadcast_like(coordinate_ds, exclude=["lat", "lon"])
        print(topog)
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
    coordinate_ds = coordinate_ds.drop_vars("landcover_level")
    return coordinate_ds

def _static_var_sin_lat_coords(coordinate_ds):
    coordinate_ds = coordinate_ds.assign({"sin_lat_coords":np.sin(coordinate_ds.lat_coords)})
    return coordinate_ds

def _static_var_sin_lon_coords(coordinate_ds):
    coordinate_ds = coordinate_ds.assign({"sin_lon_coords":np.sin(coordinate_ds.lon_coords)})
    return coordinate_ds

def _static_var_cos_lat_coords(coordinate_ds):
    coordinate_ds = coordinate_ds.assign({"cos_lat_coords":np.cos(coordinate_ds.lat_coords)})
    return coordinate_ds

def _static_var_cos_lon_coords(coordinate_ds):
    coordinate_ds = coordinate_ds.assign({"cos_lon_coords":np.cos(coordinate_ds.lon_coords)})
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

    print(np.shape(grid_coords))
    coord = np.dstack([grid_coords[1]]*coordinate_ds.fp_time.size).transpose([2,0,1])
    coordinate_ds = coordinate_ds.assign({"x_coords":(("fp_time", "lat", "lon"), coord)})

    return coordinate_ds 

def _binary_centre(coordinate_ds):
    assert coordinate_ds.lat.size == coordinate_ds.lon.size, "_binary_centre function only works for square datasets!"

    centre = int(coordinate_ds.lat.size/2)
    grid = np.zeros((coordinate_ds.lat.size, coordinate_ds.lon.size)) -1
    grid[centre, centre] = 1
    coord = np.dstack([grid]*coordinate_ds.fp_time.size).transpose([2,0,1])
    coordinate_ds = coordinate_ds.assign({"binary_centre":(("fp_time", "lat", "lon"), coord)})

def _xy_distance_centre(coordinate_ds):
    assert coordinate_ds.lat.size == coordinate_ds.lon.size, "_binary_centre function only works for square datasets!"

    centre = int(coordinate_ds.lat.size/2)
    grid_coords = np.meshgrid(np.arange(coordinate_ds.lat.size), np.arange(coordinate_ds.lon.size)) 
    distance = np.sqrt(np.abs(grid_coords[0]-centre)**2 + np.abs(grid_coords[1]-centre)**2)

    coord = np.dstack([distance]*coordinate_ds.fp_time.size).transpose([2,0,1])
    coordinate_ds = coordinate_ds.assign({"xy_distance_centre":(("fp_time", "lat", "lon"), coord)})


    
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
                                NotImplemented
    }    
            
    return static_variables_functions

