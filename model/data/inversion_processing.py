import numpy as np
import xarray as xr
import pandas as pd
import glob
import dask
import sys
import os


def cut_dataset_to_intersect(year, month, to_cut_footprint_path, target_footprints_path, to_save_footprints_path):
    """
    If a dataset is a subset of another, cut the larger one to have only the same timestamps as the target one

    should streamline this to take more general inputs
    
    """
    if name_footprint_path is None:
        name_footprint_path = "/group/chemistry/acrg/LPDM/fp_NAME_pre20210701/SOUTHAMERICA/GOSAT-BRAZIL-column_SOUTHAMERICA_"

    if emulated_footprints_path is None:
        emulated_footprints_path = "/group/chemistry/acrg/LPDM/fp_Elena/satellite_emulated_logv4/def_lc_biascorrected/SOUTHAMERICA/GOSAT-BRAZIL-column_SOUTHAMERICA_"

    if to_save_footprints_path is None:
        to_save_footprints_path = "/group/chemistry/acrg/LPDM/fp_Elena/satellite_emulated_logv4/NAME_fps_intersect/SOUTHAMERICA/GOSAT-BRAZIL-column_SOUTHAMERICA_"

    to_cut_fps = xr.open_dataset(f"{to_cut_footprint_path}{year}{month}.nc")
    target_fps = xr.open_dataset(f"{target_footprints_path}{year}{month}.nc")

    try:
        print(f"{year}{month}: removing {len(to_cut_fps.time) - len(target_fps.time)} indices")

        cut_fps = to_cut_fps.sel(time=target_fps.time)

        
    
    except KeyError: # if not a subset for whatever reason...
        inters = np.intersect1d(to_cut_fps.time.values, target_fps.time.values)
        cut_fps = to_cut_fps.sel(time=inters)

        print(f"something went wrong so instead using intersect method")
    

    cut_fps.to_netcdf(f"{to_save_footprints_path}{year}{month}.nc")
    
    print(f"saved at {to_save_footprints_path}{year}{month}.nc")


    return None


def cut_dataset_to_square(year, month, size, to_cut_footprint_path, to_save_footprints_path):
    """
    Load a footprint dataset and save a copy, but where the footprints have been cropped to a square of size x size of the measurement point 
    """
    if to_cut_footprint_path is None:
        footprint_path = "/group/chemistry/acrg/LPDM/fp_NAME_pre20210701/NORTHAFRICA/GOSAT-SAHARA-column_NORTHAFRICA_"

    if to_save_footprints_path is None:
        to_save_footprints_path = "/group/chemistry/acrg/LPDM/fp_Elena/satellite_emulated_NA/NAME_fps_square/NORTHAFRICA/GOSAT-SAHARA-column_NORTHAFRICA_"


    original_fps = xr.open_dataset(f"{to_cut_footprint_path}{year}{month}.nc")

    # make an array of zeros, and keep only the relevant values
    cut_fps = np.zeros_like(original_fps.fp.values)
    release_idxs = get_release_idxs(original_fps)

    half = int(size/2)

    for rel_unique in np.unique(release_idxs, axis=0):
        idxs = np.where((release_idxs == rel_unique).all(axis=1))[0]

        lower_lat = np.max((0, rel_unique[0]-half))
        lower_lon = np.max((0, rel_unique[1]-half))
        upper_lat = np.min((len(original_fps.lat.values), rel_unique[0]+half))
        upper_lon = np.min((len(original_fps.lon.values), rel_unique[1]+half))

        # cut the part of the footprint within domain area
        f = original_fps.fp.values[lower_lat:upper_lat,lower_lon:upper_lon,idxs]


        cut_fps[lower_lat:upper_lat,lower_lon:upper_lon, idxs] = f

        
    full_emulated_footprints = original_fps.copy()
    full_emulated_footprints["fp"] = (("lat", "lon", "time"), cut_fps)

    full_emulated_footprints.attrs["cut to domain size"] = str(size)
    full_emulated_footprints.to_netcdf(f"{to_save_footprints_path}{year}{month}.nc")
    
    print(f"saved at {to_save_footprints_path}{year}{month}.nc")