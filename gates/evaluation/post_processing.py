import numpy as np
import xarray as xr

def threshold_fps(fps, threshold, fp_label=["fp_pred"]):
    """
    Takes in an xarray and for each variable in fp_labels, applies the given threshold to set values below the threshold to zero. It saves it as a new variable in the xarray with the name "{original_variable_name}_thres".
    """
    for label in fp_label:
        thres_label = f"{label}_thres"
        fps[thres_label] = fps[label].where(fps[label] >= threshold, 0)
    return fps