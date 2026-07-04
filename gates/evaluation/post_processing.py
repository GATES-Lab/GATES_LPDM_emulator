import numpy as np
import xarray as xr

def threshold_fps(fps, threshold, fp_label=["fp_pred"]):
    """Threshold footprint variables, zeroing out values below the threshold.

    For each variable named in ``fp_label``, saves the thresholded result as a new
    variable named ``"{original_variable_name}_thres"``.

    Args:
        fps (xr.Dataset): Dataset containing the footprint variable(s) to threshold.
        threshold (float): Values below this threshold are set to zero.
        fp_label (list[str], optional): Names of the variables in ``fps`` to
            threshold. Defaults to ["fp_pred"].

    Returns:
        xr.Dataset: ``fps`` with the new ``"{label}_thres"`` variable(s) added.
    """
    for label in fp_label:
        thres_label = f"{label}_thres"
        fps[thres_label] = fps[label].where(fps[label] >= threshold, 0)
    return fps