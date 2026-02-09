import xarray as xr
import numpy as np
from scipy.stats import binned_statistic_2d
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd

import sys
sys.path.insert(0, "../..")
from model.metrics_and_figures import plot_binned_map

def plot_seasonal_footprint_histogram(ds, metric="count", degree_bins=2, vmin_vmax=None, return_fig=True):
    """
    Plot histograms of footprint locations, as four panels in a row, one for each season.
    Choose what metric to plot with the metric argument
    - "count" for the number of footprints in each measurement bin,  
    - "mean_sum" for the mean footprint sum in each measurement bin 
    Parameters:
    ds: xarray.Dataset containing 'footprint_lon', 'footprint_lat', 'time', 'lat' and 'lon' variables
    degree_bins: size of the latitude/longitude bins in degrees
    vmin_vmax: color scale limits for the histograms, as a list [vmin, vmax] e.g. [0, 10]
    """

    release_lons = ds.release_lon.values
    release_lats = ds.release_lat.values

    dom_lons = ds.lon.values
    dom_lats = ds.lat.values

    dates = pd.DatetimeIndex(ds.time)

    seasons = {"JFM":["01", "02", "03"], "AMJ":["04", "05", "06"], "JAS":["07", "08", "09"], "OND":["10", "11", "12"]}


    # create the bins for the data
    lat_bins = range(int(np.floor(release_lats.min())- degree_bins/2), int(np.ceil(release_lats.max())+ degree_bins/2) + 1, degree_bins) 
    lon_bins = range(int(np.floor(release_lons.min())- degree_bins/2), int(np.ceil(release_lons.max())+ degree_bins/2) + 1, degree_bins) 

    # set up the figure and axis
    fig = plt.figure(figsize=(12,5),  dpi=400, constrained_layout=True)
    gs = fig.add_gridspec(1, 5, figure=fig, width_ratios=[1, 1, 1,1,0.05])

    ax = np.empty((1, 4), dtype=object)

    for i in range(1):  
        for j in range(4):
            ax[i, j] = fig.add_subplot(gs[i, j], projection=ccrs.PlateCarree()) 

    if vmin_vmax is None:
        print("note that vmin_vmax is None, so the color scale will be different for each season, and not comparable across seasons! To compare across seasons, set vmin_vmax to a list of [vmin, vmax] e.g. [0, 10]")

    for season_n, seas in enumerate(seasons.keys()):
        #season_results[seas] = {"mean_mfs":mean_mfs, "mean_mfs_emulated":mean_mfs_emulated}
        seasonal_idxs = np.where(np.bitwise_and(dates.month>=int(seasons[seas][0]), dates.month<=int(seasons[seas][-1])))[0]

        if len(seasonal_idxs) == 0:
            print(f"No data for season {seas}")
            continue

        if metric == "count":
            metric_args =  {"statistic": "count", "values": None, "metric_name": "Observation count", "cmap": "Greens"}
        elif metric == "mean_sum":
            metric_args =  {"statistic": "mean", "values": ds.fp.isel(time=seasonal_idxs).sum(dim = ["lat", "lon"]).values, "metric_name": "Mean footprint Sum", "cmap": "Reds"} # np.nansum(ds.fp.isel(time=seasonal_idxs).values, axis=(1,2))
        else:
            raise ValueError(f"Invalid metric {metric}. Choose either 'count' or 'mean_sum', or add your own!")

        print(season_n, seas, " - n values:", len(seasonal_idxs))

        # extract 2D-bins
        binned_fps,  lat_edges, lon_edges, binnumber = binned_statistic_2d(
                release_lats[seasonal_idxs],
                release_lons[seasonal_idxs],
                metric_args["values"],
                bins=[lat_bins, lon_bins],
                statistic=metric_args["statistic"],
                expand_binnumbers=True
                )
        
        binned_fps[binned_fps==0] = np.nan

        cbar = True if season_n == 3 else False

        ax[0, season_n], cbar_hist = plot_binned_map(ax[0, season_n], lon_edges, lat_edges, binned_fps, cut_lats=(40,20), cmap=metric_args["cmap"], metric_name=metric_args["metric_name"], vmin_vmax=vmin_vmax, fig=fig, cbar=cbar, cbar_position="right", title_str=seas, return_cbar=True, domain_lats=dom_lats, domain_lons=dom_lons)

    if return_fig:
        return fig
    
