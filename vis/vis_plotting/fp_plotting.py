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


def compute_footprint_availability_metrics(ds):
    """Compute basic data-availability metrics for loaded footprint observations.

    Parameters
    ----------
    ds : xarray.Dataset
        Footprint dataset containing a ``time`` coordinate and per-observation
        release locations.

    Returns
    -------
    dict
        Dictionary containing total count, seasonal counts, monthly counts,
        and date-range information.
    """
    if "time" not in ds:
        raise ValueError("Dataset must contain a 'time' coordinate for availability metrics.")

    times = pd.DatetimeIndex(ds.time.values)
    total_footprints = int(len(times))

    metrics = {
        "total_footprints": total_footprints,
        "date_start": str(times.min()) if total_footprints else None,
        "date_end": str(times.max()) if total_footprints else None,
        "n_unique_days": int(len(pd.unique(times.normalize()))) if total_footprints else 0,
        "n_unique_months": int(len(pd.unique(times.to_period("M")))) if total_footprints else 0,
        "seasonal_counts": {"JFM": 0, "AMJ": 0, "JAS": 0, "OND": 0},
        "monthly_counts": {m: 0 for m in range(1, 13)},
    }

    if total_footprints == 0:
        return metrics

    months = pd.Series(times.month)
    for m in range(1, 13):
        metrics["monthly_counts"][m] = int((months == m).sum())

    metrics["seasonal_counts"]["JFM"] = int(months.isin([1, 2, 3]).sum())
    metrics["seasonal_counts"]["AMJ"] = int(months.isin([4, 5, 6]).sum())
    metrics["seasonal_counts"]["JAS"] = int(months.isin([7, 8, 9]).sum())
    metrics["seasonal_counts"]["OND"] = int(months.isin([10, 11, 12]).sum())

    return metrics


def write_footprint_availability_txt(metrics, out_path, region, domain, date):
    """Write footprint availability metrics to a plain-text file.

    Parameters
    ----------
    metrics : dict
        Output dictionary from ``compute_footprint_availability_metrics``.
    out_path : str or pathlib.Path
        File path for the output text file.
    region : str
        Region name used for data loading.
    domain : str
        Domain name used for data loading.
    date : str
        Date selector used for data loading.
    """
    lines = [
        "Footprint Availability Metrics",
        "=" * 31,
        f"Region: {region}",
        f"Domain: {domain}",
        f"Date selector: {date}",
        "",
        f"Total footprints loaded: {metrics['total_footprints']}",
        f"Unique days represented: {metrics['n_unique_days']}",
        f"Unique months represented: {metrics['n_unique_months']}",
        f"Date range start: {metrics['date_start']}",
        f"Date range end: {metrics['date_end']}",
        "",
        "Seasonal counts:",
    ]

    for season in ("JFM", "AMJ", "JAS", "OND"):
        lines.append(f"  - {season}: {metrics['seasonal_counts'][season]}")

    lines.extend(["", "Monthly counts (1-12):"])
    for month in range(1, 13):
        lines.append(f"  - {month:02d}: {metrics['monthly_counts'][month]}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def plot_footprint_availability(ds, return_fig=True):
    """Visualize footprint availability as monthly and seasonal bar charts.

    Parameters
    ----------
    ds : xarray.Dataset
        Footprint dataset containing a ``time`` coordinate.
    return_fig : bool, optional
        If ``True``, return the created figure object.

    Returns
    -------
    matplotlib.figure.Figure, optional
        Figure object containing monthly and seasonal availability plots.
    """
    metrics = compute_footprint_availability_metrics(ds)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=220, constrained_layout=True)

    month_labels = [f"{m:02d}" for m in range(1, 13)]
    month_values = [metrics["monthly_counts"][m] for m in range(1, 13)]
    axes[0].bar(month_labels, month_values, color="tab:blue", alpha=0.85)
    axes[0].set_title("Monthly footprint count")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("N footprints")
    axes[0].tick_params(axis="x", rotation=45)

    season_labels = ["JFM", "AMJ", "JAS", "OND"]
    season_values = [metrics["seasonal_counts"][s] for s in season_labels]
    axes[1].bar(season_labels, season_values, color="tab:orange", alpha=0.85)
    axes[1].set_title("Seasonal footprint count")
    axes[1].set_xlabel("Season")
    axes[1].set_ylabel("N footprints")

    fig.suptitle(
        f"Footprint availability (total={metrics['total_footprints']}, unique days={metrics['n_unique_days']})"
    )

    if return_fig:
        return fig
    
