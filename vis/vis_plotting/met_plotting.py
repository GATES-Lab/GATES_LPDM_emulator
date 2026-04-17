import numpy as np
import matplotlib.pyplot as plt
from windrose import WindroseAxes


def calculate_wind_speed_stats_from_ds(ds):
    """
    Calculate summary statistics of wind speed from an xarray Dataset.

    Wind speed is computed as:
        speed = sqrt(x_wind^2 + y_wind^2)

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing variables 'x_wind' and 'y_wind'.
        Can contain any shape/dimensions; values are flattened internally.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'count' : int
            Number of finite wind-speed values.
        - 'mean', 'std', 'min', 'p25', 'median', 'p75', 'max' : float
            Summary statistics of finite wind-speed values.

    Notes
    -----
    If no finite values are available, returns count=0 and NaN for all
    other statistics.
    """
    u = np.asarray(ds["x_wind"].values)
    v = np.asarray(ds["y_wind"].values)
    speed = np.sqrt(u**2 + v**2).ravel()
    speed = speed[np.isfinite(speed)]

    if speed.size == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "p25": np.nan,
            "median": np.nan,
            "p75": np.nan,
            "max": np.nan,
        }

    return {
        "count": int(speed.size),
        "mean": float(np.mean(speed)),
        "std": float(np.std(speed)),
        "min": float(np.min(speed)),
        "p25": float(np.percentile(speed, 25)),
        "median": float(np.median(speed)),
        "p75": float(np.percentile(speed, 75)),
        "max": float(np.max(speed)),
    }


def compute_wind_speed_metrics(met_ds, level=1):
    """
    Compute global and seasonal wind-speed statistics from meteorology data.

    Parameters
    ----------
    met_ds : xarray.Dataset
        Meteorology dataset containing 'x_wind' and 'y_wind'.
        Typically this is `data.met_file` from LoadBaseSatelliteData.
    level : int, optional
        Atmospheric level to select if the dataset contains a `levels` dimension.
        Default is 1.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'level_used' : int or str
            Selected level when `levels` exists, otherwise a descriptive string.
        - 'global' : dict
            Domain-wide summary statistics as returned by
            `calculate_wind_speed_stats_from_ds`.
        - 'seasonal' : dict[str, dict]
            Seasonal statistics for JFM, AMJ, JAS, OND when `time` is available.

    Raises
    ------
    ValueError
        If either 'x_wind' or 'y_wind' is missing from `met_ds`.
    """
    if "x_wind" not in met_ds or "y_wind" not in met_ds:
        raise ValueError("x_wind and y_wind are required to compute wind-speed metrics")

    ds = met_ds
    if "levels" in ds.dims:
        ds = ds.sel(levels=level)

    global_stats = calculate_wind_speed_stats_from_ds(ds)

    season_months = {
        "JFM": [1, 2, 3],
        "AMJ": [4, 5, 6],
        "JAS": [7, 8, 9],
        "OND": [10, 11, 12],
    }
    seasonal_stats = {}

    if "time" in ds.coords:
        for season, months in season_months.items():
            season_ds = ds.where(ds.time.dt.month.isin(months), drop=True)
            seasonal_stats[season] = calculate_wind_speed_stats_from_ds(season_ds)

    return {
        "level_used": level if "levels" in met_ds.dims else "N/A (single-level met)",
        "global": global_stats,
        "seasonal": seasonal_stats,
    }


def write_wind_speed_metrics_txt(metrics, out_path, region, domain, date):
    """
    Write wind-speed metrics to a plain text file.

    Parameters
    ----------
    metrics : dict
        Dictionary returned by `compute_wind_speed_metrics`.
    out_path : str or pathlib.Path
        Output text-file path.
    region : str
        Region identifier to include in the file header.
    domain : str
        Domain identifier to include in the file header.
    date : str or int
        Input date string/value used for data loading, included in the header.

    Returns
    -------
    None
        Writes formatted metrics to `out_path`.

    Notes
    -----
    The output contains one global/domain section and, when available,
    one section per season (JFM, AMJ, JAS, OND).
    """
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Wind speed metrics\n")
        f.write("==================\n")
        f.write(f"Region: {region}\n")
        f.write(f"Domain: {domain}\n")
        f.write(f"Date input: {date}\n")
        f.write(f"Level used: {metrics['level_used']}\n\n")

        f.write("Global/domain stats (m s^-1)\n")
        f.write("---------------------------\n")
        for key in ("count", "mean", "std", "min", "p25", "median", "p75", "max"):
            val = metrics["global"][key]
            if key == "count":
                f.write(f"{key}: {int(val)}\n")
            else:
                f.write(f"{key}: {val:.4f}\n")

        if metrics["seasonal"]:
            f.write("\nSeasonal stats (m s^-1)\n")
            f.write("-----------------------\n")
            for season in ("JFM", "AMJ", "JAS", "OND"):
                stats = metrics["seasonal"].get(season)
                if stats is None:
                    continue
                f.write(f"\n{season}\n")
                for key in ("count", "mean", "std", "min", "p25", "median", "p75", "max"):
                    val = stats[key]
                    if key == "count":
                        f.write(f"  {key}: {int(val)}\n")
                    else:
                        f.write(f"  {key}: {val:.4f}\n")


def plot_single_windrose(met_data, bins=[0, 2, 4, 6, 8, 10], limit=20, tick_interval=5, ax=None, legend=True, title=None):
    """
    Plot a single windrose from x_wind and y_wind features in an xarray Dataset.

    Parameters
    ----------
    data : xarray.Dataset
        Dataset containing 'x_wind' and 'y_wind'. Must be pre-filtered to 
        a single atmospheric level.
    bins : list of float, optional
        Wind speed bin edges. Default [0, 2, 4, 6, 8, 10].
    limit : float, optional
        Maximum radial limit (percentage frequency). Default 20.
    ax : matplotlib.axes.Axes, optional
        Existing axis to plot on. If None, a new figure is created.
    legend : bool, optional
        Whether to display the windspeed legend. Default True.
    title : str, optional
        Title for the individual windrose.

    Returns
    -------
    wax : WindroseAxes
        The generated windrose axis object.
    """
    # 1. Validation: Ensure only one level exists
    if 'levels' in met_data.dims and met_data.levels.size > 1:
        raise ValueError(f"Data contains {met_data.levels.size} levels. Please select a single level before plotting.")

    # 2. Setup the Windrose Axes
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        wax = WindroseAxes.from_ax(fig=fig)
    else:
        # Get the position of the provided axis to overlay the WindroseAxes
        fig = ax.get_figure()
        rect = ax.get_position()
        wax = WindroseAxes(fig, rect)
        fig.add_axes(wax)
        ax.set_visible(False) # Hide the original placeholder axis

    # 3. Flatten and Calculate
    u = met_data['x_wind'].values.ravel()
    v = met_data['y_wind'].values.ravel()
    
    mask = ~np.isnan(u) & ~np.isnan(v)
    speed = np.sqrt(u[mask]**2 + v[mask]**2)
    # Note wind direction has been corrected to follow meteorological convention (0° = from North, increasing clockwise, and plotting incoming wind direction)
    direction = (270 - np.rad2deg(np.arctan2(v[mask], u[mask]))) % 360


    # 4. Plotting
    wax.contourf(direction, speed, normed=True, bins=bins, cmap=plt.cm.RdBu, nsector=16)
    wax.set_rmax(limit)
    ticks = np.arange(0, limit + tick_interval, tick_interval)
    wax.set_rgrids(ticks[1:], labels=[f'{t:.0f}%' for t in ticks[1:]])

    if legend:
        wax.set_legend(title="Windspeed (m s$^{-1}$)", loc='lower right', bbox_to_anchor=(1.2, 0))
    
    if title:
        wax.set_title(title, fontweight='bold', fontsize=10, y=1.1)

    return wax
