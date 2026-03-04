import numpy as np
import matplotlib.pyplot as plt
from windrose import WindroseAxes

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
