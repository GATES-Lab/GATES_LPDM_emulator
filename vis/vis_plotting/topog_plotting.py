import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

import cartopy.crs as ccrs
import cartopy.feature as cfeature


def plot_topography(
    data,
    cmap='RdYlGn_r',
    n_bins=10,
    vmin=None,
    vmax=None,
    plot_country_boundaries=True,
    ocean_colour="#1d9cfe",
    land_only=True,
    figure_size=(10.5, 5),
):
    """
    Plot topography (surface altitude), with:
      - oceans in a fixed colour (not part of the colour scale), and
      - a discrete colour scale with `n_bins` bins.

    Parameters
    ----------
    data : LoadBaseSatelliteData
        Container with topography dataset in `data.topog_file`, variable 'surface_altitude' (lat, lon).
        If available, a land mask is read from `data.landcover_file['land_binary_mask']` and used in "land_only" masking.
    cmap : str, optional
        Base colormap name to discretize. Default 'RdYlGn_r', other options include 'terrain'.
    n_bins : int, optional
        Number of discrete bins. Default 10.
    vmin, vmax : float or None
        Optional min/max (in metres) for bin edges. If None, inferred from land-only finite values.
    plot_country_boundaries : bool, optional
        Overlay country boundaries. Default True.
    ocean_colour : str, optional
        Colour for oceans (not included in colourbar). Default blue.
    land_only : bool, optional
        If True, only include land points (mask out ocean points).
        Default is True.
    figure_size : tuple, optional
        Figure size in inches.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    mesh : QuadMesh
        The pcolormesh artist.
    (bin_edges) : np.ndarray
        The bin edges used for the discrete colour mapping.
    """

    ds = data.topog_file
    da = ds['surface_altitude']  # (lat, lon)

    # Keep only finite numeric values
    da = da.where(np.isfinite(da))
    Z = da.values  # 2D array


    # Apply optional mask (keep True, drop False)
    if land_only:
        # Use land mask (True=land, False=ocean)
        land_mask = (data.landcover_file['land_binary_mask'] > 0.9)
        da = da.where(land_mask)

    # Compute bin edges
    Z_land = da.values
    Z_land_flat = Z_land[np.isfinite(Z_land)]
    if Z_land_flat.size == 0:
        raise ValueError("No valid land altitude values found after masking/cleaning.")

    # Set bin edges
    lo = np.min(Z_land_flat) if vmin is None else float(vmin)
    hi = np.max(Z_land_flat) if vmax is None else float(vmax)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        raise ValueError("Invalid vmin/vmax or data range for discrete binning.")

    bin_edges = np.linspace(lo, hi, n_bins + 1)

    # Build a discrete colormap from the base cmap
    base_cmap = plt.get_cmap(cmap)
    # Sample at bin centers to get n distinct colours
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    centers01 = (centers - lo) / (hi - lo + 1e-15)
    colours = base_cmap(centers01)
    discrete_cmap = ListedColormap(colours, name=f"{cmap}_{n_bins}bins")

    # Make sure masked (ocean) cells use a fixed colour that is NOT part of the colour scale
    discrete_cmap.set_bad(ocean_colour)

    # Discrete normalization
    norm = BoundaryNorm(bin_edges, ncolors=discrete_cmap.N, clip=True)

    lon = ds['lon'].values
    lat = ds['lat'].values

    # --- Plot ---
    fig = plt.figure(figsize=figure_size)
    proj = ccrs.PlateCarree()
    ax = plt.axes(projection=proj)

    # Background features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    if plot_country_boundaries:
        ax.add_feature(cfeature.BORDERS, linewidth=0.7, edgecolor="black", alpha=0.7)

    gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False,
                      linewidth=0.3, color='0.5', alpha=0.6)

    # pcolormesh with discrete colors and masked oceans
    mesh = ax.pcolormesh(
        lon, lat, da.values,
        transform=proj,
        cmap=discrete_cmap,
        norm=norm,
        shading='auto'
    )

    # Colourbar: show only land scale (oceans are masked and not shown in the bar)
    cb = plt.colorbar(
        mesh, ax=ax, orientation='horizontal', pad=0.06, shrink=0.85,
        boundaries=bin_edges,
        ticks=centers
    )

    # Labels for the colourbar
    units = da.attrs.get('units', 'm')
    long_name = da.attrs.get('long_name', 'Surface altitude')
    cb.set_label(f"{long_name} [{units}]")

    # Optionally format tick labels (e.g., integers for metres)
    try:
        cb.ax.set_xticklabels([f"{v:.0f}" for v in centers])
    except Exception:
        pass

    # Title & layout
    ax.set_title(long_name if long_name else 'Topography')
    plt.tight_layout()
    plt.show()

    return fig, ax, mesh, bin_edges



def plot_topography_histogram(
    data,
    bins=50,
    range=None,
    density=False,
    cumulative=False,
    land_only=True,
    title=None,
    figsize=(8, 5),
    colour="#4C78A8",
    alpha=0.8,
    edgecolour='white',
    log_y=False
):
    """
    Create a histogram of topography (surface altitude) values from `data.topog_file`.

    Parameters
    ----------
    data : LoadBaseSatelliteData
        Container with the topography dataset in `data.topog_file`.
        Must contain variable 'surface_altitude' with dims (lat, lon).
    bins : int or sequence, optional
        Number of bins or explicit bin edges. Default is 50.
    range : tuple(float, float) or None
        Lower/upper range of the bins, e.g., (0, 5000) metres. If None, inferred from data. 
    density : bool, optional
        If True, plot probability density (area under histogram = 1).
        If False, plot raw counts. Default False.
    cumulative : bool, optional
        If True, plot cumulative histogram. Default False.
    land_only : bool, optional
        If True, only include land points (mask out ocean points).
        Default is True.
    title : str or None
        Custom title. If None, uses the variable long_name.
    figsize : tuple
        Figure size in inches. Default (8, 5).
    colour : str
        Bar colour. Default '#4C78A8'.
    alpha : float
        Bar transparency. Default 0.8.
    edgecolour : str
        Bar edge colour. Default 'white'.
    log_y : bool
        If True, use a logarithmic y-axis. Useful when the distribution spans orders of magnitude.

    Returns
    -------
    fig, ax : matplotlib Figure, Axes
        Figure and axes of the histogram.
    (counts, bin_edges) : tuple(np.ndarray, np.ndarray)
        The histogram values returned by matplotlib (counts or density) and the bin edges.
    """

    ds = data.topog_file
    da = ds['surface_altitude']  # (lat, lon)

    # Ensure finite values only
    da = da.where(np.isfinite(da))

    # Apply optional mask (keep True, drop False)
    if land_only:
        land_mask = (data.landcover_file['land_binary_mask'] > 0.9)
        da = da.where(land_mask)
    Z = da.values.ravel()
    Z = Z[np.isfinite(Z)]  # remove NaN after ravel

    if Z.size == 0:
        raise ValueError("No valid altitude values found after masking/cleaning.")

    # Units and name (fallbacks)
    units = da.attrs.get('units', 'm')
    long_name = da.attrs.get('long_name', 'Surface altitude')

    # Build histogram
    fig, ax = plt.subplots(figsize=figsize)

    counts, bin_edges, patches = ax.hist(
        Z,
        bins=bins,
        range=range,
        density=density,
        cumulative=cumulative,
        color=colour,
        alpha=alpha,
        edgecolor=edgecolour
    )

    # Labels & title
    if density:
        y_label = 'Density'
    else:
        y_label = 'Cumulative count' if cumulative else 'Count'

    if log_y:
        ax.set_yscale('log')
        # Ensure a positive lower bound (hist may have zeros)
        positive = counts[counts > 0]
        if positive.size > 0:
            ymin = max(positive.min() * 0.5, 1e-2 if density else 1.0)
            # keep current top if already set by Matplotlib; otherwise set a sensible top
            ytop = ax.get_ylim()[1]
            ax.set_ylim(bottom=ymin, top=ytop)
        # Clarify label
        y_label += ' (log scale)'

    ax.set_xlabel(f"{long_name} [{units}]")
    ax.set_ylabel(y_label)
    ax.set_title(title if title else f"Histogram of {long_name}")

    # Grid and layout
    ax.grid(True, which='both', linestyle=':', linewidth=0.6, alpha=0.7)
    plt.tight_layout()
    plt.show()

    return fig, ax, (counts, bin_edges)