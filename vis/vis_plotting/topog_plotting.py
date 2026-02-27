import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr

from vis_plotting.general_plotting import country_mask


def plot_topography(
    data,
    cmap='RdYlGn_r',
    n_bins=10,
    vmin=None,
    vmax=None,
    plot_country_boundaries=True,
    title=None,
    country=None,
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
    title : str
        Custom title
    country : str
        If provided, applies a country mask to include only points within this country.
        The country name must match one of the entries in `data.countries.country_mask['name']`.
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

    if country is not None:
        da = country_mask(data, country, da)

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
    if title is None:
        title_base = long_name or "Topography"
    else:
        title_base = title

    if country is not None:
        ax.set_title(f"{title_base} - {country}")
    else:
        ax.set_title(title_base)
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
    log_y=False,
    country=None,
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
    country : str
        If provided, applies a country mask to include only points within this country.
        The country name must match one of the entries in `data.countries.country_mask['name']`.

    Returns
    -------
    fig, ax : matplotlib Figure, Axes
        Figure and axes of the histogram.
    (counts, bin_edges) : tuple(np.ndarray, np.ndarray)
        The histogram values returned by matplotlib (counts or density) and the bin edges.
    """

    ds = data.topog_file

    # Apply optional country mask
    if country is not None:
        ds = country_mask(data, country, ds)

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
    title_base = title if title else f"Histogram of {long_name}"
    if country is not None:
        ax.set_title(f"{title_base} – {country}")
    else:
        ax.set_title(title_base)

    # Grid and layout
    ax.grid(True, which='both', linestyle=':', linewidth=0.6, alpha=0.7)
    plt.tight_layout()
    plt.show()

    return fig, ax, (counts, bin_edges)

def plot_landuse_frequency(
    data,
    country=None,
    outfile=None,
    log_y=False,
    title = "Land use frequency (sum of fractions per class)",
    area_weighted=False,
    as_percent=True,
):
    """
    Plot a bar chart showing the total fractional area for each land use class.

    Parameters
    ----------
    data : object
        Container with `data.landcover_file['landcover_fraction']`.
        Expected dims: (lat, lon, pseudo_level) or (pseudo_level, lat, lon).
        Ocean/invalid grid cells are typically NaN and are ignored by the sum.
    country : str
        If provided, applies a country mask to include only points within this country.
        The country name must match one of the entries in `data.countries.country_mask['name']`.
    outfile : str or None
        If provided, saves the figure to this filepath.
    log_y : bool
        If True, use a logarithmic y-axis.
        Default is False (linear scale).
    title : str
        Plot title.
    area_weighted : bool
        If True, weight each grid cell by cos(latitude) before summation (equal-area approx).
        Default is False (simple sum of fractions).
    as_percent : bool
        If True, convert totals to percentage of the total fractional area over all classes.
        Default is True (plot share of each class rather than raw totals).

    Returns
    -------
    fig, ax : Matplotlib Figure and Axes
    freq : dict
        Mapping {class_id: total} for the plotted classes.
    """

    da = data.landcover_file['landcover_fraction']

    if country is not None:
        da = country_mask(data, country, da)


    # Normalise dims to (lat, lon, pseudo_level)
    if da.dims == ('pseudo_level', 'lat', 'lon'):
        da = da.transpose('lat', 'lon', 'pseudo_level')
    elif da.dims != ('lat', 'lon', 'pseudo_level'):
        raise ValueError(f"Unexpected dims {da.dims}. Expected (lat, lon, pseudo_level).")

    class_ids = [int(c) for c in da['pseudo_level'].values.tolist()]  # typically 1..9

    # Optional area weights - takes into account the convergence of meridians
    # at high/low latitudes for a more accurate global summary (not just a simple sum of fractions)
    weights = None
    if area_weighted:
        lat = da['lat'].values
        cosw = np.cos(np.deg2rad(lat))
        weights2d = xr.DataArray(
            np.repeat(cosw[:, None], da.sizes['lon'], axis=1),
            coords={'lat': da['lat'], 'lon': da['lon']},
            dims=('lat', 'lon')
        )
        weights = weights2d.broadcast_like(da)

    # Sum over lat/lon, ignoring NaN (ocean)
    if weights is None:
        summed = da.sum(dim=('lat', 'lon'), skipna=True)
    else:
        summed = (da * weights).sum(dim=('lat', 'lon'), skipna=True)

    totals = np.array(summed.values, dtype=float)

    freq_all = {cid: totals[i] for i, cid in enumerate(class_ids)}
    freq_all = {cid: v for cid, v in freq_all.items() if np.isfinite(v)}

    # Convert to % if requested
    if as_percent:
        total_sum = np.sum(list(freq_all.values()))
        if total_sum > 0:
            freq_all = {cid: (v / total_sum) * 100.0 for cid, v in freq_all.items()}

    # Sorted plotting order
    plot_ids = sorted(freq_all.keys())
    labels = [CLASS_LABELS[cid] for cid in plot_ids]
    values = [freq_all[cid] for cid in plot_ids]
    bar_colors = [CLASS_COLOURS.get(cid, "steelblue") for cid in plot_ids]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(plot_ids))
    ax.bar(x, values, color=bar_colors, edgecolor='black')

    if as_percent:
        y_label = "Share of fractional landcover (%)"
    else:
        y_label = "Total fractional landcover"

    if area_weighted:
        y_label += " (area-weighted)"
    if log_y:
        ax.set_yscale('log')
        y_label += " (log scale)"

    ax.set_ylabel(y_label)

    if country is not None:
        ax.set_title(f"{title} – {country}")
    else:
        ax.set_title(title)
        
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.grid(True, axis='y', linestyle=':', linewidth=0.6, alpha=0.7)

    plt.tight_layout()

    if outfile:
        plt.savefig(outfile, bbox_inches='tight')
        print(f"Saved land use bar chart to: {outfile}")

    plt.show()

    return fig, ax, freq_all


def plot_majority_landuse_map(
    data,
    country=None,
    figure_size=(10.5, 5),
    plot_country_boundaries=True,
    coastline=True,
    title="Majority land use class",
    show_legend=True,
    legend_loc="lower left",
):
    """
    Plot a spatial map of the majority land‑use class for each grid cell,
    using `data.landcover_file['landcover_fraction']`.

    Rules
    -----
    - Majority class = argmax over `pseudo_level` of landcover_fraction.
    - Cells where all classes are NaN are treated as `sea` (class 0).

    Parameters
    ----------
    data : object
        Container with `data.landcover_file['landcover_fraction']`.
        Expected dims: (lat, lon, pseudo_level) or (pseudo_level, lat, lon).
    country : str
        If provided, applies a country mask to include only points within this country.
        The country name must match one of the entries in `data.countries.country_mask['name']`.
    figure_size : tuple
        Figure size in inches.
    plot_country_boundaries : bool
        If True, overlay country boundaries.
        Default to True.
    coastline : bool
        If True, overlay a coastline.
    title : str
        Plot title.
    show_legend : bool
        If True, show a categorical legend for classes present in the domain.
    legend_loc : str
        Matplotlib legend location string.

    Returns
    -------
    fig, ax, mesh, majority_da, (cmap, norm)
        - fig, ax: Matplotlib Figure and Axes
        - mesh: QuadMesh artist from pcolormesh
        - majority_da: xarray.DataArray of integer class IDs (0..9)
        - (cmap, norm): the discrete colormap and normaliser used
    """

    # --- Load and normalise dims ---
    da = data.landcover_file['landcover_fraction']

    if country is not None:
        da = country_mask(data, country, da)

    if da.dims == ('pseudo_level', 'lat', 'lon'):
        da = da.transpose('lat', 'lon', 'pseudo_level')
    elif da.dims != ('lat', 'lon', 'pseudo_level'):
        raise ValueError(f"Unexpected dims {da.dims}. Expected (lat, lon, pseudo_level).")

    # Grid coords
    lon = da['lon'].values
    lat = da['lat'].values

    # Identify cells with any finite class value (land or inland water, etc.)
    valid = np.isfinite(da).any(dim='pseudo_level')  # (lat, lon) boolean

    # Compute majority class on valid cells:
    # Fill NaNs with a very negative number so they never win argmax,
    # but *only* use this for argmax (not for plotting actual fractions).
    filled = da.fillna(-1e15)
    # argmax gives positional indices 0..(n_classes-1)
    idx = filled.argmax(dim='pseudo_level')  # (lat, lon), dtype=int
    # Map positional indices to actual pseudo_level IDs (typically 1..9)
    pseudo_ids = da['pseudo_level']
    majority_da = pseudo_ids.isel(pseudo_level=idx)  # (lat, lon), values in 1..9

    # Assign sea=0 where no class is present (all-NaN originally)
    majority_da = xr.where(valid, majority_da, 0).astype(int)

    # --- Build discrete colormap & norm for classes 0-9 ---
    ordered_ids = list(range(0, 10))
    colours_list = [CLASS_COLOURS[i] for i in ordered_ids]
    cmap = ListedColormap(colours_list, name="landuse_maj_0to9")
    # Boundaries at half-integers to capture exact integer values
    boundaries = np.arange(-0.5, 10.5, 1.0)
    norm = BoundaryNorm(boundaries, ncolors=cmap.N, clip=False)

    # --- Plot ---
    fig = plt.figure(figsize=figure_size)
    proj = ccrs.PlateCarree()
    ax = plt.axes(projection=proj)

    if coastline:
        ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    if plot_country_boundaries:
        ax.add_feature(cfeature.BORDERS, linewidth=0.7, edgecolor="black", alpha=0.7)

    gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False,
                      linewidth=0.3, color='0.5', alpha=0.6)

    # Remove top and right labels (keep left and bottom)
    try:
        gl.top_labels = False
        gl.right_labels = False
    except Exception:
        pass

    mesh = ax.pcolormesh(
        lon, lat, majority_da.values,
        transform=proj,
        cmap=cmap,
        norm=norm,
        shading='nearest'  # categorical look; avoids smoothing between classes
    )

    if country is not None:
        ax.set_title(f"{title} – {country}")
    else:
        ax.set_title(title)

    # --- Legend to the RIGHT of the map ---
    if show_legend:
        present_ids = np.unique(majority_da.values[np.isfinite(majority_da.values)])
        present_ids = [int(i) for i in present_ids if 0 <= i <= 9]

        handles = [
            Patch(facecolor=CLASS_COLOURS[i], edgecolor='black', label=CLASS_LABELS[i])
            for i in present_ids
        ]

        leg = ax.legend(
            handles=handles,
            loc="center left",            # legend’s LEFT side is the anchor
            bbox_to_anchor=(1.02, 0.5),   # x>1 moves outside the axes
            borderaxespad=0.0,
            frameon=True,
            framealpha=0.9,
            fontsize=9,
            title="Majority land use"
        )

        try:
            leg._legend_box.align = "left"
        except Exception:
            pass

    # ensure space on right so legend isn’t clipped
    plt.subplots_adjust(right=0.80)

    plt.tight_layout()
    plt.show()

    return fig, ax, mesh, majority_da, (cmap, norm)


# --- Shared labels and colours for land use classes ---
CLASS_LABELS = {
    0: "sea",
    1: "broadleaf trees",
    2: "needleleaf trees",
    3: "C3 (temperate) grass",
    4: "C4 (tropical) grass",
    5: "shrubs",
    6: "urban",
    7: "inland water",
    8: "bare soil",
    9: "ice",
}

CLASS_COLOURS = {
    0:  "#1d9cfe",  # sea (light blue)
    1:  "#2ca02c",  # broadleaf trees (green)
    2:  "#1f7a1f",  # needleleaf trees (dark green)
    3:  "#8fd175",  # C3 grass (light green)
    4:  "#ffd166",  # C4 grass (yellow-ish)
    5:  "#b38f5a",  # shrubs (brown)
    6:  "#7f7f7f",  # urban (grey)
    7:  "#2ca8c2",  # inland water (cyan)
    8:  "#d2b48c",  # bare soil (tan)
    9:  "#a0c4ff",  # ice (pale blue)
}