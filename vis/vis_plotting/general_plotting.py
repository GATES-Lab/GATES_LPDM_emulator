import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import numpy as np
import cartopy.feature as cfeature
import cartopy.crs as ccrs

from pathlib import Path
from itertools import cycle

def _get_lat_lon_bounds(data, type="met", wrap_center=0.0):
    """
    Return a dictionary with the min/max lat/lon bounds from the specified data series (met, fp, or topo).

    Parameters
    ----------
    data : LoadBaseSatelliteData
        Container with met/fp/topo datasets
    type : str
        Which dataset to extract bounds from: "met", "fp", or "topo"
    wrap_center : float
        Center longitude for wrapping (default 0.0). Only relevant if the data crosses the anti-meridian (near NZ/Alaska), in which case the bounds will be split into two.
        
    Returns
    -------
    dict with keys: min_lat, max_lat, min_lon, max_lon

    """
    if type == "met":
        src = data.met_file
    elif type == "fp":
        src = data.fp_data_full
    elif type == "topo":
        src = data.topog_file
    else:
        raise ValueError("type must be one of: 'met', 'fp', 'topo'")

    lat = np.asarray(src["lat"]).ravel()
    lon = np.asarray(src["lon"]).ravel()

    return dict(
        min_lat=float(np.min(lat)),
        max_lat=float(np.max(lat)),
        min_lon=float(np.min(lon)),
        max_lon=float(np.max(lon)),
    )

def _rectangle_segments(min_lon, max_lon, min_lat, max_lat, wrap_center=0.0, eps=1e-9):
    """
    Return 1 or 2 polygons (lon, lat) for a rectangle, splitting if it crosses the anti-meridian (near NZ/Alaska).
    Longitudes assumed to be wrapped around 'wrap_center'.

    Parameters
    ----------
    min_lon, max_lon, min_lat, max_lat : float
        Bounds of the rectangle in degrees, extracted with _get_lat_lon_bounds() function
    wrap_center : float
        Center longitude for wrapping (default 0.0). Only relevant if the data crosses the anti-meridian (near NZ/Alaska), in which case the bounds will be split into two.
    eps : float
        Small tolerance to handle numerical precision issues when checking for anti-meridian crossing (default 1e-9 degrees, which is about 0.1 mm at the equator)    

    Returns
    -------
    list of polygons (list of (lon, lat) tuples)
    """
    # To cover case where min_lon > max_lon due to anti-meridian crossing (ie near New Zealand / Alaska), we split into two segments
    crosses = (max_lon + eps) < (min_lon - eps)
    if crosses:
        segA = [
            (min_lon, min_lat), (wrap_center+180, min_lat),
            (wrap_center+180, max_lat), (min_lon, max_lat)
        ]
        segB = [
            (wrap_center-180, min_lat), (max_lon, min_lat),
            (max_lon, max_lat), (wrap_center-180, max_lat)
        ]
        return [segA, segB]
    else:
        return [[(min_lon, min_lat), (max_lon, min_lat), (max_lon, max_lat), (min_lon, max_lat)]]

def _add_lonlat_rectangle(ax, min_lon, max_lon, min_lat, max_lat,
                         edgecolor='tab:blue', facecolor='none', linewidth=2.2,
                         linestyle='-', zorder=5, wrap_center=0.0,
                         crs=ccrs.PlateCarree(), label=None, clip_on=False):
    """
    Creates a rectangle to cover the outline of the data series domain

    Parameters
    ----------
    ax : matplotlib axis
        The axis to which the rectangle patch will be added
    min_lon, max_lon, min_lat, max_lat : float
        Bounds of the rectangle in degrees, extracted with _get_lat_lon_bounds() function
    edgecolor : str
        Color of the rectangle edge (default 'tab:blue')
    facecolor : str
        Color of the rectangle face (default 'none' for transparent)
    linewidth : float
        Width of the rectangle edge (default 2.2)
    linestyle : str
        Style of the rectangle edge (default '-')
    zorder : int
        Z-order for layering the rectangle (default 5, which is above coastlines and borders)
    wrap_center : float
        Center longitude for wrapping (default 0.0). Only relevant if the data crosses the anti-meridian (near NZ/Alaska), in which case the bounds will be split into two
    crs : cartopy.crs
        Coordinate reference system for the rectangle (default PlateCarree, which is standard lon/lat)
    label : str
        Optional label for the rectangle to be used in the legend (default None)
    clip_on : bool
        Whether to allow the rectangle to be clipped at the axis boundaries (default False, which is usually desirable for global maps to prevent clipping at the projection boundary)
    
    Returns
    -------
    None (adds a Polygon patch to the provided axis)

    """
    polys = _rectangle_segments(min_lon, max_lon, min_lat, max_lat, wrap_center=wrap_center)
    first = True
    for poly in polys:
        patch = Polygon(np.array(poly), closed=True,
                        edgecolor=edgecolor, facecolor=facecolor, linewidth=linewidth,
                        linestyle=linestyle, zorder=zorder, transform=crs,
                        label=label if first and label else None,
                        clip_on=clip_on)
        ax.add_patch(patch)
        first = False

def plot_domain(data,
                type,
                projection="PlateCarree",
                wrap_center=0.0,
                zoom_to_data=True,
                title=None):
    """
    Plot the bounding rectangle from the provided `data` container (met/fp/topo).
    
    Parameters
    ----------
    data : LoadBaseSatelliteData
        Container with met/fp/topo datasets
    type : str
        Which dataset to extract bounds from: "met", "fp", or "topo"
    projection : str
        Which cartopy projection to use for the plot (default "PlateCarree"). Options include "Robinson", "Mollweide", "Mercator", "LambertConformal", etc.
    wrap_center : float
        Center longitude for wrapping (default 0.0). Only relevant if the data crosses the anti-meridian (near NZ/Alaska), in which case the bounds will be split into two
    zoom_to_data : bool
        Whether to zoom the map to the data bounds (default True). Only applies if the bounds do not cross the anti-meridian, as zooming is not possible in that case.
    title : str
        Optional title for the plot (default None, which will use "{type.upper()} Domain")
        
    Returns
    -------
    fig, ax : matplotlib figure and axis objects

    """
    # 1) Get bounds with robust helper
    b = _get_lat_lon_bounds(data, type=type, wrap_center=wrap_center)

    # 2) Choose projection
    proj_map = {
        "PlateCarree": ccrs.PlateCarree(),
        "Robinson": ccrs.Robinson(),
        "Mollweide": ccrs.Mollweide(),
        "Mercator": ccrs.Mercator(),
        "LambertConformal": ccrs.LambertConformal(),
    }.get(projection, ccrs.PlateCarree())

    # 3) Base map
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=proj_map)
    ax.set_global()
    # IMPORTANT: prevent polygon clipping at projection boundary
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

    ax.coastlines(linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor='k')
    ax.gridlines(draw_labels=False, linewidth=0.4, color='gray', alpha=0.5, linestyle='--')

    # 4) Draw rectangle
    _add_lonlat_rectangle(ax, b["min_lon"], b["max_lon"], b["min_lat"], b["max_lat"],
                         edgecolor='tab:blue', linewidth=2.2, linestyle='-',
                         label=f'{type} domain', wrap_center=wrap_center)

    # 5) Legend
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='lower left', frameon=True)

    # 6) Optional zoom (only if no anti-meridian crossing)
    if zoom_to_data and b["max_lon"] >= b["min_lon"]:
        ax.set_extent([b["min_lon"], b["max_lon"], b["min_lat"], b["max_lat"]],
                      crs=ccrs.PlateCarree())

    plt.title(title or f"{type.upper()} Domain", pad=10)
    plt.tight_layout()
    return fig, ax



def plot_multiple_data_series(
    datasets,
    region_names=None,
    show=("met",),
    colours=None,
    linewidth=2.2,
    facealpha=0.0,
    projection="PlateCarree",
    wrap_center=0.0,
    zoom_to_union=True,
    title="UM Domains",
    save=False,
    outdir="multi_domain_characterisation",
    filename="domain_bounds.png",
    show_measurement_locs=False,
):
    """
    Plot a single bounding rectangle per region, using different colours per region.

    Parameters
    ----------
    datasets : sequence
        Iterable of LoadBaseSatelliteData objects
    region_names : sequence of str
        Names of each region
    show : tuple of str
        Single-element tuple, e.g. ("met",), ("fp",), or ("topo",)
    colours : sequence of str
        Colours per region (cycled if fewer than regions)
    show_measurement_locs : bool
        If True, overlay an "x" marker at each unique footprint release location
        (release_lat / release_lon from fp_data_full), coloured to match the region.
    """

    if len(show) != 1:
        raise ValueError("This simplified version expects exactly one entry in `show`")

    n = len(datasets)

    if region_names is None:
        region_names = [f"Region {i+1}" for i in range(n)]
    if len(region_names) != n:
        raise ValueError("region_names must match datasets")

    if colours is None:
        colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if len(colours) < n:
        colours = (colours * n)[:n]

    domain_type = show[0]

    # Projection
    proj = {
        "PlateCarree": ccrs.PlateCarree(),
        "Robinson": ccrs.Robinson(),
        "Mollweide": ccrs.Mollweide(),
        "Mercator": ccrs.Mercator(),
        "LambertConformal": ccrs.LambertConformal(),
    }.get(projection, ccrs.PlateCarree())

    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=proj)
    ax.set_global()
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

    ax.coastlines(linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="k")
    ax.gridlines(
        draw_labels=False, linewidth=0.4,
        color="gray", alpha=0.5, linestyle="--"
    )

    bounds_list = []

    # Plot one rectangle per region
    for data, name, colour in zip(datasets, region_names, colours):
        try:
            b = _get_lat_lon_bounds(data, type=domain_type, wrap_center=wrap_center)
            polys = _rectangle_segments(
                b["min_lon"], b["max_lon"],
                b["min_lat"], b["max_lat"],
                wrap_center=wrap_center,
            )

            first = True
            for poly in polys:
                patch = Polygon(
                    np.array(poly),
                    closed=True,
                    edgecolor=colour,
                    facecolor=colour if facealpha > 0 else "none",
                    linewidth=linewidth,
                    alpha=facealpha if facealpha > 0 else 1.0,
                    transform=ccrs.PlateCarree(),
                    clip_on=False,
                    label=name if first else None,
                    zorder=5,
                )
                ax.add_patch(patch)
                first = False

            bounds_list.append(b)

            if show_measurement_locs:
                try:
                    fp = data.fp_data_full
                    lats = np.asarray(fp["release_lat"]).ravel()
                    lons = np.asarray(fp["release_lon"]).ravel()
                    # Deduplicate to avoid overplotting thousands of identical points
                    coords = np.unique(np.stack([lats, lons], axis=1), axis=0)
                    ax.scatter(
                        coords[:, 1], coords[:, 0],
                        marker="x", color=colour, s=40, linewidths=1.5,
                        transform=ccrs.PlateCarree(), zorder=10,
                    )
                except Exception as me:
                    print(f"[plot_multiple_data_series] Could not plot measurement locs for '{name}': {me}")

        except Exception as e:
            print(f"[plot_multiple_data_series] Skipped '{name}': {e}")

    # Legend
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="lower left", title="Regions", frameon=True)

    # Zoom to union if safe
    def crosses_antimeridian(b):
        return b["max_lon"] < b["min_lon"]

    if zoom_to_union and bounds_list:
        if not any(crosses_antimeridian(b) for b in bounds_list):
            ax.set_extent([
                min(b["min_lon"] for b in bounds_list),
                max(b["max_lon"] for b in bounds_list),
                min(b["min_lat"] for b in bounds_list),
                max(b["max_lat"] for b in bounds_list),
            ], crs=ccrs.PlateCarree())

    plt.title(title, pad=10)
    plt.tight_layout()

    # Save if requested
    if save:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / filename, dpi=300, bbox_inches="tight")

    return fig, ax

def country_mask(
        data,
        country_name,
        ds=None
    ):
    """
    Return a boolean mask for the specified country name from the data.countries.country_mask dataset.

    Parameters
    ----------
    data : LoadBaseSatelliteData
        Container with met/fp/topo datasets, including data.countries.country_mask
    country_name : str
        Name of the country to extract the mask for. Must match one of the entries in data.countries.country_mask['name'].
    ds: xarray.Dataset or xarray.DataArray (optional)
        Option to directly provide eg the footprint dataset
    Returns
    -------
    xarray.DataArray
        Filtered dataset with points within the specified country.
    """
    print("Applying country mask for:", country_name)
    data.get_country_masks()
    mask = data.countries.country_mask.sel(name=country_name)
    if ds is not None:
        filtered = ds.where(mask == 1)
    else:
        filtered = data.where(mask == 1)

    return filtered


def loop_over_countries(
        data,
        func,
        countries,
        **common_kwargs
    ):
    """
    Apply a plotting function `func` to every country in `countries`.
    If countries is None, run a single global plot (no country mask).

    Parameters
    ----------
    data : LoadBaseSatelliteData
        Container with met/fp/topo datasets, including data.countries.country_mask
    func : str
        Plotting function to call, e.g. plot_topography_histogram
    countries : list[str]
        List of country names
    **common_kwargs :
        Keyword arguments forwarded to `func`

    Returns
    -------
    None
    #results : dict[country, output or Exception]
    #    Stores output for each successful country, or an Exception on failure.
    """
    # Deal with case-insensitive country names by converting to uppercase to match data.countries.country_mask['name'] being uppercase
    # Also deal with trailing spaces or extra commas
    if countries is not None:
        countries = [c.strip().upper() for c in countries]

    # Plot everything globally if no countries specified
    if countries is None:
        print("\n----- Global (no country mask) -----")
        try:
            out = func(data, country=None, **common_kwargs)
        except Exception as e:
            print(f"Skipping GLOBAL due to error: {e}")
        return    

    # Otherwise loop through countries
    for country in countries:
        print(f"\n----- {country} -----")
        try:
            out = func(
                data,
                country=country,
                **common_kwargs,
            )
        except Exception as e:
            print(f"Skipping {country!r} due to error: {e}")

    return #results

def run_all_plots(
        data,
        plot_functions,
        countries=None,
        **kwargs
    ):
    """
    Cycle through all `plot_functions` and apply them to each country in "countries" using "loop_over_countries()".

    Parameters
    ----------
    data : LoadBaseSatelliteData
        Container with met/fp/topo datasets, including data.countries.country_mask
    plot_functions : list[callable]
        List of plotting functions to call, e.g. [plot_topography_histogram]
    countries : list[str]
        List of country names. If no countries are specified then plot "globally" (i.e. no country mask applied).
    **common_kwargs :
        Keyword arguments forwarded to `func`

    Returns
    -------
    None
    
    
    """
    for f in plot_functions:
        print(f"--- {f.__name__} ---")
        try:
            loop_over_countries(data, f, countries=countries, **kwargs)
        except Exception as e:
            print(f"   Skipped: {e}")