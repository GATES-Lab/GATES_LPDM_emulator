
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np


import os
import pickle
import glob
import xarray as xr
import pandas as pd
import einops
import cartopy
import cartopy.feature as cfeature


from sklearn.metrics import mean_squared_error, r2_score
import time
from datetime import datetime
import json
import argparse

import random

from matplotlib import gridspec
import matplotlib.dates as mdates
import cartopy.crs as ccrs

def retrend_predictions_cams(dataset, cams_file = None, trend_height=1500, detrended_attr_name = "detrended", retrended_attr_name = "flux", cams_units_multiplier=1e9):
    """
    Retrend modelled and true background concentrations using CAMS data. 
    Takes the monthly average of the CAMS data along the southern boundary of the domain at the specified height, and and adds the monthly mean back to the detrended data.

    dataset - xarray dataset object, with the detrended molefractions saved under true_ and pred_ attributes (e.g. true_bc_detrended)
    cams_file - path to CAMS netCDF file(s), cut to the same domain as the footprints. If None, will use default path based on year in dataset for SOUTHAMERICA domain.
    trend_height - height (in meters) at which to extract CAMS data for correction. Default is 1500m.
    detrended_attr_name - string, suffix of the detrended attributes in the dataset
    retrended_attr_name - string, suffix for the retrended attributes to be created in the dataset
    cams_units_multiplier - multiplier to convert CAMS units to dataset units (default assumes CAMS in mol/mol and dataset in ppb, so multiplier is 1e9. Use cams_units_multiplier=1 if the detrended mf is also in mol/mol)
    """
    if cams_file is None:
        year = np.unique(dataset.time.dt.year.values)
        if len(year)>1:
            raise ValueError("Dataset contains multiple years! please provide a specific CAMS path.")
        cams_file = f"/group/chem/acrg/LPDM/bc/SOUTHAMERICA/ch4_SOUTHAMERICA_{year[0]}*_CAMS-inversion.nc"

    cams = xr.open_mfdataset(sorted(glob.glob(cams_file)), combine='nested', concat_dim='time')
    cams = cams.sel(height=trend_height)

    correction = cams.vmr_s.mean(dim="lon") * cams_units_multiplier
    correction = correction.assign_coords(month=correction['time'].dt.month)
    correction = correction.swap_dims({'time': 'month'}).drop_vars('time')
    # correcting, by month
    to_correct = dataset[[f'true_bc_{detrended_attr_name}', f'pred_bc_{detrended_attr_name}']]
    corrected_subset = to_correct.groupby('time.month') + correction


    # rename the now corrected variables
    corrected_subset = corrected_subset.rename(
        {f'true_bc_{detrended_attr_name}': f'true_bc_{retrended_attr_name}',
         f'pred_bc_{detrended_attr_name}': f'pred_bc_{retrended_attr_name}'}
    )

    corrected_ds = dataset.assign(
        {f'true_bc_{retrended_attr_name}':(("time"), corrected_subset[f'true_bc_{retrended_attr_name}'].values),
         f'pred_bc_{retrended_attr_name}':(("time"), corrected_subset[f'pred_bc_{retrended_attr_name}'].values)}
    )

    return corrected_ds




def plot_bc_timeseries(dataset, start_date, min_length=30, days_to_plot=7, ylim=None, unit_multiplier=1, attribute_name = "bc_detrended", scatter=False, flux=None):
    """
    Plot modelled background fluxes over a given period from a dataset, splitting days into subplots
    dataset - xarray dataset object, with true_ and pred_ attributes
    start_date - string, e.g. "2018-10-01" of the first day
    min_length - minimum number of observations in a day to be included
    days_to_plot - number of days to plot from start date
    ylim - list, as [lower, upper] limits for y-axis, or None, for automatic limits
    unit_multiplier - multiplier to convert data to desired units for plotting
    attribute_name - string, name of the attribute to plot (without true_ or pred_)

    the figure is built by creating a series of subplots with GridSpec, one for each valid date.
    For each valid date, the subplot is split into chunks (n_overpasses) based on time gaps larger than 30 minutes.
    Each chunk is plotted in its own mini-axis within the date's subplot.
    """
    end_date = pd.to_datetime(start_date) + pd.DateOffset(days=days_to_plot)

    bc_selected = dataset.sel(time=slice(start_date, end_date))

    bc_selected['time'] = pd.to_datetime(bc_selected['time'].values)

    grouped = bc_selected.groupby('time.date')
    
    if flux is not None:
        flux_selected = flux.sel(time=slice(start_date, end_date))
        flux_selected['time'] = pd.to_datetime(flux_selected['time'].values)

    gap_threshold = pd.Timedelta('30min')

    attribute_names = {"true": f"true_{attribute_name}", "pred": f"pred_{attribute_name}"}
    if ylim is None:
        if np.any(bc_selected[attribute_names["true"]].values<0):
            # data is detrended
            min_val = np.min(bc_selected[attribute_names["true"]].values)*unit_multiplier
            min_val = np.min(bc_selected[attribute_names["true"]].values)*unit_multiplier
            max_val = np.max(bc_selected[attribute_names["true"]].values)*unit_multiplier
            ylim = [min_val - 0.15*max(abs(min_val), abs(max_val)), max_val + 0.15*max(abs(min_val), abs(max_val))]
        else:
            # data is in original units
            ylim = [1690,1800]

    fig = plt.figure(figsize=(20, 5), dpi=300)

    valid_dates = []
    ignore_dates = []
    for n, (date, group) in enumerate(grouped):
        times = group['time'].values
        if len(times)>min_length and date not in ignore_dates:
            valid_dates.append(date)

            
        else:
            print(f"removing date {date}")

    #print(len(valid_dates))
    outer_gs = gridspec.GridSpec(1,len(valid_dates), figure=fig, wspace=0.15)
  
    first_axis = True
    legend=False

    name_col = "#768732"
    em_col = "#D79706"
    name_col = "orange"
    em_col = "green"
    
    n=0
    #for n, (date, group) in enumerate(grouped):
    for date, group in grouped:
        if date in valid_dates:
            times = group['time'].values
            time_diffs = np.diff(times)
            split_indices = np.where(time_diffs > gap_threshold)[0] + 1

            indices = np.concatenate(([0], split_indices, [len(times)]))
            
            # Filter out chunks with <= 4 time steps
            start_indices = []
            end_indices = []
            for start, end in zip(indices[:-1], indices[1:]):
                
                # remove this condition for now
                if end - start > 4:# and np.mean(group.isel(time=slice(start, end)).true_flux.values/1e-9)>2:
                    start_indices.append(start)
                    end_indices.append(end)
            
            
            n_chunks = len(start_indices)
            
            inner_gs = gridspec.GridSpecFromSubplotSpec(
                1, n_chunks, subplot_spec=outer_gs[n], wspace=0.1
            )

            date_label = date.strftime('%d/%m')  # Format date as DD/MM/YYYY
            outer_ax = fig.add_subplot(outer_gs[n])  # Get the whole subplot
            outer_ax.set_xlabel(date_label, fontsize=12, labelpad=30)  # Set the label
            outer_ax.spines['left'].set_visible(False)  # Hide the left spine of the outer axis
            outer_ax.spines['right'].set_visible(False)  # Hide the right spine
            outer_ax.spines['top'].set_visible(False)  # Hide the top spine
            outer_ax.spines['bottom'].set_visible(False)  # Hide the bottom spine
            outer_ax.set_xticks([])  # Hide x-ticks for the outer axis
            outer_ax.set_yticks([])  # Hide y-ticks for the outer axis

            #for i, (start, end) in enumerate(zip(indices[:-1], indices[1:])):
            for i, (start, end) in enumerate(zip(start_indices, end_indices)):
                chunk = group.isel(time=slice(start, end))
                if flux is not None:
                    chunk_flux = flux_selected.sel(time=chunk.time)
                if n_chunks==1:
                    mini_ax = fig.add_subplot(inner_gs[0])
                else:
                    mini_ax = fig.add_subplot(inner_gs[0, i])

                mini_ax.plot(chunk["time"], chunk[attribute_names["true"]].values*unit_multiplier, c=name_col, lw=2)
                mini_ax.scatter(chunk["time"], chunk[attribute_names["true"]].values*unit_multiplier, c=name_col, label="with NAME bc", lw=2, s=5)
                mini_ax.plot(chunk["time"], chunk[attribute_names["pred"]].values*unit_multiplier, c=em_col, lw=2)
                mini_ax.scatter(chunk["time"], chunk[attribute_names["pred"]].values*unit_multiplier, c=em_col, label="with GATES bc", lw=2, s=5)

                if flux is not None:
                    mini_ax.plot(chunk["time"], chunk[attribute_names["true"]].values*unit_multiplier + chunk_flux["flux"].values*unit_multiplier, c=name_col, ls="--", lw=2)

                    #mini_ax.scatter(chunk["time"], chunk[attribute_names["true"]].values*unit_multiplier, c=name_col, ls="--", label="with NAME bc", lw=2, s=5)
                    mini_ax.plot(chunk["time"], chunk[attribute_names["pred"]].values*unit_multiplier + chunk_flux["flux"].values*unit_multiplier, c=em_col, ls="--", lw=2)
                    #mini_ax.plot(chunk["time"], chunk[attribute_names["pred"]].values*unit_multiplier, c=em_col, lw=2)
                    #mini_ax.scatter(chunk["time"], chunk[attribute_names["pred"]].values*unit_multiplier, c=em_col, ls="--", label="with GATES bc", lw=2, s=5)                    

                mini_ax.set_ylim(ylim[0],ylim[1])
                
                if not first_axis:
                    mini_ax.yaxis.set_visible(False)
                    mini_ax.spines['left'].set_visible(False)
                    mini_ax.spines['right'].set_visible(False)
                else:
                    first_axis=False
                    mini_ax.set_ylabel("ppb", fontsize=20)
                    mini_ax.spines['right'].set_visible(False)
                    

                mini_ax.spines['top'].set_visible(False)
            
                # Set x-ticks at 10-minute intervals
                time_range = chunk["time"].values
                if len(time_range)>1:
                    tick_interval = np.timedelta64(5, 'm')  # 10-minute interval
                    tick_times = np.arange(time_range[0], time_range[-1], tick_interval)
                else:
                    tick_times = [time_range[0]]

                # Format ticks
                try:
                    mini_ax.set_xticks([tick_times[0]])
                    mini_ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  
                except:
                    continue  

                mini_ax.set_xticks(np.arange(time_range[0], time_range[-1], np.timedelta64(2, 'm')), minor=True)
                mini_ax.tick_params(axis='x', which='minor', length=3, width=1)  # Minor ticks without labels

            if not legend:
                mini_ax.legend(bbox_to_anchor=(3, 1), fontsize=12)
                mini_ax.set_zorder(1)
                legend=True

            n=n+1
        else:
            print(f"ignoring date {date}")


    if flux is None:
        fig.suptitle(f"Modelled background concentration - {start_date[:4]}",fontsize=20)
    else:
        fig.suptitle(f"Modelled background concentration (-) and background + flux (--) - {start_date[:4]}",fontsize=20)

    plt.show()


def plot_fp_and_bcs(fp_dataset, idx=0, timestamp=None, levels=None, bc_minmax=(-4, -1), figsize=6):
    """
    Plot a full footprint and the corresponding boundary condition footprints on each direction.
    Parameters:
    fp_dataset: xarray.Dataset of footprints
    idx: index of the time dimension to plot (used if timestamp is None)
    timestamp: specific timestamp to plot (overrides idx if provided)
    levels: contour levels for the footprint plot
    bc_minmax: min and max values for boundary condition plots, logged
    figsize: size of the figure
    
    todo - add colorbar!
    """
    if timestamp is not None:
        # check that timestamp is valid and inside the footprint time
        timestamp = pd.to_datetime(timestamp)
        if timestamp not in fp_dataset.time.values:
            raise ValueError("Provided timestamp is not in the footprint dataset time values.")
    else:
        timestamp = pd.to_datetime(fp_dataset.time.values[idx])

    f = fp_dataset.sel(time=timestamp).copy()
    fp = np.copy(f.fp.values)

    lon_values = f.lon.values
    lat_values = f.lat.values

    width_ratios=[1, 6, 1]
    height_ratios=[1, 12, 1]
    figsize = (figsize, figsize)
    # create figure and subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, width_ratios=width_ratios, height_ratios=height_ratios)

    # plot footprint on the central subplot
    ax_main = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
    extent = (np.min(lon_values), np.max(lon_values), np.min(lat_values), np.max(lat_values))
    ax_main.set_extent(extent, crs=cartopy.crs.PlateCarree())
    ax_main.coastlines(resolution='110m', color='black', linewidth=1, alpha=0.5)
    ax_main.add_feature(cfeature.LAND)
    ax_main.add_feature(cfeature.OCEAN)  

    cmap = plt.cm.Reds
    cmap.set_over = "k"
    alpha =0.4

    plot_params = {"transform":cartopy.crs.PlateCarree(), "cmap":cmap}

    contour = True
    if levels is None:
        levels = [-4.5, -4, -3.5, -3,  -2.5, -2, -1.5]
    thres=1e-4
    if contour:
        cb = ax_main.contourf(lon_values, lat_values, np.log10(fp),  **plot_params, levels=levels, extend="both", alpha=alpha)
        fp[fp<thres] = 0
        cb = ax_main.contourf(lon_values, lat_values, np.log10(fp), **plot_params, levels=levels, extend="both")


    ## plot the four sides
    bc_min, bc_max = bc_minmax
    bc_cmap = plt.cm.hot
    bc_cmap.set_over("yellow")
    bc_cmap.set_under("k")

    ax_north = fig.add_subplot(gs[0, 1])
    c_north = ax_north.contourf(np.where(f.particle_locations_n==0, np.nan, np.log10(f.particle_locations_n)), origin="lower", extend="both", cmap=bc_cmap, vmin=bc_min, vmax=bc_max)
    ax_north.set_xticks([])
    ax_north.set_yticks([0,15])

    ax_north.set_title(f"FP at {timestamp.strftime('%Y-%m-%d %H:%M UTC')}", fontsize=12)

    ax_south = fig.add_subplot(gs[2, 1])
    c_south = ax_south.contourf(np.where(f.particle_locations_s==0, np.nan, np.log10(f.particle_locations_s)), origin="lower", extend="both", cmap=bc_cmap, vmin=bc_min, vmax=bc_max)
    ax_south.set_xticks([])
    ax_south.set_yticks([0,15])
    ax_south.invert_yaxis()

    ax_west = fig.add_subplot(gs[1, 0])
    c_west = ax_west.contourf(np.where(f.particle_locations_w==0, np.nan, np.log10(f.particle_locations_w)).T, origin="lower",  cmap=bc_cmap, vmin=bc_min, vmax=bc_max)
    ax_west.invert_xaxis()
    ax_west.set_yticks([])
    ax_west.set_xticks([0,15])

    ax_east = fig.add_subplot(gs[1, 2])
    c_east = ax_east.contourf(np.where(f.particle_locations_e==0, np.nan, np.log10(f.particle_locations_e)).T, origin="lower", extend="both", cmap=bc_cmap, vmin=bc_min, vmax=bc_max)
    ax_east.set_yticks([])
    ax_east.set_xticks([0,15])

    #return fig