"""
author: Elena Fillola @elenafillo
"""

import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np

sys.path.insert(0,"/software/local/languages/miniforge3/envs/elena/lib/python3.12/site-packages/")

sys.path.insert(0,"/user/work/ef17148/oldstuff/ef17148/.conda/envs/new_graphnet/lib/python3.12/site-packages")

print(sys.path)

import torch
import os
import pickle
import einops

#import plenoptic as po

from model.data.load_data import *
#from loss_functions import *
#from evaluation import *

import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
import time
from datetime import datetime
import json
import argparse
import cartopy
import cartopy.feature as cfeature
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import Normalize
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
import random

"""
Functions used to build the figures and metrics from the paper
These load full footprint files - they compare LPDM outputs with the predictions in the same format

to achieve this format, use integrate_fps.py after you have general_make_prediction.py to make model predictions (we could probably streamline this?) 

These functions still need to be documented/improved

author: Elena Fillola @elenafillo

"""


def MAE(true, pred):
    pred= np.copy(pred.flatten())
    true= np.copy(true.flatten())
    #mae = np.mean(np.abs(true-pred))
    mask = (true != 0) & (pred != 0) & ~np.isnan(true) & ~np.isnan(pred) & ~np.isinf(true) & ~np.isinf(pred)
    mae = np.mean(np.abs(true[mask]-pred[mask]))
    return mae 


def get_stats(mod, i, do_print=True, do_return=False):
    valid_nans = ~np.isnan(mod.all_fps.fp_pred[i].values.flatten())
    valid_zeros = np.bitwise_and(mod.all_fps.fp_pred[i].values.flatten()>0, mod.all_fps.fp_true[i].values.flatten()>0)
    
    valid_mask = np.bitwise_and(valid_nans, valid_zeros)


    mse = mean_squared_error(mod.all_fps.fp_true[i].values.flatten()[valid_mask], 
                             mod.all_fps.fp_pred[i].values.flatten()[valid_mask])
    log_mse = mean_squared_error(np.log10(mod.all_fps.fp_true[i].values.flatten()[valid_mask]), 
                             np.log10(mod.all_fps.fp_pred[i].values.flatten()[valid_mask]))   
    corrcoef = np.corrcoef(mod.all_fps.fp_true[i].values.flatten()[valid_mask], 
                           mod.all_fps.fp_pred[i].values.flatten()[valid_mask])[0, 1]
    
    log_corrcoef = np.corrcoef(np.log10(mod.all_fps.fp_true[i].values.flatten()[valid_mask]), 
                           np.log10(mod.all_fps.fp_pred[i].values.flatten()[valid_mask]))[0, 1]


    stats_dict = {
        
        "MSE": mse,
        "Corrcoef": corrcoef,
        "log MSE": log_mse,
        "log Corrcoef": log_corrcoef
    }
    
    if do_print:
        print(i)

    if do_return:
        return stats_dict


def plot_footprint_from_model(data, idx, size, lon_squeeze=0, vmin=None, vmax=None):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(projection=cartopy.crs.Mercator())
    extent = (data.fp_lons[idx,lon_squeeze], data.fp_lons[idx,-1-lon_squeeze], data.fp_lats[idx,0], data.fp_lats[idx,-1])
    ax.set_extent(extent, crs=cartopy.crs.PlateCarree())
    ax.coastlines(resolution='110m', color='black', linewidth=1, alpha=0.5)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    #ax.scatter(data.fp_lons[idx,int(size/2)], data.fp_lats[idx,int(size/2)], marker="x", transform=cartopy.crs.PlateCarree(), linewidths=2, color="#aa5b3a", s=70, zorder=10)
    f = np.reshape(np.copy(data.fp_data[idx]), (size,size))
    f[f<0.0001] = 0
    cb = ax.imshow(np.log10(f), extent=extent, origin="lower", transform=cartopy.crs.PlateCarree(), zorder=5)

def plot_a_day_ax(ax, subgs,fig, day, model, labels=False,ylim=45, days_before=0, days_after=0, dateline_idx=None,title=None):
    import matplotlib.gridspec as gridspec
    import matplotlib.dates as mdates

    fig.delaxes(ax)
    

    outer_gs = gridspec.GridSpecFromSubplotSpec(1,1+days_before+days_after, subplot_spec=subgs, wspace=0.15)

    
    period = (np.datetime64(day)-np.timedelta64(days_before, 'D'), np.datetime64(day)+np.timedelta64(days_after, 'D'))
    period = (str(period[0]), str(period[1]))
    
    fp_selected = model.all_fps.sel(time=slice(period[0], period[1]))

    fp_selected['time'] = pd.to_datetime(fp_selected['time'].values)

    grouped = fp_selected.groupby('time.date')
    gap_threshold = pd.Timedelta('30min')

    first_axis = True

    for n, (date, group) in enumerate(grouped):
        ax = fig.add_subplot(outer_gs[n])
        ax.set_xlabel(date.strftime('%d/%m'), fontsize=12, labelpad=24)  # Set the label ## labelpad=30


        ax.spines['left'].set_visible(False)  # Hide the left spine of the outer axis
        ax.spines['right'].set_visible(False)  # Hide the right spine
        ax.spines['top'].set_visible(False)  # Hide the top spine
        ax.spines['bottom'].set_visible(False)  # Hide the bottom spine
        ax.set_xticks([])  # Hide x-ticks for the outer axis
        ax.set_yticks([])  # Hide y-ticks for the outer axis

        times = group['time'].values

        
        time_diffs = np.diff(times)
        split_indices = np.where(time_diffs > gap_threshold)[0] + 1
        indices = np.concatenate(([0], split_indices, [len(times)]))
        #n_chunks =  len(indices) - 1

        # Filter out chunks with <= 4 time steps
        start_indices = []
        end_indices = []
        for start, end in zip(indices[:-1], indices[1:]):

            if end - start > 4 and np.mean(group.isel(time=slice(start, end)).true_flux.values/1e-9)>2:
                start_indices.append(start)
                end_indices.append(end)
        n_chunks = len(start_indices)
        
        inner_gs = gridspec.GridSpecFromSubplotSpec(1, n_chunks, subplot_spec=outer_gs[n], wspace=0.1)
        #inner_gs = gridspec.GridSpecFromSubplotSpec(1, n_chunks, subplot_spec=subgs, wspace=0.1)

        name_col = "#768732"
        em_col = "#D79706"
        name_col = "orange"
        em_col = "green"


        #for i, (start, end) in enumerate(zip(indices[:-1], indices[1:])):
        for i, (start, end) in enumerate(zip(start_indices, end_indices)):
            chunk = group.isel(time=slice(start, end))
            mini_ax = fig.add_subplot(inner_gs[0, i])
            #mini_ax = fig.add_subplot(subgs)

            mini_ax.plot(chunk["time"], chunk.true_flux.values/1e-9, c=name_col, label="with NAME footprints", lw=2)
            mini_ax.plot(chunk["time"], chunk.pred_flux.values/1e-9, c=em_col, label="with GATES footprints", lw=2)
            

            if not first_axis:
                mini_ax.yaxis.set_visible(False)
                mini_ax.spines['left'].set_visible(False)
                mini_ax.spines['right'].set_visible(False)
            else:
                first_axis=False
                if labels:
                    mini_ax.set_ylabel("above baseline \n mole fractions \n (ppb)", fontsize=11)
                    mini_ax.legend(loc="upper right", bbox_to_anchor=(5.5, 1))
                    mini_ax.set_zorder(1)

                    #mini_ax.set_yticklabels(list(range(0, ylim, 10)))#, list(range(0, ylim, 10)))

            if dateline_idx is not None:
                if pd.to_datetime(str(model.all_fps.time[dateline_idx].values)) in chunk.time:
                    mini_ax.axvline(pd.to_datetime(str(model.all_fps.time[dateline_idx].values)), alpha=0.5, ls = "--", color="k")

                #else:
                    #mini_ax.set_yticks([])
                mini_ax.set_yticks(list(range(0, ylim, 2)), minor=True)
                mini_ax.set_yticks(list(range(0, ylim, 10)), list(range(0, ylim, 10)), minor=False)
                
                mini_ax.spines['right'].set_visible(False)

            mini_ax.spines['top'].set_visible(False)
            
            mini_ax.set_ylim(0,ylim)
            
            # Set x-ticks at 10-minute intervals
            time_range = chunk["time"].values
            tick_interval = np.timedelta64(5, 'm')  # 10-minute interval
            tick_times = np.arange(time_range[0], time_range[-1], tick_interval)
            # Format ticks
            mini_ax.set_xticks([tick_times[0]])
            mini_ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))    
            ax.tick_params(axis='both', which='major', labelsize=5)

            mini_ax.set_xticks(np.arange(time_range[0], time_range[-1], np.timedelta64(2, 'm')), minor=True)
            mini_ax.tick_params(axis='x', which='minor', length=3, width=1)  # Minor ticks without labels
            

def plot_hist_ax(ax, data, idx, plot_zeros=True, plot_corrcoef=True, print_stats=True, plot_logged=True, ylabels=False):

    zeros_as = -5.8
    pred= np.copy(data.all_fps.fp_pred[idx].values.flatten())
    true = np.copy(data.all_fps.fp_true[idx].values.flatten())

    if plot_logged and not plot_zeros:
        mask = (true != 0) & (pred != 0) & ~np.isnan(true) & ~np.isnan(pred) & ~np.isinf(true) & ~np.isinf(pred)

        ax.scatter(np.log10(true[mask]), np.log10(pred[mask]), edgecolor='k', facecolor='none', marker='.', alpha=0.1)
        ax.set_xlabel("LPDM footprint value")
        if ylabels:
            ax.set_ylabel("GATES footprint value")
        #ax.set_xlabel("log(f)")

    elif plot_logged:
        mask = ~np.isnan(true) & ~np.isnan(pred) & ~np.isinf(true) & ~np.isinf(pred)
        
        ax.scatter(np.where(true[mask]>0, np.log10(true[mask]), zeros_as), np.where(pred[mask]>0, np.log10(pred[mask]), zeros_as), edgecolor='k', facecolor='none', marker='.', alpha=0.1)

        ax.set_xticks([zeros_as,-5,-4,-3,-2,-1], labels=[r"f=0",-5,-4,-3,-2,-1])
        ax.set_yticks([zeros_as,-5,-4,-3,-2,-1], labels=[r"$\hat{f}$=0",-5,-4,-3,-2,-1])
        ax.set_xlabel("   log(LPDM footprint value)")
        if ylabels:
            ax.set_ylabel("log(GATES footprint value)")        
        #ax.set_ylabel(r"       log($\hat{f}$)")
        ax.plot([-5,-1], [-5,-1])

    else:
        mask = ~np.isnan(true) & ~np.isnan(pred) & ~np.isinf(true) & ~np.isinf(pred)
        ax.scatter(true[mask], pred[mask], edgecolor='k', facecolor='none', marker='.', alpha=0.2)
        ax.plot([0,np.max(true[mask])], [0,np.max(true[mask])])    
        ax.set_xlabel("LPDM footprint value")
        if ylabels:
            ax.set_ylabel("GATES footprint value")

    if plot_corrcoef:
        plot_high_corr_coeffs = False
        if plot_logged:
            if plot_high_corr_coeffs:
                mask = (true != 0) & (pred != 0) & ~np.isnan(true) & ~np.isnan(pred) & ~np.isinf(true) & ~np.isinf(pred) 
                second_mask = (np.log10(true[mask])>-3) | (np.log10(pred[mask])>-3) 

                z = np.polyfit(np.log10(true[mask])[second_mask], np.log10(pred[mask])[second_mask], 1)
                p = np.poly1d(z)
                _xx = np.arange(-3, -1, 0.1)
                ax.plot(_xx, p(_xx), color='red', ls='dotted')

            mask = (true != 0) & (pred != 0) & ~np.isnan(true) & ~np.isnan(pred) & ~np.isinf(true) & ~np.isinf(pred)
            z = np.polyfit(np.log10(true[mask]), np.log10(pred[mask]), 1)
            p = np.poly1d(z)
            _xx = np.arange(-5, -1, 0.1)
            ax.plot(_xx, p(_xx), color='red', alpha=0.9, ls='--')
        else:
            mask = (true != 0) & (pred != 0) & ~np.isnan(true) & ~np.isnan(pred) & ~np.isinf(true) & ~np.isinf(pred)
            z = np.polyfit(true[mask], pred[mask], 1)
            p = np.poly1d(z)
            print(z)
            _xx = np.arange(0, np.max(true[mask]), 0.0005)
            ax.plot(_xx, p(_xx), color='red', alpha=0.9, ls='--')            

    if print_stats:
        stats = get_stats(data, idx, do_print=False, do_return=True)
        if plot_logged:
            # Use single quotes for the f-string so double-quoted dict keys are valid
            ax.text(0.05, 0.95, f'Log Corr {str(stats["log Corrcoef"])[:4]}', transform=ax.transAxes,
                    ha="left", va="top", fontsize=12)
        else:
            ax.text(0.05, 0.95,
                    f'IoU: {str(100*stats["IoU"])[:4]}% \nMSE: {str(stats["MSE"])[:4]+str(stats["MSE"])[-4:]} \nCorr {str(stats["Corrcoef"])[:4]}',
                    transform=ax.transAxes, ha="left", va="top", fontsize=12)
        
def plot_footprint_ax(ax_true, ax_pred, model, idx, lon_squeeze=0, vmin=None, vmax=None, thres=0, contour=False, share_minmax=False, cbar=True,nlevels=None, title=False, return_minmax=False, print_stats=False, plot_cities=False):


    np.seterr(divide='ignore')

    extent = (model.all_fps.fp_lon[idx,lon_squeeze], model.all_fps.fp_lon[idx,-1-lon_squeeze], model.all_fps.fp_lat[idx,0], model.all_fps.fp_lat[idx,-1])
    ax_true.set_extent(extent, crs=cartopy.crs.PlateCarree())
    ax_true.coastlines(resolution='110m', color='black', linewidth=1, alpha=0.5)
    ax_true.add_feature(cfeature.LAND)
    ax_true.add_feature(cfeature.OCEAN)    

    alpha =0.4

    f = np.copy(model.all_fps.fp_true[idx])
    #f[f<thres] = 0
    #f[f<0.0001] = 0

    if share_minmax:
        if vmin is None:
            vmin=np.min(np.log10(f[f>0]))
        if vmax is None:
            vmax=np.nanmax(np.log10(f))

        #magma_r
    cmap = plt.cm.Reds
    cmap.set_over = "k"
    plot_params = {"transform":cartopy.crs.PlateCarree(), "cmap":cmap, "vmin":vmin, "vmax":vmax}

    if contour:
        cb = ax_true.contourf(model.all_fps.fp_lon.values[idx], model.all_fps.fp_lat.values[idx],np.log10(f), **plot_params, levels=nlevels, extend="both", alpha=alpha)
        f[f<thres] = 0
        cb = ax_true.contourf(model.all_fps.fp_lon.values[idx], model.all_fps.fp_lat.values[idx],np.log10(f), **plot_params, levels=nlevels, extend="both")


    else:
        cb = ax_true.imshow(np.log10(f), extent=extent, origin="lower", **plot_params, zorder=5)

    if title:
        #time_str = np.datetime_as_string(mod.all_fps.time.values[idx], unit='m')
        #formatted_time = np.datetime64(time_str).astype('datetime64[m]').astype('O').strftime('%d-%m-%Y %H:%M')
        formatted_time = model.all_fps.time.values[idx].astype('datetime64[m]').astype('O').strftime('%d-%m-%Y %H:%M')
        ax_true.set_title(formatted_time)
        #ax_true.set_title(str(np.datetime_as_string(mod.all_fps.time.values[idx], unit='m'))[:10]+ " " + str(np.datetime_as_string(mod.all_fps.time.values[idx], unit='m'))[11:])


    ax_pred.set_extent(extent, crs=cartopy.crs.PlateCarree())
    ax_pred.coastlines(resolution='110m', color='black', linewidth=1, alpha=0.5)
    ax_pred.add_feature(cfeature.LAND)
    ax_pred.add_feature(cfeature.OCEAN)    


    f = np.copy(model.all_fps.fp_pred[idx])
    #f[f<0.0001] = 0
    #f[f<thres] = 0
    
    brazil_cities = {
        "Rio de Janeiro": (-22.9068, -43.1729),
        "São Paulo": (-23.5505, -46.6333),
        "Brasília": (-15.7801, -47.9292),
        "Salvador": (-12.9714, -38.5014),
        "Fortaleza": (-3.7172, -38.5437)}
    
    south_america_cities = {
        "São Paulo, Brazil": (-23.5505, -46.6333),
        "Buenos Aires, Argentina": (-34.6037, -58.3816),
        "Rio de Janeiro, Brazil": (-22.9068, -43.1729),
        "Lima, Peru": (-12.0464, -77.0428),
        "Bogotá, Colombia": (4.7110, -74.0721),
        "Santiago, Chile": (-33.4489, -70.6693),
        "Caracas, Venezuela": (10.4806, -66.9036),
        "Quito, Ecuador": (-0.1807, -78.4678),
        "São Luís, Brazil": (-2.5391, -44.2823),
        "Fortaleza, Brazil": (-3.7172, -38.5437),
        "Belo Horizonte, Brazil": (-19.9191, -43.9346),
        "Manaus, Brazil": (-3.1190, -60.2442),
        "Recife, Brazil": (-8.0476, -34.8770),
        "Curitiba, Brazil": (-25.4297, -49.2717),
        "Porto Alegre, Brazil": (-30.0331, -51.2300)
    }

    main_cities = {
                "Rio de Janeiro": (-22.9068, -43.1729),
        "São Paulo": (-23.5505, -46.6333)}

    if plot_cities:
        for city, (lat, lon) in main_cities.items():
            ax_true.scatter(lon, lat, color='k',  s=10,transform=ccrs.PlateCarree(), label=city, zorder=100)
            ax_pred.scatter(lon, lat, color='k',  s=10,transform=ccrs.PlateCarree(), label=city, zorder=100)
    

    if print_stats:
        stats = get_stats(model, idx, do_print=False, do_return=True)
        ax_pred.text(
        0.5, -0.1,
        f'IoU: {str(100*stats["IoU"])[:4]}% \n MSE: {str(stats["MSE"])[:4]+str(stats["MSE"])[-4:]} \n Corr {str(stats["Corrcoef"])[:4]}\n',
        transform=ax_pred.transAxes,
        ha="center", va="top", fontsize=12
    )

    if contour:
        cb = ax_pred.contourf(model.all_fps.fp_lon.values[idx], model.all_fps.fp_lat.values[idx],np.log10(f), **plot_params, levels=nlevels, extend="both",alpha=alpha)

        f[f<thres] = 0

        cb = ax_pred.contourf(model.all_fps.fp_lon.values[idx], model.all_fps.fp_lat.values[idx],np.log10(f), **plot_params, levels=nlevels, extend="both")
    else:
        cb = ax_pred.imshow(np.log10(f), extent=extent, origin="lower", **plot_params, zorder=5)
    
    if return_minmax:
        return vmin, vmax
    if cbar:
        return cb


def plot_emissions_ax(ax_ems, ax_true, ax_pred, model, idx, lon_squeeze=0, vmin=-13, vmax=-9, thres=0, contour=False, share_minmax=False, cbar=True,nlevels=None, title=False, return_minmax=False, print_stats=False, plot_cities=False):


    np.seterr(divide='ignore')

    extent = (model.all_fps.fp_lon[idx,lon_squeeze], model.all_fps.fp_lon[idx,-1-lon_squeeze], model.all_fps.fp_lat[idx,0], model.all_fps.fp_lat[idx,-1])
    ax_true.set_extent(extent, crs=cartopy.crs.PlateCarree())
    ax_true.coastlines(resolution='110m', color='black', linewidth=1, alpha=0.5)
    ax_true.add_feature(cfeature.LAND)
    ax_true.add_feature(cfeature.OCEAN)    


    if share_minmax:
        if vmin is None:
            vmin=np.min(np.log10(f[f>0]))
        if vmax is None:
            vmax=np.nanmax(np.log10(f))

        #magma_r

    plot_params = {"transform":cartopy.crs.PlateCarree(), "cmap":"Purples", "vmin":vmin, "vmax":vmax}

    if contour:
        print("contour isnt available yet")
        # cb = ax_true.contourf(model.all_fps.release_lon.values[idx], model.all_fps.release_lat.values[idx],np.log10(f), **plot_params, levels=nlevels, extend="both")
    else:
        if ax_ems is not None:
            _ = ax_ems.imshow(np.log10(model.emissions[idx]), extent=extent, origin="lower", zorder=5, cmap="Blues", vmin=-15, vmax=-6, transform=cartopy.crs.PlateCarree())

        cb = ax_true.imshow(np.log10(model.emissions[idx]*model.all_fps.fp_true[idx]), extent=extent, origin="lower", **plot_params, zorder=5)

        cb = ax_pred.imshow(np.log10(model.emissions[idx]*model.all_fps.fp_pred[idx]), extent=extent, origin="lower", **plot_params, zorder=5)

    if title:
        #time_str = np.datetime_as_string(mod.all_fps.time.values[idx], unit='m')
        #formatted_time = np.datetime64(time_str).astype('datetime64[m]').astype('O').strftime('%d-%m-%Y %H:%M')
        formatted_time = model.all_fps.time.values[idx].astype('datetime64[m]').astype('O').strftime('%d-%m-%Y %H:%M')
        ax_true.set_title(formatted_time)
        #ax_true.set_title(str(np.datetime_as_string(mod.all_fps.time.values[idx], unit='m'))[:10]+ " " + str(np.datetime_as_string(mod.all_fps.time.values[idx], unit='m'))[11:])
    if ax_ems is not None:
        all_ax =[ax_ems, ax_true, ax_pred]
    else:
        all_ax =[ax_true, ax_pred]
    for axis in all_ax:
        axis.set_extent(extent, crs=cartopy.crs.PlateCarree())
        axis.coastlines(resolution='110m', color='black', linewidth=1, alpha=0.5)
        axis.add_feature(cfeature.LAND)
        axis.add_feature(cfeature.OCEAN)    
    
    


    f = np.copy(model.all_fps.fp_pred[idx])
    
    brazil_cities = {
        "Rio de Janeiro": (-22.9068, -43.1729),
        "São Paulo": (-23.5505, -46.6333),
        "Brasília": (-15.7801, -47.9292),
        "Salvador": (-12.9714, -38.5014),
        "Fortaleza": (-3.7172, -38.5437)}
    
    south_america_cities = {
        "São Paulo, Brazil": (-23.5505, -46.6333),
        "Buenos Aires, Argentina": (-34.6037, -58.3816),
        "Rio de Janeiro, Brazil": (-22.9068, -43.1729),
        "Lima, Peru": (-12.0464, -77.0428),
        "Bogotá, Colombia": (4.7110, -74.0721),
        "Santiago, Chile": (-33.4489, -70.6693),
        "Caracas, Venezuela": (10.4806, -66.9036),
        "Quito, Ecuador": (-0.1807, -78.4678),
        "São Luís, Brazil": (-2.5391, -44.2823),
        "Fortaleza, Brazil": (-3.7172, -38.5437),
        "Belo Horizonte, Brazil": (-19.9191, -43.9346),
        "Manaus, Brazil": (-3.1190, -60.2442),
        "Recife, Brazil": (-8.0476, -34.8770),
        "Curitiba, Brazil": (-25.4297, -49.2717),
        "Porto Alegre, Brazil": (-30.0331, -51.2300)
    }

    if plot_cities:
        for city, (lat, lon) in south_america_cities.items():
            ax_true.scatter(lon, lat, color='k', alpha=0.8, s=6,transform=ccrs.PlateCarree(), label=city, zorder=100)
            ax_pred.scatter(lon, lat, color='k',  s=10,transform=ccrs.PlateCarree(), label=city, zorder=100)
    

    if print_stats:
        stats = get_stats(model, idx, do_print=False, do_return=True)
        ax_pred.text(
        0.5, -0.1,
        f'IoU: {str(100*stats["IoU"])[:4]}% \n MSE: {str(stats["MSE"])[:4] + str(stats["MSE"])[-4:]} \n Corr {str(stats["Corrcoef"])[:4]}',
        transform=ax_pred.transAxes,
        ha="center", va="top", fontsize=12
    )  

    
    if return_minmax:
        return vmin, vmax
    if cbar:
        return cb
        #cbar = plt.colorbar(cb, ax=[ax_true, ax_pred], location='right', label='Number of data points', extend="both", shrink=0.5)




from scipy.stats import binned_statistic_2d


def bin_data(data, lats, lons, degree_bins=1):
    # Nawid- This function bins scattered data into a latitude/longitude grid, computing the mean value of the data in each grid cell.
    # Nawid - Creates latitude and longitude bin edges at a specified degree resolution.
    lat_bins = range(int(np.floor(lats.min())), int(np.ceil(lats.max())) + 1, degree_bins) 
    lon_bins = range(int(np.floor(lons.min())), int(np.ceil(lons.max())) + 1, degree_bins) 
    # Nawis -This assigns each point (lats[i], lons[i]) to a bin, and computes the mean value of data inside each lat/lon grid cell
    binned_data, lat_edges, lon_edges, binnumber = binned_statistic_2d(
        lats, lons, data, statistic='mean', bins=[lat_bins, lon_bins]
    )
    return binned_data, lat_edges, lon_edges


from scipy.stats import binned_statistic_2d
import numpy as np
# Nawid.- version just to get the density
def bin_density(lats, lons, degree_bins=1):
    """
    Bin data point density into a lat/lon grid.
    Returns a 2D array of point counts.
    """
    lat_bins = range(int(np.floor(lats.min())), int(np.ceil(lats.max())) + 1, degree_bins)
    lon_bins = range(int(np.floor(lons.min())), int(np.ceil(lons.max())) + 1, degree_bins)

    density, lat_edges, lon_edges, binnumber = binned_statistic_2d(
        lats, lons, None, statistic='count', bins=[lat_bins, lon_bins]
    )

    return density, lat_edges, lon_edges

def get_binned_seasonal_density(dates, release_lats, release_lons, degree_bins=2):
    seasons = {"JFM":[1,2,3], "AMJ":[4,5,6], "JAS":[7,8,9], "OND":[10,11,12]}
    
    season_results = {}

    for seas, months in seasons.items():
        # seasonal time indices
        idxs = np.where(np.isin(dates.dt.month, months))[0]

        print(f"{seas} | {len(idxs)} points")

        # get seasonal subset
        s_lats = release_lats[idxs]
        s_lons = release_lons[idxs]

        # compute density map
        density_binned, lat_edges, lon_edges = bin_density(
            s_lats, s_lons,
            degree_bins=degree_bins
        )

        season_results[seas] = {
            "lat_edges": lat_edges,
            "lon_edges": lon_edges,
            "density_binned": density_binned,
        }

    return season_results


def vcorrcoef(fp, pred, threshold = None, return_mean=False):
    # Nawid - reshape to 2d
    X = fp.reshape(fp.shape[0], -1)
    y = pred.reshape(pred.shape[0], -1)
    # Nawid -  build a mask to filter invalid values
    if threshold is None:
        mask = (X != 0) & (y != 0) & ~np.isnan(X) & ~np.isnan(y) & ~np.isinf(X) & ~np.isinf(y)
    else:
        mask = (X != 0) & (y != 0) & ~np.isnan(X) & ~np.isnan(y) & ~np.isinf(X) & ~np.isinf(y) & (X > threshold) & (y > threshold)
    # Nawid.- replace masked values with nans
    X_masked = np.where(mask, X, np.nan)
    y_masked = np.where(mask, y, np.nan)

    # Nawid -Compute correlation coefficient per sample (vectorized Pearson correlation)
    Xm = np.nanmean(X_masked, axis=1, keepdims=True)
    ym = np.nanmean(y_masked, axis=1)
    r_num = np.nansum((X_masked - Xm) * (y_masked - ym[:, None]), axis=1)
    r_den = np.sqrt(np.nansum((X_masked - Xm) ** 2, axis=1) * np.nansum((y_masked - ym[:, None]) ** 2, axis=1))

    r = r_num / r_den

    return np.nanmean(r) if return_mean else r

def get_binned_seasonal_scores(true_fps, predictions, dates, release_lats, release_lons, degree_bins=2):
    seasons = {"JFM":["01","02","03"], "AMJ":["04","05","06"], "JAS":["07","08","09"], "OND":["10","11","12"]}

    # FIX HERE ↓↓↓  (your data is already 1D!)
    
    all_mses = ((true_fps - predictions) ** 2).flatten()
    print("all_mses shape:", all_mses.shape)
    print("true_fps shape:", true_fps.shape)
    print("predictions shape:", predictions.shape)

    season_results = {}
    for n, seas in enumerate(seasons.keys()):
        seasonal_idxs = np.where(
            np.bitwise_and(
                dates.dt.month >= int(seasons[seas][0]),
                dates.dt.month <= int(seasons[seas][-1])
            )
        )[0]
        print(
            "Season:", seas,
            " | len(mses):", len(all_mses[seasonal_idxs]),
            " | len(lats):", len(release_lats[seasonal_idxs]),
            " | len(lons):", len(release_lons[seasonal_idxs]),
        )
        mses_binned, lat_edges, lon_edges = bin_data(
            all_mses[seasonal_idxs],
            release_lats[seasonal_idxs],
            release_lons[seasonal_idxs],
            degree_bins=degree_bins
        )

        season_results[seas] = {
            "lat_edges": lat_edges,
            "lon_edges": lon_edges,
            "mses_binned": mses_binned,
        }

    return season_results

def get_binned_seasonal_scores_old(true_fps, predictions, dates, release_lats, release_lons, degree_bins=2):
    '''
    dates	Array or pandas datetime index of length time
    release_lats, release_lons	The latitude/longitude for each time index (shape = time)
    degree_bins	Grid size (in degrees) for aggregating metrics spatially
    '''
    # Nawid- this function splits the dataset into 4 season, computes perofrmance metrics, them bins the scores into a spatial grid
    seasons = {"JFM":["01", "02", "03"], "AMJ":["04", "05", "06"], "JAS":["07", "08", "09"], "OND":["10", "11", "12"]}

    #all_mses = np.nanmean((true_fps - predictions) ** 2, axis=(1, 2))
    print("all_mses shape:", all_mses.shape)
    print("true_fps shape:", true_fps.shape)
    print("predictions shape:", predictions.shape)
    all_mses = (true_fps - predictions) ** 2
    #all_corrcoeffs = vcorrcoef(true_fps, predictions, return_mean=False)
    #all_logged_corrcoeffs = vcorrcoef(np.log10(true_fps), np.log10(predictions), return_mean=False)

    season_results = {}
    degree_bins = 2
    # Nawid - Loop through each season
    for n, seas in enumerate(list(seasons.keys())):
            print(n, seas)
            #seasonal_idxs = np.bitwise_and(dates.month==m)[0]
            # Nawid - Finds the indices (time steps) belonging to that season.
            '''
            seasonal_idxs = np.where(np.bitwise_and(dates.month>=int(seasons[seas][0]), dates.month<=int(seasons[seas][-1])))[0]
            '''
            seasonal_idxs = np.where(
            np.bitwise_and(
            dates.dt.month >= int(seasons[seas][0]),
            dates.dt.month <= int(seasons[seas][-1])))[0]
            #Nawid - Bin MSE values into 2° spatial cells:
            print(
            "Season:", seas,
            " | len(mses):", len(all_mses[seasonal_idxs]),
            " | len(lats):", len(release_lats[seasonal_idxs]),
            " | len(lons):", len(release_lons[seasonal_idxs]),
        )
            mses_binned, lat_edges, lon_edges = bin_data(all_mses[seasonal_idxs], release_lats[seasonal_idxs], release_lons[seasonal_idxs], degree_bins=degree_bins)
            '''
            corrcoeffs_binned, _, _ = bin_data(all_corrcoeffs[seasonal_idxs], release_lats[seasonal_idxs], release_lons[seasonal_idxs], degree_bins=degree_bins)
            '''
            #logged_corrcoeffs_binned, _, _ = bin_data(all_logged_corrcoeffs[seasonal_idxs], release_lats[seasonal_idxs], release_lons[seasonal_idxs], degree_bins=degree_bins)

            #season_results[seas] = {"lat_edges": np.copy(lat_edges), "lon_edges": np.copy(lon_edges), "mses_binned":np.copy(np.array(mses_binned)), "ious_binned":np.copy(np.array(ious_binned)), "corrcoeffs_binned":np.copy(np.array(corrcoeffs_binned)), "logged_corrcoeffs_binned":np.copy(np.array(logged_corrcoeffs_binned))}
            # Nawid - Save results in a dictionary
            '''
            season_results[seas] = {"lat_edges": np.copy(lat_edges), "lon_edges": np.copy(lon_edges), "mses_binned":np.copy(np.array(mses_binned)), "corrcoeffs_binned":np.copy(np.array(corrcoeffs_binned))}
            '''
            season_results[seas] = {"lat_edges": np.copy(lat_edges), "lon_edges": np.copy(lon_edges), "mses_binned":np.copy(np.array(mses_binned))}

    return season_results



def plot_binned_map(ax, binned_lons, binned_lats, metric, metric_name = "", extent="default", cut_lats=[0,0], title_modifier="", bin=False, domain_lats=None, domain_lons=None, divergent=False, vmin_vmax = None, cmap="metrics", fig=None, cbar=True, cbar_position="bottom", title="top", return_cbar=False):
    '''
    This function plots a 2-D gridded metric (e.g., MSE, correlation, etc.) on a geographical map using Cartopy.
    Each grid cell represents the value of the metric in that spatial bin (lat/lon), using a colormap.
    '''
    # Nawid.- checks domain is provided
    if domain_lats is None:
        print("need domain lats!")
    if domain_lons is None:
        print("need domain lons!")
    # nawid - set map exten/crop domain
    extent = (domain_lons[0], domain_lons[-1], domain_lats[cut_lats[0]], domain_lats[-1-cut_lats[1]])
    ax.set_extent(extent, crs=cartopy.crs.PlateCarree())

    # Nawid.- Select colormap based on metric type
    higher_or_lower = {'MSE':"Lower", "Corr Coeff": "Higher"}

    if cmap == "metrics":
        if metric_name in higher_or_lower.keys():
            if higher_or_lower[metric_name] == "Higher": 
                cmap="autumn"
            if higher_or_lower[metric_name] == "Lower": 
                cmap="autumn_r"
        else:
            cmap="autumn"
    

    if vmin_vmax is None:
        print("repla")
        vmin_vmax = [np.nanmin(metric), np.nanmax(metric)]
        
    
    if divergent:
        norm = TwoSlopeNorm(vmin=vmin_vmax[0], vcenter=0, vmax=vmin_vmax[1])
        cmap="PiYG"
    else:
        norm = Normalize(vmin=vmin_vmax[0], vmax=vmin_vmax[1])
    # Nawid - Plot the metric using pcolormesh, drawing colored gridcells representing the metric
    im = ax.pcolormesh(binned_lons, binned_lats, metric, transform=ccrs.PlateCarree(),cmap=cmap, norm=norm) 
    # Nawid - add title
    if title=="top":    
        ax.set_title(metric_name)
    if title=="left":
        coord = -0.1
        if "\n" in metric_name:
            coord = -0.2
        ax.text(coord, 0.5, metric_name,  
              va="center", ha="center",  
              rotation="vertical", fontsize=15,  
              transform=ax.transAxes, multialignment="center")


    # Nawid - add colobar
    if cbar:
        assert fig is not None, "fig cant be empty if you want a cbar in this axis!"
        
        if metric_name in higher_or_lower.keys():
            cbar_label = f"{metric_name} \n {higher_or_lower[metric_name]} is better"
        else:
            cbar_label = f"{metric_name}" 

        if cbar_position == "bottom":
            cbar = fig.colorbar(im, ax=ax, orientation="horizontal", extend='both', shrink=0.7).set_label(cbar_label)
        if cbar_position == "right":
            cbar_label = "ppb"
            cbar = fig.colorbar(im, ax=ax, orientation="vertical", extend='both', shrink=0.7).set_label(cbar_label)
    # Nawid - add map features
    ax.coastlines()
    ax.add_feature(cartopy.feature.BORDERS,linewidth=1.)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)

    if return_cbar:
        return ax, im
    else:
        return ax
    

def plot_massive_binned_map_mse_auto(season_results, domain_lats, domain_lons, region="SOUTHAMERICA", figsize=None):
    """
    Creates a panel of maps showing spatial MSE for each season (JFM, AMJ, JAS, OND).
    Colormap automatically scales to the min and max of all seasonal MSE values.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    # Only one row needed (MSE)
    rows = 1

    # Set figure size and cut domain based on region
    if region == "SOUTHAMERICA":
        figsize = (17, 5)
        cut_domain = {"lats": (40, 20), "lons": (0, 0)}
    elif region == "NORTHAFRICA":
        figsize = (15, 5)
        cut_domain = {"lats": (0, 0), "lons": (0, 0)}

    fig = plt.figure(figsize=figsize, dpi=400, constrained_layout=True)
    gs = fig.add_gridspec(rows, 5, figure=fig, width_ratios=[1, 1, 1, 1, 0.05])

    # Determine global vmin and vmax across all seasons
    all_mses = np.concatenate([season_results[seas]["mses_binned"].ravel() for seas in season_results])
    vmin, vmax = np.nanmin(all_mses), np.nanmax(all_mses)
    print(f"Automatic MSE color scale: vmin={vmin:.2e}, vmax={vmax:.2e}")

    # Create subplot array
    ax = np.empty((rows, 4), dtype=object)
    seasons = {"JFM": ["01", "02", "03"],
               "AMJ": ["04", "05", "06"],
               "JAS": ["07", "08", "09"],
               "OND": ["10", "11", "12"]}

    for i in range(rows):
        for j in range(4):
            ax[i, j] = fig.add_subplot(gs[i, j], projection=ccrs.PlateCarree())

    # Plot MSE for each season
    for season_n, seas in enumerate(seasons.keys()):
        ax[0, season_n], cbar_mses = plot_binned_map(
            ax[0, season_n],
            season_results[seas]["lon_edges"],
            season_results[seas]["lat_edges"],
            season_results[seas]["mses_binned"],
            cut_lats=cut_domain["lats"],
            cmap="metrics",
            metric_name="MSE",
            vmin_vmax=[vmin, vmax],  # automatic scale
            cbar=False,
            cbar_position="right",
            title=None,
            return_cbar=True,
            domain_lats=domain_lats,
            domain_lons=domain_lons
        )

        ax[0, season_n].set_title(seas, fontsize=15)

    # Add single colorbar for MSE
    gs_cb = gs[0, -1].subgridspec(3, 1, height_ratios=[1, 50, 1])
    cbar_ax = fig.add_subplot(gs_cb[1, 0])
    plt.colorbar(cbar_mses, cax=cbar_ax, location='right', extend="both").set_label(label='MSE', size=12)
    ax[0, 0].text(-0.1, 0.5, "MSE",
                  va="center", ha="center",
                  rotation="vertical", fontsize=14,
                  transform=ax[0, 0].transAxes, multialignment="center")

    fig.suptitle("Spatial MSE by season", fontsize=16)
    gs.update(top=0.95)

def plot_massive_binned_map_mse_only(season_results, domain_lats, domain_lons, vmin_vmax=None, region="SOUTHAMERICA", figsize=None):
    """
    Creates a panel of maps showing spatial MSE for each season (JFM, AMJ, JAS, OND).
    Each column = season, each row = metric (here only MSE).
    """
    # Only one row needed (MSE)
    rows = 1

    # Set figure size and cut domain based on region
    if region == "SOUTHAMERICA":
        figsize = (17, 5)
        cut_domain = {"lats": (40, 20), "lons": (0, 0)}
    elif region == "NORTHAFRICA":
        figsize = (15, 5)
        cut_domain = {"lats": (0, 0), "lons": (0, 0)}

    fig = plt.figure(figsize=figsize, dpi=400, constrained_layout=True)
    gs = fig.add_gridspec(rows, 5, figure=fig, width_ratios=[1, 1, 1, 1, 0.05])

    # Default MSE color range
    vmin_vmax_default = {"mses_binned": [0, 1e-6]}
    if vmin_vmax is not None:
        vmin_vmax_default.update(vmin_vmax)
    vmin_vmax = vmin_vmax_default

    print("vmin_vmax:", vmin_vmax)

    # Create subplots array
    ax = np.empty((rows, 4), dtype=object)
    seasons = {"JFM": ["01", "02", "03"],
               "AMJ": ["04", "05", "06"],
               "JAS": ["07", "08", "09"],
               "OND": ["10", "11", "12"]}

    for i in range(rows):
        for j in range(4):
            ax[i, j] = fig.add_subplot(gs[i, j], projection=ccrs.PlateCarree())

    # Plot MSE for each season
    for season_n, seas in enumerate(seasons.keys()):
        ax[0, season_n], cbar_mses = plot_binned_map(
            ax[0, season_n],
            season_results[seas]["lon_edges"],
            season_results[seas]["lat_edges"],
            season_results[seas]["mses_binned"],
            cut_lats=cut_domain["lats"],
            cmap="metrics",
            metric_name="MSE",
            vmin_vmax=vmin_vmax["mses_binned"],
            cbar=False,
            cbar_position="right",
            title=None,
            return_cbar=True,
            domain_lats=domain_lats,
            domain_lons=domain_lons
        )

        ax[0, season_n].set_title(seas, fontsize=15)

    # Add single colorbar for MSE
    gs_cb = gs[0, -1].subgridspec(3, 1, height_ratios=[1, 50, 1])
    cbar_ax = fig.add_subplot(gs_cb[1, 0])
    plt.colorbar(cbar_mses, cax=cbar_ax, location='right', extend="both").set_label(label='MSE', size=12)
    ax[0, 0].text(-0.1, 0.5, "MSE",
                  va="center", ha="center",
                  rotation="vertical", fontsize=14,
                  transform=ax[0, 0].transAxes, multialignment="center")

    fig.suptitle("Spatial MSE by season", fontsize=16)
    gs.update(top=0.95)

def plot_massive_binned_map_density_only(season_results, domain_lats, domain_lons,
                                         vmin_vmax=None, region="SOUTHAMERICA", figsize=None):
    """
    Plot seasonal spatial *density* maps (JFM, AMJ, JAS, OND).
    Each column = a season.
    One row only (density).
    """

    rows = 1   # only density

    # Region-dependent cropping
    if region == "SOUTHAMERICA":
        figsize = (17, 5)
        cut_domain = {"lats": (40, 20), "lons": (0, 0)}
    elif region == "NORTHAFRICA":
        figsize = (15, 5)
        cut_domain = {"lats": (0, 0), "lons": (0, 0)}

    fig = plt.figure(figsize=figsize, dpi=400, constrained_layout=True)
    gs = fig.add_gridspec(rows, 5, figure=fig, width_ratios=[1, 1, 1, 1, 0.05])

    # Default range for density
    vmin_vmax_default = {"density_binned": [0, np.nanmax(
        np.array([season_results[s]["density_binned"] for s in season_results])
    )]}
    if vmin_vmax is not None:
        vmin_vmax_default.update(vmin_vmax)
    vmin_vmax = vmin_vmax_default

    # Season order
    seasons = ["JFM", "AMJ", "JAS", "OND"]

    # Allocate axes
    ax = np.empty((rows, 4), dtype=object)
    for i in range(rows):
        for j in range(4):
            ax[i, j] = fig.add_subplot(gs[i, j], projection=ccrs.PlateCarree())

    # Plot seasonal density
    for j, seas in enumerate(seasons):

        ax[0, j], cbar_density = plot_binned_map(
            ax[0, j],
            season_results[seas]["lon_edges"],
            season_results[seas]["lat_edges"],
            season_results[seas]["density_binned"],
            cut_lats=cut_domain["lats"],
            cmap="viridis",
            metric_name="Density",
            vmin_vmax=vmin_vmax["density_binned"],
            cbar=False,
            cbar_position="right",
            title=None,
            return_cbar=True,
            domain_lats=domain_lats,
            domain_lons=domain_lons
        )
        ax[0, j].set_title(seas, fontsize=15)

    # Shared colorbar
    gs_cb = gs[0, -1].subgridspec(3, 1, height_ratios=[1, 50, 1])
    cbar_ax = fig.add_subplot(gs_cb[1, 0])
    plt.colorbar(cbar_density, cax=cbar_ax, location='right') \
        .set_label(label='Point Density', size=12)

    fig.suptitle("Spatial Point Density by Season", fontsize=16)
    gs.update(top=0.95)

def plot_massive_binned_map(season_results, domain_lats, domain_lons, vmin_vmax = None, region="SOUTHAMERICA", figsize=None):
    '''
    It creates a large 4×4 panel figure of maps showing spatial performance metrics (MSE, CorrCoeff) for each season (JFM, AMJ, JAS, OND).
    '''
    # Nawid - Each column is a seaon and each row is a etric
    rows = 2
    if region == "SOUTHAMERICA":
        figsize= (17, 15)
        cut_domain = {"lats":(40,20), "lons":(0,0)}
    elif region=="NORTHAFRICA":
        figsize= (15, 10)
        cut_domain = {"lats":(0,0), "lons":(0,0)}
    # Nawid- . Setup the figure and subplot grid
    fig = plt.figure(figsize=figsize,  dpi=400, constrained_layout=True)
    # Nawid - Creates a 4×5 grid layout (last column is reserved for colorbars)
    gs = fig.add_gridspec(rows, 5, figure=fig, width_ratios=[1, 1,11,0.05])

    # Nawid - defines metric value range
    vmin_vmax_default = {"mses_binned":[0,1e-6], "corrcoeffs_binned":[0.4,0.7]}

    if vmin_vmax is not None:
        vmin_vmax_default.update(vmin_vmax)


    vmin_vmax = vmin_vmax_default
    print(vmin_vmax)



    ax = np.empty((rows, 4), dtype=object)
    seasons = {"JFM":["01", "02", "03"], "AMJ":["04", "05", "06"], "JAS":["07", "08", "09"], "OND":["10", "11", "12"]}


    for i in range(rows):  
        for j in range(4):
            ax[i, j] = fig.add_subplot(gs[i, j], projection=ccrs.PlateCarree())    
    # Nawid - loo throught he different regions
    for season_n, seas in enumerate(seasons.keys()):
        #season_results[seas] = {"mean_mfs":mean_mfs, "mean_mfs_emulated":mean_mfs_emulated}
        # Nawid - For each season Extract binned metric arrays from season_results Plot map for each metric using plot_binned_map()
        ax[0, season_n], cbar_mses = plot_binned_map(ax[0, season_n], season_results[seas]["lon_edges"], season_results[seas]["lat_edges"], season_results[seas]["mses_binned"], cut_lats=cut_domain["lats"], cmap="metrics", metric_name="MSE", vmin_vmax=vmin_vmax["mses_binned"], cbar=False, cbar_position="right", title=None, return_cbar=True, domain_lats=domain_lats, domain_lons=domain_lons)

        ax[1, season_n], cbar_corrcoeff = plot_binned_map(ax[2, season_n], season_results[seas]["lon_edges"], season_results[seas]["lat_edges"], season_results[seas]["corrcoeffs_binned"], cut_lats=cut_domain["lats"], cmap="metrics", metric_name="Corr Coeff", vmin_vmax=vmin_vmax["corrcoeffs_binned"], cbar=False, cbar_position="right", title=None, return_cbar=True, domain_lats=domain_lats, domain_lons=domain_lons)

        ax[0,season_n].set_title(seas, fontsize=15) 
    # Nawid - Assign titles for each column (season label)
    metric_names = [ "MSE", 'Corr Coeff']
    for ax_num, (cbar_here, metric_name) in enumerate(zip([cbar_mses, cbar_corrcoeff], metric_names)):
        gs_cb = gs[ax_num, -1].subgridspec(3, 1, height_ratios=[1, 50, 1])  
        cbar_ax = fig.add_subplot(gs_cb[1, 0])  
        
        cbar = plt.colorbar(cbar_here, cax=cbar_ax, location='right', extend="both").set_label(label=f'{metric_name}', size=12)  

        ax[ax_num, 0].text(-0.1, 0.5, metric_name,  
                va="center", ha="center",  
                rotation="vertical", fontsize=14,  
                transform=ax[ax_num, 0].transAxes, multialignment="center")


    #plt.tight_layout()
    fig.suptitle("Metrics in space: Red means worse performance", fontsize=16)

    #gs.tight_layout(fig)
    gs.update(top=0.95)


    
# import gridspec
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates

def plot_new_period(model, start_date, min_length=30, print_stats=True, days_to_plot=7, ylim=35):

    #. Nawid - define the time window
    end_date = pd.to_datetime(start_date) + pd.DateOffset(days=days_to_plot)

    fp_selected = model.all_fps.sel(time=slice(start_date, end_date))
    # Nawid - Convert times to pandas datetime
    fp_selected['time'] = pd.to_datetime(fp_selected['time'].values)
    # Nawid - Group data by day
    grouped = fp_selected.groupby('time.date')

    # Nawid -Filter valid days
    gap_threshold = pd.Timedelta('30min')


    fig = plt.figure(figsize=(20, 5), dpi=300)

    valid_dates = []
    ignore_dates = ["2018-10-20"]
    
    for n, (date, group) in enumerate(grouped):
        times = group['time'].values
        if len(times)>min_length and date not in ignore_dates:
            valid_dates.append(date)
        else:
            print(f"removing date {date}")

    print(len(valid_dates))
    outer_gs = gridspec.GridSpec(1,len(valid_dates), figure=fig, wspace=0.15)

    first_axis = True
    legend=False

    name_col = "#768732"
    em_col = "#D79706"
    name_col = "orange"
    em_col = "green"
    
    n=0
    #for n, (date, group) in enumerate(grouped):
    # Nawid = loop pver each valid day
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

                if end - start > 4 and np.mean(group.isel(time=slice(start, end)).true_flux.values/1e-9)>2:
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
                if n_chunks==1:
                    mini_ax = fig.add_subplot(inner_gs[0])
                else:
                    mini_ax = fig.add_subplot(inner_gs[0, i])
                # Nawid - Plot the fluxes
                mini_ax.plot(chunk["time"], chunk.true_flux.values/1e-9, c=name_col, label="with NAME footprints", lw=2)
                mini_ax.plot(chunk["time"], chunk.pred_flux.values/1e-9, c=em_col, label="with GATES footprints", lw=2)
            
                mini_ax.set_ylim(0,ylim)
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
                tick_interval = np.timedelta64(5, 'm')  # 10-minute interval
                tick_times = np.arange(time_range[0], time_range[-1], tick_interval)

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



    fig.suptitle(f"Modelled above-baseline methane concentration - {start_date[:4]}",fontsize=20)
    
    #plt.yticks(fontsize=14)
    #plt.xticks(fontsize=14)

    

    #if print_stats:
    #    plt.text(292, 48, f"Correlation Coefficient: {np.corrcoef(fp_selected.true_flux.values/1e-9, fp_selected.pred_flux.values/1e-9)[0][1]:.2f} \nMean Absolute Error: {np.mean(abs(fp_selected.true_flux.values/1e-9 - fp_selected.pred_flux.values/1e-9)):.2f} ppb", fontsize=14)


    plt.show()


def plot_seasonal_density_histogram(ds, degree_bins=2, vmin_vmax=None):
    """
    Plot density histograms of release locations, as four panels in a row, one for each season.
    Parameters:
    ds: xarray.Dataset containing 'release_lon', 'release_lat', 'time', 'lat' and 'lon' variables
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


    for season_n, seas in enumerate(seasons.keys()):
        #season_results[seas] = {"mean_mfs":mean_mfs, "mean_mfs_emulated":mean_mfs_emulated}
        seasonal_idxs = np.where(np.bitwise_and(dates.month>=int(seasons[seas][0]), dates.month<=int(seasons[seas][-1])))[0]
        print(season_n, seas)

        # extract 2D-bins
        binned_fps,  lat_edges, lon_edges, binnumber = binned_statistic_2d(
                release_lats[seasonal_idxs],
                release_lons[seasonal_idxs],
                None,
                bins=[lat_bins, lon_bins],
                statistic='count',
                expand_binnumbers=True
                )
        
        binned_fps[binned_fps==0] = np.nan

        cbar = True if season_n == 3 else False

        ax[0, season_n], cbar_hist = plot_binned_map(ax[0, season_n], lon_edges, lat_edges, binned_fps, cut_lats=(40,20), cmap="Greens", metric_name="Observation count", vmin_vmax=vmin_vmax, fig=fig, cbar=cbar, cbar_position="right", title_str=seas, return_cbar=True, domain_lats=dom_lats, domain_lons=dom_lons)



import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs

def plot_seasonal_truth_pred_mse(season_results, ground_truth_results, prediction_results, domain_lats, domain_lons, region="SOUTHAMERICA", figsize=None):
    """
    Plot 3 rows per season:
        Row 1: Ground truth
        Row 2: Predictions
        Row 3: MSE
    Columns = seasons (JFM, AMJ, JAS, OND)
    
    ground_truth_results and prediction_results are dictionaries structured like season_results,
    containing the binned values.
    """
    seasons = ["JFM", "AMJ", "JAS", "OND"]
    rows = 3
    cols = 4
    
    if region == "SOUTHAMERICA":
        figsize = (17, 12)
        cut_domain = {"lats": (40, 20), "lons": (0, 0)}
    elif region == "NORTHAFRICA":
        figsize = (15, 10)
        cut_domain = {"lats": (0, 0), "lons": (0, 0)}
    
    fig = plt.figure(figsize=figsize, dpi=400, constrained_layout=True)
    gs = fig.add_gridspec(rows, cols + 1, width_ratios=[1]*cols + [0.05])
    
    ax = np.empty((rows, cols), dtype=object)
    
    # Create subplots for each row and column
    for i in range(rows):
        for j in range(cols):
            ax[i, j] = fig.add_subplot(gs[i, j], projection=ccrs.PlateCarree())
    
    # Determine global vmin and vmax for MSE for consistent colormap
    all_mses = np.concatenate([season_results[seas]["mses_binned"].ravel() for seas in seasons])
    mse_vmin, mse_vmax = np.nanmin(all_mses), np.nanmax(all_mses)
    
    for season_idx, seas in enumerate(seasons):
        # Row 0: Ground truth
        plot_binned_map(
            ax[0, season_idx],
            season_results[seas]["lon_edges"],
            season_results[seas]["lat_edges"],
            ground_truth_results[seas],
            cut_lats=cut_domain["lats"],
            metric_name="Ground Truth",
            domain_lats=domain_lats,
            domain_lons=domain_lons,
            cbar=False
        )
        if season_idx == 0:
            ax[0, season_idx].text(-0.1, 0.5, "Ground Truth", rotation="vertical",
                                   va="center", ha="center", fontsize=14,
                                   transform=ax[0, season_idx].transAxes)
        
        # Row 1: Prediction
        plot_binned_map(
            ax[1, season_idx],
            season_results[seas]["lon_edges"],
            season_results[seas]["lat_edges"],
            prediction_results[seas],
            cut_lats=cut_domain["lats"],
            metric_name="Prediction",
            domain_lats=domain_lats,
            domain_lons=domain_lons,
            cbar=False
        )
        if season_idx == 0:
            ax[1, season_idx].text(-0.1, 0.5, "Prediction", rotation="vertical",
                                   va="center", ha="center", fontsize=14,
                                   transform=ax[1, season_idx].transAxes)
        
        # Row 2: MSE
        _, cbar_mse = plot_binned_map(
            ax[2, season_idx],
            season_results[seas]["lon_edges"],
            season_results[seas]["lat_edges"],
            season_results[seas]["mses_binned"],
            cut_lats=cut_domain["lats"],
            metric_name="MSE",
            vmin_vmax=[mse_vmin, mse_vmax],
            domain_lats=domain_lats,
            domain_lons=domain_lons,
            return_cbar=True,
            cbar=False
        )
        if season_idx == 0:
            ax[2, season_idx].text(-0.1, 0.5, "MSE", rotation="vertical",
                                   va="center", ha="center", fontsize=14,
                                   transform=ax[2, season_idx].transAxes)
        
        # Add season title at top row
        ax[0, season_idx].set_title(seas, fontsize=15)
    
    # Add single colorbar for MSE on the right
    gs_cb = gs[2, -1].subgridspec(3, 1, height_ratios=[1, 50, 1])
    cbar_ax = fig.add_subplot(gs_cb[1, 0])
    plt.colorbar(cbar_mse, cax=cbar_ax, location='right', extend="both").set_label(label="MSE", size=12)
    
    fig.suptitle("Ground Truth, Prediction, and MSE by Season", fontsize=16)
    gs.update(top=0.95)



import numpy as np

def compute_binned_truth_pred(true_values, pred_values, dates, release_lats, release_lons, degree_bins=2):
    """
    Compute binned ground truth, predictions, and MSE per season.
    
    Returns:
        season_truths: dict of binned ground truth arrays per season
        season_preds: dict of binned prediction arrays per season
        season_mses: dict of binned MSE arrays per season
        lat_edges, lon_edges: edges of the lat/lon bins
    """
    seasons = {"JFM": ["01","02","03"], "AMJ": ["04","05","06"], 
               "JAS": ["07","08","09"], "OND": ["10","11","12"]}
    
    season_truths = {}
    season_preds = {}
    season_mses = {}
    
    # Flatten spatial dimensions if needed
    true_flat = true_values.reshape(true_values.shape[0], -1)
    pred_flat = pred_values.reshape(pred_values.shape[0], -1)
    # Nawid - used a sqrt to make it from MSE to RMSE
    all_mses = np.sqrt((true_flat - pred_flat) ** 2)
    
    for seas in seasons:
        # Select indices for this season
        seasonal_idxs = np.where(
            np.bitwise_and(dates.dt.month >= int(seasons[seas][0]),
                           dates.dt.month <= int(seasons[seas][-1]))
        )[0]
        
        # Average over spatial dimensions for each time step
        seasonal_truth = np.nanmean(true_flat[seasonal_idxs], axis=1)
        seasonal_pred  = np.nanmean(pred_flat[seasonal_idxs], axis=1)
        seasonal_mse   = np.nanmean(all_mses[seasonal_idxs], axis=1)
        
        # Bin in lat/lon space
        binned_truth, lat_edges, lon_edges = bin_data(seasonal_truth, release_lats[seasonal_idxs], release_lons[seasonal_idxs], degree_bins=degree_bins)
        binned_pred, _, _ = bin_data(seasonal_pred, release_lats[seasonal_idxs], release_lons[seasonal_idxs], degree_bins=degree_bins)
        binned_mse, _, _  = bin_data(seasonal_mse, release_lats[seasonal_idxs], release_lons[seasonal_idxs], degree_bins=degree_bins)
        
        season_truths[seas] = binned_truth
        season_preds[seas]  = binned_pred
        season_mses[seas]   = binned_mse
    
    return season_truths, season_preds, season_mses, lat_edges, lon_edges

import matplotlib as mpl

import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs

def plot_seasonal_truth_pred_mse_shared_cbar(season_results, ground_truth_results, prediction_results, domain_lats, domain_lons, region="SOUTHAMERICA", figsize=None):
    """
    Plot 3 rows per season:
        Row 1: Ground truth
        Row 2: Predictions
        Row 3: MSE

    One shared colorbar for Ground Truth & Prediction rows, and one colorbar for MSE row.
    """
    seasons = ["JFM", "AMJ", "JAS", "OND"]
    rows = 3
    cols = 4
    
    if region == "SOUTHAMERICA":
        figsize = (17, 12)
        cut_domain = {"lats": (40, 20), "lons": (0, 0)}
    elif region == "NORTHAFRICA":
        figsize = (15, 10)
        cut_domain = {"lats": (0, 0), "lons": (0, 0)}
    
    fig = plt.figure(figsize=figsize, dpi=400, constrained_layout=True)
    gs = fig.add_gridspec(rows, cols + 1, width_ratios=[1]*cols + [0.05])
    
    ax = np.empty((rows, cols), dtype=object)
    for i in range(rows):
        for j in range(cols):
            ax[i, j] = fig.add_subplot(gs[i, j], projection=ccrs.PlateCarree())
    
    # Compute shared vmin/vmax for Ground Truth and Predictions
    all_vals = np.concatenate([ground_truth_results[s].ravel() for s in seasons] +
                              [prediction_results[s].ravel() for s in seasons])
    vmin_vmax_shared = [np.nanmin(all_vals), np.nanmax(all_vals)]
    
    # Compute vmin/vmax for MSE
    all_mses = np.concatenate([season_results[s]["mses_binned"].ravel() for s in seasons])
    vmin_vmax_mse = [np.nanmin(all_mses), np.nanmax(all_mses)]
    
    for season_idx, seas in enumerate(seasons):
        # Row 0: Ground Truth
        plot_binned_map(
            ax[0, season_idx],
            season_results[seas]["lon_edges"],
            season_results[seas]["lat_edges"],
            ground_truth_results[seas],
            cut_lats=cut_domain["lats"],
            metric_name="Ground Truth",
            domain_lats=domain_lats,
            domain_lons=domain_lons,
            vmin_vmax=vmin_vmax_shared,
            cmap="viridis",
            cbar=False
        )
        if season_idx == 0:
            ax[0, season_idx].text(-0.1, 0.5, "Ground Truth", rotation="vertical",
                                   va="center", ha="center", fontsize=14,
                                   transform=ax[0, season_idx].transAxes)
        
        # Row 1: Predictions
        plot_binned_map(
            ax[1, season_idx],
            season_results[seas]["lon_edges"],
            season_results[seas]["lat_edges"],
            prediction_results[seas],
            cut_lats=cut_domain["lats"],
            metric_name="", # Nawid = giving an empy name this is the name of the inidivual subplot
            domain_lats=domain_lats,
            domain_lons=domain_lons,
            vmin_vmax=vmin_vmax_shared,
            cmap="viridis",
            cbar=False
        )
        if season_idx == 0:
            ax[1, season_idx].text(-0.1, 0.5, "Prediction", rotation="vertical",
                                   va="center", ha="center", fontsize=14,
                                   transform=ax[1, season_idx].transAxes)
        
        # Row 2: MSE
        plot_binned_map(
            ax[2, season_idx],
            season_results[seas]["lon_edges"],
            season_results[seas]["lat_edges"],
            season_results[seas]["mses_binned"],
            cut_lats=cut_domain["lats"],
            metric_name="", # Nawid - giving an empty name since this is the name of the individual subplot
            domain_lats=domain_lats,
            domain_lons=domain_lons,
            vmin_vmax=vmin_vmax_mse,
            cmap="Reds",
            cbar=False
        )
        if season_idx == 0:
            ax[2, season_idx].text(-0.1, 0.5, "RMSE", rotation="vertical",
                                   va="center", ha="center", fontsize=14,
                                   transform=ax[2, season_idx].transAxes)
        
        ax[0, season_idx].set_title(seas, fontsize=15)
    
    # Shared colorbar for Ground Truth & Prediction
    gs_cb_truth = gs[0, -1].subgridspec(2, 1, height_ratios=[50, 1])
    cbar_ax_truth = fig.add_subplot(gs_cb_truth[0, 0])
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(*vmin_vmax_shared))
    sm.set_array([])
    plt.colorbar(sm, cax=cbar_ax_truth, orientation='vertical').set_label(label="Ground Truth / Prediction", size=12)
    
    # Colorbar for MSE
    gs_cb_mse = gs[2, -1].subgridspec(2, 1, height_ratios=[50, 1])
    cbar_ax_mse = fig.add_subplot(gs_cb_mse[0, 0])
    sm_mse = plt.cm.ScalarMappable(cmap="Reds", norm=plt.Normalize(*vmin_vmax_mse))
    sm_mse.set_array([])
    plt.colorbar(sm_mse, cax=cbar_ax_mse, orientation='vertical').set_label(label="RMSE", size=12)
    
    fig.suptitle("Ground Truth, Prediction, and RMSE by Season", fontsize=16)
    gs.update(top=0.95)


# NAWID - CALCULATING THE DENSITY SPATIALLY 
import numpy as np
