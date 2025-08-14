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

from graphnet_LPDM_emulator.model.data.load_data import *
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


class FPModel():
    """ 
    """
    def __init__(self, year, pred_fp_path, months=None,  region="BRAZIL", true_fp_path="NAME_fps_intersect", size=200, remove_val=True, return_fullfiles=False, fp_folder="satellite_emulated_logv4"):
        # remove_val: remove validation set 
        
        self.year = year
        self.pred_fp_path = pred_fp_path
        self.true_fp_path = true_fp_path
        self.size = size
        self.emissions_loaded = False
        self.fluxes_loaded = False

        self.fp_folder = fp_folder
        self.true_fp_path=true_fp_path

        self.region=region

        if region=="BRAZIL":
            self.domain="SOUTHAMERICA"
        elif region=="SAHARA":
            self.domain="NORTHAFRICA"
        else:
            print("adding a different domain doesnt work yet!")

        print("loading footprints")
        print(f"/group/chemistry/acrg/LPDM/fp_Elena/{fp_folder}/{true_fp_path}/{self.domain}/GOSAT-{self.region}-column_{self.domain}_{year}*.nc")
        if months is None:
            true_fp = load_fps(f"/group/chemistry/acrg/LPDM/fp_Elena/{fp_folder}/{true_fp_path}/{self.domain}/GOSAT-{self.region}-column_{self.domain}_{year}*.nc")
            #true_fp = xr.open_mfdataset(glob.glob(f"/group/chemistry/acrg/LPDM/fp_Elena/{fp_folder}/{true_fp_path}/{self.domain}/GOSAT-{self.region}-column_{self.domain}_{year}*.nc"))
            pred_fp = load_fps(f"/group/chemistry/acrg/LPDM/fp_Elena/{fp_folder}/{pred_fp_path}/{self.domain}/GOSAT-{self.region}-column_{self.domain}_{year}*.nc")
            #pred_fp = xr.open_mfdataset(glob.glob(f"/group/chemistry/acrg/LPDM/fp_Elena/{fp_folder}/{pred_fp_path}/{self.domain}/GOSAT-{self.region}-column_{self.domain}_{year}*.nc"))
        else:
            true_files = [x for x in glob.glob(f"/group/chemistry/acrg/LPDM/fp_Elena/{fp_folder}/{true_fp_path}/{self.domain}/GOSAT-{self.region}-column_{self.domain}_{year}{"*"}.nc") if np.any([mo in x for mo in months])]
            true_fp = xr.open_mfdataset(true_files)
            em_files = [x for x in glob.glob(f"/group/chemistry/acrg/LPDM/fp_Elena/{fp_folder}/{pred_fp_path}/{self.domain}/GOSAT-{self.region}-column_{self.domain}_{year}{"*"}.nc") if np.any([mo in x for mo in months])]
            pred_fp = xr.open_mfdataset(em_files)


        assert len(pred_fp) == len(true_fp), "these two files arent aligned \n this doesnt work yet"
        
        if return_fullfiles:
            return pred_fp, true_fp


        print(f"prepping predicted data ({len(pred_fp.time)} footprints)")


        pred_fp = pred_fp[["fp", "release_lon", "release_lat"]]
        pred_fp.load()
        pred_fp = cut_satellite_data(pred_fp, size=size, fill_bads_with="nans", return_as="netcdf")
        
        print("prepping true data")
        true_fp = true_fp[["fp", "release_lon", "release_lat"]]
        true_fp.load()
        true_fp = cut_satellite_data(true_fp, size=size, fill_bads_with="nans", return_as="netcdf")

        self.all_fps = xr.merge([true_fp.rename({"fp":"fp_true"}), pred_fp.rename({"fp":"fp_pred"})])
        
        self.all_fps = self.all_fps.rename({"lat_coords":"release_lat", "lon_coords":"release_lon"})

        if remove_val:
            self.remove_val_set()

    def remove_val_set(self):
        print("removing val set")
        self.all_fps=self.all_fps.sel(time=slice("2016-04-01", "2025-03-31"))

    def plot_fp(self, i):
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(np.log10(self.all_fps.fp_true[i]), origin="lower")
        ax[0].set_title("true fp")
        ax[1].imshow(np.log10(self.all_fps.fp_pred[i]), origin="lower")
        ax[1].set_title("pred fp")

        plt.show()

    
    def load_emissions(self, same_month=True):
        true_fp = xr.open_mfdataset(glob.glob(f"/group/chemistry/acrg/LPDM/fp_Elena/{self.fp_folder}/{self.true_fp_path}/{self.domain}/GOSAT-{self.region}-column_{self.domain}_{self.year}*.nc"))
        shared_times = np.intersect1d(true_fp.time.values, self.all_fps.time)
        true_fp = true_fp.sel(time=shared_times)
        self.all_fps = self.all_fps.sel(time=shared_times)

        if same_month:
            print("loading the same monthly prior for the whole dataset")
            if self.region=="BRAZIL":
                flux = load_default_brazil_emissions() 
            if self.region=="SAHARA":
                flux = load_default_sahara_emissions()

            emissions = cut_emissions_data(flux, true_fp, self.size)
            self.emissions = np.transpose(emissions, [2, 0,1])

        else:
            c = 0
            print("loading the monthly priors (separately)")
            monthly_ems = np.zeros_like(self.all_fps.fp_true.values)
            for y in np.unique(pd.DatetimeIndex(true_fp.time).year):
                for m in np.unique(pd.DatetimeIndex(true_fp.time).month):
                    where_month = np.where(np.bitwise_and(pd.DatetimeIndex(true_fp.time).month ==m, pd.DatetimeIndex(true_fp.time).year ==y))[0]
                    flux = load_default_brazil_emissions(month_to_use=m)
                    emissions = cut_emissions_data(flux, true_fp.sel(time=true_fp.time.values[where_month]), self.size) 
                    emissions = np.transpose(emissions, [2, 0,1])
                    monthly_ems[c:c+len(where_month)] = emissions
                    c = c+len(where_month)

            self.emissions=monthly_ems

        self.emissions_loaded = True

    def get_fluxes(self):
        if not self.emissions_loaded:
            self.load_emissions()

        true_flux, pred_flux = predict_fluxes(np.nan_to_num(self.all_fps.fp_true.values), np.nan_to_num(self.all_fps.fp_pred.values), self.emissions, units_transform=None)

        self.all_fps['true_flux'] = (('time'), true_flux)
        self.all_fps['pred_flux'] = (('time'), pred_flux)
        #self.all_fps
        self.fluxes_loaded = True


def predict_fluxes(true_fp, pred_fp, flux, units_transform = None):
    ## convolute predicted footprints and fluxes, returns two np arrays, one with the true flux and one with the emulated flux, of shape (n_footprints,)
    ## flux is an array, regridded and cut to the same resolution and size of the footprints
    ## units_transform can be None (use fluxes directly), "default" (performs flux*1e3 / CH4molarmass) or another function (which should return an array of the same shape as the original flux)

    if units_transform != None:
        if units_transform == "default":
            molarmass = 16.0425
            flux = flux*1e3 / molarmass
        else:
            flux = units_transform(flux)
    


    true_concentration = true_fp*flux
    true_flux = np.sum(true_concentration, axis = (1,2))
    pred_concentration = pred_fp*flux
    pred_flux = np.sum(pred_concentration, axis = (1,2))
    
    return true_flux, pred_flux


def IoU_here(fp, pred,threshold=0):
    fps_bin = np.copy(fp)>threshold
    preds_bin = np.copy(pred)>threshold
    intersection = np.sum(np.logical_and(fps_bin==1, preds_bin==1, where=1))
    #print(np.shape(fps))
    #print(intersection)
    union = np.sum(np.logical_or(fps_bin==1, preds_bin==1, where=1))
    IoU = intersection/union

    
    return IoU

def IoU(fp, pred, threshold=0,return_mean=True):
    fps_bin = np.copy(fp)>threshold
    preds_bin = np.copy(pred)>threshold
    intersection = np.sum(np.logical_and(fps_bin==1, preds_bin==1, where=1), axis=(1,2))
    #print(np.shape(fps))
    #print(intersection)
    union = np.sum(np.logical_or(fps_bin==1, preds_bin==1, where=1), axis=(1,2))
    IoU = intersection/union
    if return_mean:
        return np.mean(IoU)
    else:
        return IoU

def MAE(true, pred):
    pred= np.copy(pred.flatten())
    true= np.copy(true.flatten())
    #mae = np.mean(np.abs(true-pred))
    mask = (true != 0) & (pred != 0) & ~np.isnan(true) & ~np.isnan(pred) & ~np.isinf(true) & ~np.isinf(pred)
    mae = np.mean(np.abs(true[mask]-pred[mask]))
    return mae 


def get_stats(mod, i, do_print=True, do_return=False):
    iou = IoU_here(mod.all_fps.fp_true[i], mod.all_fps.fp_pred[i])
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
        "IoU": iou,
        "MSE": mse,
        "Corrcoef": corrcoef,
        "log MSE": log_mse,
        "log Corrcoef": log_corrcoef
    }
    
    if do_print:
        print(i)
        print(f"IoU: {iou}, MSE: {mse}, Corrcoef: {corrcoef}, log MSE: {log_mse}, log Corrcoef: {log_corrcoef}")

    if do_return:
        return stats_dict


def plot_four_fps(model, idx_1, idx_2, idx_3, idx_4, fig_title=None, plot_hist=False, plot_timeseries=False, plot_cities=False, plot_fluxes=False, print_stats=False, contour=False, thres=0, ylim_timeseries=45):
    """

    """
    plot_prior=False


    # altnerative fig creating for more flexibility
    n_rows = 2 + 2*plot_hist + plot_timeseries + 2*plot_fluxes + print_stats
    height_ratios = [1] * n_rows  # Default all equal
    if print_stats:
        height_ratios[2] = 0
    print(height_ratios)
    
    fig = plt.figure(figsize=(16,6+3*(n_rows-2-print_stats)), dpi=400)


    

    gs = fig.add_gridspec(n_rows, 5, width_ratios=[1, 1, 1,1,0.1], height_ratios=height_ratios)
    #gs = fig.add_gridspec(n_rows, 4, width_ratios=[1, 1, 1,1], height_ratios=height_ratios)

    ax = np.empty((n_rows, 4), dtype=object)
    import cartopy.crs as ccrs

    # Create geoaxes for the first two rows (LPDM and gates footprints)
    for i in range(2):  
        for j in range(4):
            ax[i, j] = fig.add_subplot(gs[i, j], projection=ccrs.PlateCarree())    

    if print_stats:
        stats_row = 2  # Last row
        #for j in range(4):
        #    ax[stats_row, j] = fig.add_subplot(gs[stats_row, j])
        #    ax[stats_row, j].axis("off")  # Hide stats axes

    if plot_hist:
        hist_row = 2+print_stats  
        for j in range(4):
            ax[hist_row, j] = fig.add_subplot(gs[hist_row, j])
            ax[hist_row+1, j] = fig.add_subplot(gs[hist_row+1, j])

    if plot_fluxes:
        fluxes_row = 2+print_stats+2*plot_hist
        for j in range(4):
            ax[fluxes_row, j] = fig.add_subplot(gs[fluxes_row, j], projection=ccrs.PlateCarree())  
            ax[fluxes_row+1, j] = fig.add_subplot(gs[fluxes_row+1, j], projection=ccrs.PlateCarree())  
            
    if plot_timeseries:
        timeseries_row = 2+print_stats +2*plot_hist + 2*plot_fluxes 
        for j in range(4):
            ax[timeseries_row, j] = fig.add_subplot(gs[timeseries_row, j])
    
    """
    plot_prior = False
    if plot_fluxes and plot_timeseries:
        fig = plt.figure(figsize=(16,12), dpi=400)
        #gs = gridspec.GridSpec(5, 4, figure=fig)#, height_ratios=[1, 1, 1])  # Adjust 
        gs = fig.add_gridspec(5, 5, width_ratios=[1, 1, 1,1,0.1])
        # height ratio if needed

        # Create an array to store the axes
        ax = np.empty((5, 4), dtype=object)

        # Create geoaxes for the first two rows
        for i in range(4):  
            for j in range(4):
                ax[i, j] = fig.add_subplot(gs[i, j], projection=ccrs.PlateCarree())

        # Create standard matplotlib Axes for the third row (time series)
        for j in range(4):
            ax[-1, j] = fig.add_subplot(gs[-1, j])  # No projection for the third row

    elif plot_fluxes:
        fig, ax = plt.subplots(4+plot_prior,4 , figsize=(12,10+2*plot_prior), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=500)
        

    elif plot_hist or plot_timeseries:
        fig = plt.figure(figsize=(16,8), dpi=400)
        gs = fig.add_gridspec(3, 4)#, height_ratios=[1,1,0.5])
        print("here")
        ax = np.array([[fig.add_subplot(gs[0, j], projection=ccrs.PlateCarree()) for j in [0,1,2,3]], [fig.add_subplot(gs[1, j], projection=ccrs.PlateCarree()) for j in [0,1,2,3]], [fig.add_subplot(gs[2, j]) for j in [0,1,2,3]]])

    else:
        fig, ax = plt.subplots(2,4 , figsize=(12,5), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=500)
    """
    
    #contour=True
    levels = np.arange(-5.5, -1.5+0.6, 0.5)
    levels = [-4, -3.5, -3,  -2.5, -2, -1.5]

    vmin, vmax = (-4, -2)
    #vmin, vmax = (-3, -1.5)

    #idx_1 = 150 
    """ 
    if plot_hist or plot_timeseries or plot_fluxes:
        print_stats = False
    else:
        print_stats = True
    """


    plot_footprint_ax(ax[0,0], ax[1,0], model, idx_1, contour=contour, share_minmax=True, cbar=False, nlevels=levels, title=True, return_minmax=False, vmin=vmin, vmax=vmax, print_stats=print_stats, thres=thres,plot_cities=plot_cities)
    get_stats(model, idx_1)

    #idx_2 = 5360
    plot_footprint_ax(ax[0,1], ax[1,1], model, idx_2, contour=contour, share_minmax=True, cbar=False, nlevels=levels, thres=thres, title=True, vmin=vmin, vmax=vmax, print_stats=print_stats,plot_cities=plot_cities)
    get_stats(model, idx_2)


    #idx_3= 154
    plot_footprint_ax(ax[0,2], ax[1,2], model, idx_3, contour=contour, share_minmax=True, cbar=False, nlevels=levels, thres=thres, title=True, vmin=vmin, vmax=vmax, print_stats=print_stats,plot_cities=plot_cities)
    get_stats(model, idx_3)

    #idx_4 = 473
    cb = plot_footprint_ax(ax[0,3], ax[1,3], model, idx_4, contour=contour, share_minmax=True, cbar=True, nlevels=levels, thres=thres, title=True, vmin=vmin, vmax=vmax, print_stats=print_stats,plot_cities=plot_cities)
    get_stats(model, idx_4)
    if plot_hist:
        print("ww")
        #plot_hist_ax(ax[2,3], mod, idx_4)

    if plot_fluxes:
        """
        if plot_prior:
            prior_ax = ax[2,n]
        else:
            prior_ax=None
        """
        for n, idx in enumerate([idx_1, idx_2, idx_3, idx_4]):
            flux_cb = plot_emissions_ax(None, ax[fluxes_row,n], ax[fluxes_row+1,n], model, idx, cbar=True,plot_cities=plot_cities)

    if plot_hist:
        labels = [True, False, False, False]
        for n, idx in enumerate([idx_1, idx_2, idx_3, idx_4]):
            plot_hist_ax(ax[hist_row,n], model, idx, plot_logged=False, ylabels=labels[n])
            plot_hist_ax(ax[hist_row+1,n], model, idx, ylabels=labels[n])

    if plot_timeseries:
        days_before = 1
        days_after=1
        plot_a_day_ax(ax[timeseries_row,0], gs[timeseries_row,0], fig, str(model.all_fps.time[idx_1].values)[:10], model, days_before=days_before, days_after=days_after, dateline_idx=idx_1, ylim=ylim_timeseries, labels=True) 
        for n, idx in enumerate([idx_2, idx_3, idx_4]):
            plot_a_day_ax(ax[timeseries_row,n+1], gs[timeseries_row,n+1], fig, str(model.all_fps.time[idx].values)[:10], model, days_before=days_before, days_after=days_after, dateline_idx=idx, ylim=ylim_timeseries)     


    ax[0, 0].text(-0.1, 0.5, "Footprints generated \n with the LPDM",  
              va="center", ha="center",  
              rotation="vertical", fontsize=14,  
              transform=ax[0, 0].transAxes, multialignment="center")
            
    ax[1, 0].text(-0.1, 0.5, "Footprints emulated \n with GATES",  
              va="center", ha="center",  
              rotation="vertical", fontsize=14,  
              transform=ax[1, 0].transAxes, multialignment="center")

    if plot_fluxes:
        # write axis labels
        ax[fluxes_row, 0].text(-0.1, 0.5, "Prior * \nLPDM footprints",  
              va="center", ha="center",  
              rotation="vertical", fontsize=14,  
              transform=ax[fluxes_row, 0].transAxes, multialignment="center")

        ax[fluxes_row+1, 0].text(-0.1, 0.5, "Prior * \nGATES footprints",  
              va="center", ha="center",  
              rotation="vertical", fontsize=14,  
              transform=ax[fluxes_row+1, 0].transAxes, multialignment="center")

        # colorbar 
        gs_cb_flux = gs[fluxes_row:fluxes_row+2, -1].subgridspec(3, 1, height_ratios=[1, 15, 1])  

        cbar_ax_flux = fig.add_subplot(gs_cb_flux[1, 0])  


        cbar = plt.colorbar(flux_cb, cax=cbar_ax_flux, location='right', extend="both").set_label(label=r'log$_{10}$'+"(contribution to mole fraction \nfrom each cell, in ppb)", size=12)


    gs_cb = gs[:2, -1].subgridspec(3, 1, height_ratios=[1, 15, 1])  

    cbar_ax = fig.add_subplot(gs_cb[1, 0])  

    #cbar_ax = fig.add_subplot(gs[:2, -1])

    cbar = plt.colorbar(cb, cax=cbar_ax, location='right', extend="both").set_label(label=r'log$_{10}$ (mol mol$^{-1}$ (mol m$^{-2}$ s$^{-1}$)$^{-1}$)', size=12)
    plt.tight_layout()

    #ax[0,0].set_title(mod.all_fps.time.values[150])
    if fig_title is not None:
        #fig.suptitle(fig_title)
        fig.text(0.3, 1, fig_title, va="center",  fontsize=21,multialignment="center")

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
            #ax.text(0.05, 0.95, f"IoU: {str(100*stats["IoU"])[:4]}% \nMSE: {str(stats["MSE"])[:4]+str(stats["MSE"])[-4:]} \nLog Corr {str(stats["log Corrcoef"])[:4]}", transform=ax.transAxes, ha="left", va="top", fontsize=12)   
            ax.text(0.05, 0.95, f"Log Corr {str(stats["log Corrcoef"])[:4]}", transform=ax.transAxes, ha="left", va="top", fontsize=12)   
        else:
            ax.text(0.05, 0.95, f"IoU: {str(100*stats["IoU"])[:4]}% \nMSE: {str(stats["MSE"])[:4]+str(stats["MSE"])[-4:]} \nCorr {str(stats["Corrcoef"])[:4]}", transform=ax.transAxes, ha="left", va="top", fontsize=12)   

def plot_footprint_ax(ax_true, ax_pred, model, idx, lon_squeeze=0, vmin=None, vmax=None, thres=0, contour=False, share_minmax=False, cbar=True,nlevels=None, title=False, return_minmax=False, print_stats=False, plot_cities=False):


    np.seterr(divide='ignore')

    extent = (model.all_fps.release_lon[idx,lon_squeeze], model.all_fps.release_lon[idx,-1-lon_squeeze], model.all_fps.release_lat[idx,0], model.all_fps.release_lat[idx,-1])
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
        cb = ax_true.contourf(model.all_fps.release_lon.values[idx], model.all_fps.release_lat.values[idx],np.log10(f), **plot_params, levels=nlevels, extend="both", alpha=alpha)
        f[f<thres] = 0
        cb = ax_true.contourf(model.all_fps.release_lon.values[idx], model.all_fps.release_lat.values[idx],np.log10(f), **plot_params, levels=nlevels, extend="both")


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
        ax_pred.text(0.5, -0.1, f"IoU: {str(100*stats["IoU"])[:4]}% \n MSE: {str(stats["MSE"])[:4]+str(stats["MSE"])[-4:]} \n Corr {str(stats["Corrcoef"])[:4]}\n", transform=ax_pred.transAxes, 
        ha="center", va="top", fontsize=12)    

    if contour:
        cb = ax_pred.contourf(model.all_fps.release_lon.values[idx], model.all_fps.release_lat.values[idx],np.log10(f), **plot_params, levels=nlevels, extend="both",alpha=alpha)

        f[f<thres] = 0

        cb = ax_pred.contourf(model.all_fps.release_lon.values[idx], model.all_fps.release_lat.values[idx],np.log10(f), **plot_params, levels=nlevels, extend="both")
    else:
        cb = ax_pred.imshow(np.log10(f), extent=extent, origin="lower", **plot_params, zorder=5)
    
    if return_minmax:
        return vmin, vmax
    if cbar:
        return cb


def plot_emissions_ax(ax_ems, ax_true, ax_pred, model, idx, lon_squeeze=0, vmin=-13, vmax=-9, thres=0, contour=False, share_minmax=False, cbar=True,nlevels=None, title=False, return_minmax=False, print_stats=False, plot_cities=False):


    np.seterr(divide='ignore')

    extent = (model.all_fps.release_lon[idx,lon_squeeze], model.all_fps.release_lon[idx,-1-lon_squeeze], model.all_fps.release_lat[idx,0], model.all_fps.release_lat[idx,-1])
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
        ax_pred.text(0.5, -0.1, f"IoU: {str(100*stats["IoU"])[:4]}% \n MSE: {str(stats["MSE"])[:4]+str(stats["MSE"])[-4:]} \n Corr {str(stats["Corrcoef"])[:4]}", transform=ax_pred.transAxes, 
        ha="center", va="top", fontsize=12)    

    
    if return_minmax:
        return vmin, vmax
    if cbar:
        return cb
        #cbar = plt.colorbar(cb, ax=[ax_true, ax_pred], location='right', label='Number of data points', extend="both", shrink=0.5)

def get_gosat(site, species, 
              start_date = None, end_date = None, max_level=17,
              data_directory = "/group/chemistry/acrg/obs"):
    """
    FROM ACRG!
    retrieves obervations for a set of sites and species between start and 
    end dates for GOSAT 
    
    Args:    
        site (str) :
            Site of interest. All sites should be defined within acrg_site_info.json. 
            E.g. ["MHD"] for Mace Head site.
        species (str) :
            Species identifier. All species names should be defined within acrg_species_info.json. 
            E.g. "ch4" for methane.
        max_level (int) : 
            Required for satellite data only. Maximum level to extract up to from within satellite data.
        start_date (str, optional) : 
            Start date in Pandas-readable format. E.g. "2000-01-01 00:00"
            Default = None.
        end_date (str, optional) : 
            End date in Pandas-readable format.
            Default = None.
        data_directory (str, optional) :
            directory can be specified if files are not in the default directory. 
            Must point to a directory which contains subfolders organized by site.
            Default=None.
            
    Returns:
        (xarray dataframe):
            xarray data frame for GOSAT observations.
            
    """
    
    if max_level is None:
        raise ValueError("'max_level' ARGUMENT REQUIRED FOR SATELLITE OBS DATA")
    

    gosat_directory = f"{data_directory}/GOSAT/{site}/"

    files = [os.path.basename(f) for f in glob.glob(f"{gosat_directory}*.nc")]

    files_date = [pd.to_datetime(f.split("_")[2][0:8]) for f in files]

    data = []
    for (f, d) in zip(files, files_date):
        if d >= pd.to_datetime(start_date) and d < pd.to_datetime(end_date):
            with xr.open_dataset(f"{gosat_directory}/{f}") as fxr:
                data.append(fxr.load())
    
    if len(data) == 0:
        return []
    
    data = xr.concat(data, dim = "time")

    lower_levels =  list(range(0,max_level))

    prior_factor = (data.pressure_weights[dict(lev=list(lower_levels))]* \
                    (1.-data.xch4_averaging_kernel[dict(lev=list(lower_levels))])* \
                    data.ch4_profile_apriori[dict(lev=list(lower_levels))]).sum(dim = "lev")
                    
    upper_levels = list(range(max_level, len(data.lev.values)))            
    prior_upper_level_factor = (data.pressure_weights[dict(lev=list(upper_levels))]* \
                    data.ch4_profile_apriori[dict(lev=list(upper_levels))]).sum(dim = "lev")
                
    data["mf_prior_factor"] = prior_factor
    data["mf_prior_upper_level_factor"] = prior_upper_level_factor
    data["mf"] = data.xch4 - data.mf_prior_factor - data.mf_prior_upper_level_factor
    data["mf_repeatability"] = data.xch4_uncertainty

    # rt17603: 06/04/2018 Added drop variables to ensure lev and id dimensions are also dropped, Causing problems in footprints_data_merge() function
    drop_data_vars = ["xch4","xch4_uncertainty","ch4_profile_apriori","xch4_averaging_kernel",
                      "pressure_levels","pressure_weights","exposure_id"]
    drop_coords = ["lev","id"]
    
    for dv in drop_data_vars:
        if dv in data.data_vars:
            data = data.drop_vars(dv)
    for coord in drop_coords:
        if coord in data.coords:
            data = data.drop_vars(coord)

    data = data.sortby("time")
    
    data.attrs["max_level"] = max_level
    if species.upper() == "CH4":
        data.mf.attrs["units"] = '1e-9'
        data.attrs["species"] = "CH4"
    if species.upper() == "CO2":
        data.mf.attrs["units"] = '1e-6'
        data.attrs["species"] = "CO2"

    data.attrs["scale"] = "GOSAT"

    # return single element list
    return [data,]


from scipy.stats import binned_statistic_2d
def bin_data(data, lats, lons, degree_bins=1):
    lat_bins = range(int(np.floor(lats.min())), int(np.ceil(lats.max())) + 1, degree_bins) 
    lon_bins = range(int(np.floor(lons.min())), int(np.ceil(lons.max())) + 1, degree_bins) 
    binned_data, lat_edges, lon_edges, binnumber = binned_statistic_2d(
        lats, lons, data, statistic='mean', bins=[lat_bins, lon_bins]
    )
    return binned_data, lat_edges, lon_edges


def vcorrcoef(fp, pred, threshold = None, return_mean=False):
    X = fp.reshape(fp.shape[0], -1)
    y = pred.reshape(pred.shape[0], -1)
    
    if threshold is None:
        mask = (X != 0) & (y != 0) & ~np.isnan(X) & ~np.isnan(y) & ~np.isinf(X) & ~np.isinf(y)
    else:
        mask = (X != 0) & (y != 0) & ~np.isnan(X) & ~np.isnan(y) & ~np.isinf(X) & ~np.isinf(y) & (X > threshold) & (y > threshold)
    X_masked = np.where(mask, X, np.nan)
    y_masked = np.where(mask, y, np.nan)

    
    Xm = np.nanmean(X_masked, axis=1, keepdims=True)
    ym = np.nanmean(y_masked, axis=1)
    r_num = np.nansum((X_masked - Xm) * (y_masked - ym[:, None]), axis=1)
    r_den = np.sqrt(np.nansum((X_masked - Xm) ** 2, axis=1) * np.nansum((y_masked - ym[:, None]) ** 2, axis=1))

    r = r_num / r_den

    return np.nanmean(r) if return_mean else r


def get_binned_seasonal_scores(true_fps, predictions, dates, release_lats, release_lons, degree_bins=2):
    seasons = {"JFM":["01", "02", "03"], "AMJ":["04", "05", "06"], "JAS":["07", "08", "09"], "OND":["10", "11", "12"]}

    all_ious = IoU(true_fps, predictions, threshold=0,return_mean=False)
    all_mses = np.nanmean((true_fps - predictions) ** 2, axis=(1, 2))
    all_corrcoeffs = vcorrcoef(true_fps, predictions, return_mean=False)
    all_logged_corrcoeffs = vcorrcoef(np.log10(true_fps), np.log10(predictions), return_mean=False)

    season_results = {}
    degree_bins = 2
    for n, seas in enumerate(list(seasons.keys())):
            print(n, seas)
            #seasonal_idxs = np.bitwise_and(dates.month==m)[0]
            seasonal_idxs = np.where(np.bitwise_and(dates.month>=int(seasons[seas][0]), dates.month<=int(seasons[seas][-1])))[0]
            
            mses_binned, lat_edges, lon_edges = bin_data(all_mses[seasonal_idxs], release_lats[seasonal_idxs], release_lons[seasonal_idxs], degree_bins=degree_bins)

            ious_binned, _, _= bin_data(all_ious[seasonal_idxs], release_lats[seasonal_idxs], release_lons[seasonal_idxs], degree_bins=degree_bins)

            corrcoeffs_binned, _, _ = bin_data(all_corrcoeffs[seasonal_idxs], release_lats[seasonal_idxs], release_lons[seasonal_idxs], degree_bins=degree_bins)

            logged_corrcoeffs_binned, _, _ = bin_data(all_logged_corrcoeffs[seasonal_idxs], release_lats[seasonal_idxs], release_lons[seasonal_idxs], degree_bins=degree_bins)

            season_results[seas] = {"lat_edges": np.copy(lat_edges), "lon_edges": np.copy(lon_edges), "mses_binned":np.copy(np.array(mses_binned)), "ious_binned":np.copy(np.array(ious_binned)), "corrcoeffs_binned":np.copy(np.array(corrcoeffs_binned)), "logged_corrcoeffs_binned":np.copy(np.array(logged_corrcoeffs_binned))}

    return season_results



def plot_binned_map(ax, binned_lons, binned_lats, metric, metric_name = "", extent="default", cut_lats=[0,0], title_modifier="", bin=False, domain_lats=None, domain_lons=None, divergent=False, vmin_vmax = None, cmap="metrics", fig=None, cbar=True, cbar_position="bottom", title="top", return_cbar=False):
    
    
    if domain_lats is None:
        print("need domain lats!")
    if domain_lons is None:
        print("need domain lons!")

    extent = (domain_lons[0], domain_lons[-1], domain_lats[cut_lats[0]], domain_lats[-1-cut_lats[1]])
    ax.set_extent(extent, crs=ccrs.PlateCarree())


    higher_or_lower = {"IoU":"Higher", 'MSE':"Lower", "Corr Coeff": "Higher", "Log CorrCoeff": "Higher"}

    if cmap == "metrics":
        if metric_name in higher_or_lower.keys():
            if higher_or_lower[metric_name] == "Higher": 
                cmap="autumn"
            if higher_or_lower[metric_name] == "Lower": 
                cmap="autumn_r"
        else:
            cmap="autumn"
    """ 
        if metric_name== "IoU":
            cmap="Reds_r"
            cmap="autumn"
        else:
            cmap = "Reds"
            cmap="autumn_r"
    """

    if vmin_vmax is None:
        print("repla")
        vmin_vmax = [np.nanmin(metric), np.nanmax(metric)]
        
    
    if divergent:
        norm = TwoSlopeNorm(vmin=vmin_vmax[0], vcenter=0, vmax=vmin_vmax[1])
        cmap="PiYG"
    else:
        norm = Normalize(vmin=vmin_vmax[0], vmax=vmin_vmax[1])

    im = ax.pcolormesh(binned_lons, binned_lats, metric, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm)

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

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=1.)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)

    if return_cbar:
        return ax, im
    else:
        return ax


def plot_massive_binned_map(season_results, domain_lats, domain_lons, vmin_vmax = None):
    rows = 4
    fig = plt.figure(figsize=(17,15),  dpi=400, constrained_layout=True)
    gs = fig.add_gridspec(rows, 5, figure=fig, width_ratios=[1, 1, 1,1,0.05])

    vmin_vmax_default = {"ious_binned":[100*0.1,100*0.5], "mses_binned":[0,1e-6], "corrcoeffs_binned":[0.4,0.7], "logged_corrcoeffs_binned":[0.2,0.5]}


    if vmin_vmax is not None:
        vmin_vmax_default.update(vmin_vmax)


    vmin_vmax = vmin_vmax_default
    print(vmin_vmax)



    ax = np.empty((rows, 4), dtype=object)
    seasons = {"JFM":["01", "02", "03"], "AMJ":["04", "05", "06"], "JAS":["07", "08", "09"], "OND":["10", "11", "12"]}


    for i in range(rows):  
        for j in range(4):
            ax[i, j] = fig.add_subplot(gs[i, j], projection=ccrs.PlateCarree())    

    for season_n, seas in enumerate(seasons.keys()):
        #season_results[seas] = {"mean_mfs":mean_mfs, "mean_mfs_emulated":mean_mfs_emulated}
        ax[0, season_n], cbar_iou = plot_binned_map(ax[0, season_n], season_results[seas]["lon_edges"], season_results[seas]["lat_edges"], 100*season_results[seas]["ious_binned"], cut_lats=(40,20), cmap="metrics", metric_name="IoU", vmin_vmax=vmin_vmax["ious_binned"], cbar=False, cbar_position="right", title=None, return_cbar=True, domain_lats=domain_lats, domain_lons=domain_lons)

        ax[1, season_n], cbar_mses = plot_binned_map(ax[1, season_n], season_results[seas]["lon_edges"], season_results[seas]["lat_edges"], season_results[seas]["mses_binned"], cut_lats=(40,20), cmap="metrics", metric_name="MSE", vmin_vmax=vmin_vmax["mses_binned"], cbar=False, cbar_position="right", title=None, return_cbar=True, domain_lats=domain_lats, domain_lons=domain_lons)

        ax[2, season_n], cbar_corrcoeff = plot_binned_map(ax[2, season_n], season_results[seas]["lon_edges"], season_results[seas]["lat_edges"], season_results[seas]["corrcoeffs_binned"], cut_lats=(40,20), cmap="metrics", metric_name="Corr Coeff", vmin_vmax=vmin_vmax["corrcoeffs_binned"], cbar=False, cbar_position="right", title=None, return_cbar=True, domain_lats=domain_lats, domain_lons=domain_lons)

        ax[3, season_n], cbar_logged_corrcoeff = plot_binned_map(ax[3, season_n], season_results[seas]["lon_edges"], season_results[seas]["lat_edges"], season_results[seas]["logged_corrcoeffs_binned"], cut_lats=(40,20), cmap="metrics", metric_name="Logged Corr Coeff", vmin_vmax=vmin_vmax["logged_corrcoeffs_binned"], cbar=False, cbar_position="right", title=None, return_cbar=True, domain_lats=domain_lats, domain_lons=domain_lons)

        ax[0,season_n].set_title(seas, fontsize=15) 

    metric_names = ["IoU", "MSE", 'Corr Coeff', 'Logged Corr Coeff']

    for ax_num, (cbar_here, metric_name) in enumerate(zip([cbar_iou, cbar_mses, cbar_corrcoeff, cbar_logged_corrcoeff], metric_names)):
        gs_cb = gs[ax_num, -1].subgridspec(3, 1, height_ratios=[1, 50, 1])  
        cbar_ax = fig.add_subplot(gs_cb[1, 0])  
        if metric_name == "IoU":
            cbar = plt.colorbar(cbar_here, cax=cbar_ax, location='right', extend="both", format=mticker.PercentFormatter(decimals=0)).set_label(label=f'{metric_name}', size=12)  
        else:
            cbar = plt.colorbar(cbar_here, cax=cbar_ax, location='right', extend="both").set_label(label=f'{metric_name}', size=12)  

        ax[ax_num, 0].text(-0.1, 0.5, metric_name,  
                va="center", ha="center",  
                rotation="vertical", fontsize=14,  
                transform=ax[ax_num, 0].transAxes, multialignment="center")


    #plt.tight_layout()
    fig.suptitle("Metrics in space: Red means worse performance", fontsize=16)

    #gs.tight_layout(fig)
    gs.update(top=0.95)