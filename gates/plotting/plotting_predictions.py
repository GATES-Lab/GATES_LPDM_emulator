import numpy as np
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature



### plotting
def plot_footprint_ax(ax_true, ax_pred, prediction_ds, plotting_labels=["fp_original", "fp_pred"], vmin=None, vmax=None, thres=0, contour=False, share_minmax=False, cbar=True,nlevels=None, title=False, return_minmax=False, print_stats=False, log=True):
    """
    Plot the true and predicted footprints on the given axes.
    Parameters:
    - ax_true: Matplotlib axis for the true footprint.
    - ax_pred: Matplotlib axis for the predicted footprint.
    - prediction_ds: xarray Dataset containing the true and predicted footprints, along with coordinates. It must have length 1 along the time dimension, and contain the variables specified in plotting_labels.
    - plotting_labels: List of two strings specifying the variable names in prediction_ds for the true and predicted footprints, respectively. Defaults to ["fp_original", "fp_pred"].
    - vmin, vmax: Color scale limits for the plots. If share_minmax is True, these limits will be applied to both plots. If None, they will be determined from the data.
    - thres: Threshold below which footprint values will be masked (set to NaN) in the second contour plot. Defaults to 0.
    - contour: If True, use contourf to plot the footprints. If False, use imshow. Defaults to False.
    - share_minmax: If True, use the same vmin and vmax for both true and predicted plots. Defaults to False.
    - cbar: If True, return the colorbar object for the predicted plot. Defaults to True.
    - nlevels: Number of contour levels or list of levels to use if contour is True. If None, levels will be automatically determined.
    - title: If True, set the title of the true plot to the timestamp of the footprint. Defaults to False.
    - return_minmax: If True, return the vmin and vmax used for the plots. Defaults to False.
    - log: If True, plot the log10 of the footprint values. Defaults to True. If plotting footprints in the transformed space, it should be False, and the color scale limits should be set accordingly (e.g. vmin=0, vmax=3 for normalized footprints).
    """

    np.seterr(divide='ignore')

    extent = (prediction_ds.lon_coords.values[0], prediction_ds.lon_coords.values[-1], prediction_ds.lat_coords.values[0], prediction_ds.lat_coords.values[-1])
    ax_true.set_extent(extent, crs=cartopy.crs.PlateCarree())
    ax_true.coastlines(resolution='110m', color='black', linewidth=1, alpha=0.5)
    ax_true.add_feature(cfeature.LAND)
    ax_true.add_feature(cfeature.OCEAN)    

    alpha =0.6


    f = np.copy(prediction_ds[plotting_labels[0]].values)
    if log:
        f_plot = np.log10(f)
    else:
        f_plot = f
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
        cb = ax_true.contourf(prediction_ds.lon_coords.values, prediction_ds.lat_coords.values, f_plot, **plot_params, levels=nlevels, extend="both", alpha=alpha)
        f_plot[f<thres] = np.nan
        cb = ax_true.contourf(prediction_ds.lon_coords.values, prediction_ds.lat_coords.values, f_plot, **plot_params, levels=nlevels, extend="both")


    else:
        cb = ax_true.imshow(f_plot, extent=extent, origin="lower", **plot_params, zorder=5)

    if title:
        #time_str = np.datetime_as_string(mod.all_fps.time.values[idx], unit='m')
        #formatted_time = np.datetime64(time_str).astype('datetime64[m]').astype('O').strftime('%d-%m-%Y %H:%M')
        formatted_time = prediction_ds.time.values.astype('datetime64[m]').astype('O').strftime('%d-%m-%Y %H:%M')
        ax_true.set_title(formatted_time)
        #ax_true.set_title(str(np.datetime_as_string(mod.all_fps.time.values[idx], unit='m'))[:10]+ " " + str(np.datetime_as_string(mod.all_fps.time.values[idx], unit='m'))[11:])


    ax_pred.set_extent(extent, crs=cartopy.crs.PlateCarree())
    ax_pred.coastlines(resolution='110m', color='black', linewidth=1, alpha=0.5)
    ax_pred.add_feature(cfeature.LAND)
    ax_pred.add_feature(cfeature.OCEAN)    


    f = np.copy(prediction_ds[plotting_labels[1]].values)
    if log:
        f_plot = np.log10(f)
    else:
        f_plot = f

    # if print_stats:
    #     stats = get_stats(model, idx, do_print=False, do_return=True)
    #     ax_pred.text(0.5, -0.1, f"IoU: {str(100*stats["IoU"])[:4]}% \n MSE: {str(stats["MSE"])[:4]+str(stats["MSE"])[-4:]} \n Corr {str(stats["Corrcoef"])[:4]}\n", transform=ax_pred.transAxes, 
    #     ha="center", va="top", fontsize=12)    

    if contour:
        cb = ax_pred.contourf(prediction_ds.lon_coords.values, prediction_ds.lat_coords.values, f_plot, **plot_params, levels=nlevels, extend="both", alpha=alpha)

        f_plot[f<thres] = np.nan

        cb = ax_pred.contourf(prediction_ds.lon_coords.values, prediction_ds.lat_coords.values, f_plot, **plot_params, levels=nlevels, extend="both")
    else:
        cb = ax_pred.imshow(f_plot, extent=extent, origin="lower", **plot_params, zorder=5)
    
    if return_minmax:
        return vmin, vmax
    if cbar:
        return cb


def plot_fp_predictions(prediction_ds, idxs_list, fig_title=None, plot_timeseries=False, plot_fluxes=False, print_stats=False, contour=False, thres=0, ylim_timeseries=45, levels=None, vmin_vmax=None, which_dataspace="original"):
    """
    Plot the true and predicted footprints for multiple time steps, along with optional timeseries and flux plots.
    Parameters:
    - prediction_ds: xarray Dataset containing the true and predicted footprints, along with coordinates.
    - idxs_list: List of integer indices along the time dimension of prediction_ds for which to plot the footprints.
    - fig_title : Optional string for the overall figure title.
    - plot_timeseries: If True, include a row of timeseries plots showing the footprint [ still in implkementation]
    """
    plot_prior=False


    # altnerative fig creating for more flexibility
    n_rows = 2 + plot_timeseries + 2*plot_fluxes + print_stats
    height_ratios = [1] * n_rows  # Default all equal
    if print_stats:
        height_ratios[2] = 0
    print(height_ratios)
    
    fig_width = 4 * len(idxs_list) if len(idxs_list) > 1 else 6
    fig = plt.figure(figsize=(fig_width,6+3*(n_rows-2-print_stats)), dpi=400)


    

    gs = fig.add_gridspec(n_rows, len(idxs_list)+1, width_ratios=[1]*len(idxs_list)+[0.1], height_ratios=height_ratios)
    #gs = fig.add_gridspec(n_rows, 4, width_ratios=[1, 1, 1,1], height_ratios=height_ratios)

    ax = np.empty((n_rows, len(idxs_list)), dtype=object)

    # Create geoaxes for the first two rows (LPDM and gates footprints)
    for i in range(2):  
        for j in range((len(idxs_list))):
            ax[i, j] = fig.add_subplot(gs[i, j], projection=ccrs.PlateCarree())    

    if print_stats:
        stats_row = 2  # Last row
        #for j in range(4):
        #    ax[stats_row, j] = fig.add_subplot(gs[stats_row, j])
        #    ax[stats_row, j].axis("off")  # Hide stats axes



    if plot_fluxes:
        fluxes_row = 2+print_stats
        for j in range(len(idxs_list)):
            ax[fluxes_row, j] = fig.add_subplot(gs[fluxes_row, j], projection=ccrs.PlateCarree())  
            ax[fluxes_row+1, j] = fig.add_subplot(gs[fluxes_row+1, j], projection=ccrs.PlateCarree())  
            
    if plot_timeseries:
        timeseries_row = 2+print_stats + 2*plot_fluxes 
        for j in range(len(idxs_list)):
            ax[timeseries_row, j] = fig.add_subplot(gs[timeseries_row, j])
    
    """

    elif plot_timeseries:
        fig = plt.figure(figsize=(16,8), dpi=400)
        gs = fig.add_gridspec(3, 4)#, height_ratios=[1,1,0.5])
        print("here")
        ax = np.array([[fig.add_subplot(gs[0, j], projection=ccrs.PlateCarree()) for j in [0,1,2,3]], [fig.add_subplot(gs[1, j], projection=ccrs.PlateCarree()) for j in [0,1,2,3]], [fig.add_subplot(gs[2, j]) for j in [0,1,2,3]]])

    else:
        fig, ax = plt.subplots(2,4 , figsize=(12,5), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=500)
    """
    
    #contour=True
    if levels is None and which_dataspace=="original":
        levels = np.arange(-5.5, -1.5+0.6, 0.5)
        levels = [-4, -3.5, -3,  -2.5, -2, -1.5]

    if vmin_vmax is None and (which_dataspace=="original" or which_dataspace=="thresholded"):
        vmin, vmax = (-4, -2)
    elif vmin_vmax is None and which_dataspace=="transformed":
        vmin, vmax = (None, None)
    else:
        vmin, vmax = vmin_vmax
    #vmin, vmax = (-3, -1.5)

    #idx_1 = 150 
    """ 
    if plot_timeseries or plot_fluxes:
        print_stats = False
    else:
        print_stats = True
    """
    time_idxs = [prediction_ds.time.values[idx] for idx in idxs_list]

    if which_dataspace== "original":
        plotting_labels=["fp_original", "fp_pred"]
        log=True
    elif which_dataspace == "thresholded":
        plotting_labels= ["fp_original", "fp_pred_thres"]
        log=True
    elif which_dataspace == "transformed":
        plotting_labels= ["fp_transformed", "fp_transformed_pred"]
        log=False

    for n, idx in enumerate(idxs_list):
        cb = plot_footprint_ax(ax[0,n], ax[1,n], prediction_ds.sel(time=time_idxs[n]), contour=contour, share_minmax=True, nlevels=levels, title=True, return_minmax=False, vmin=vmin, vmax=vmax, print_stats=print_stats, thres=thres,plotting_labels=plotting_labels, log=log, cbar=True)

    """
    if plot_fluxes:
        
        if plot_prior:
            prior_ax = ax[2,n]
        else:
            prior_ax=None
        
        for n, idx in enumerate([idx_1, idx_2, idx_3, idx_4]):
            flux_cb = plot_emissions_ax(None, ax[fluxes_row,n], ax[fluxes_row+1,n], model, idx, cbar=True)


    if plot_timeseries:
        days_before = 1
        days_after=1
        plot_a_day_ax(ax[timeseries_row,0], gs[timeseries_row,0], fig, str(model.all_fps.time[idx_1].values)[:10], model, days_before=days_before, days_after=days_after, dateline_idx=idx_1, ylim=ylim_timeseries, labels=True) 
        for n, idx in enumerate([idx_2, idx_3, idx_4]):
            plot_a_day_ax(ax[timeseries_row,n+1], gs[timeseries_row,n+1], fig, str(model.all_fps.time[idx].values)[:10], model, days_before=days_before, days_after=days_after, dateline_idx=idx, ylim=ylim_timeseries)     
        """


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


        #cbar = plt.colorbar(flux_cb, cax=cbar_ax_flux, location='right', extend="both").set_label(label=r'log$_{10}$'+"(contribution to mole fraction \nfrom each cell, in ppb)", size=12)


    gs_cb = gs[:2, -1].subgridspec(3, 1, height_ratios=[1, 15, 1])  

    cbar_ax = fig.add_subplot(gs_cb[1, 0])  

    #cbar_ax = fig.add_subplot(gs[:2, -1])

    cbar = plt.colorbar(cb, cax=cbar_ax, location='right', extend="both").set_label(label=r'log$_{10}$ (mol mol$^{-1}$ (mol m$^{-2}$ s$^{-1}$)$^{-1}$)', size=12)
    plt.tight_layout()

    #ax[0,0].set_title(mod.all_fps.time.values[150])
    if fig_title is not None:
        #fig.suptitle(fig_title)
        fig.text(0.3, 1, fig_title, va="center",  fontsize=21,multialignment="center")
