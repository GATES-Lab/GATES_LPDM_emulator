import numpy as np
import glob
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import xarray as xr
from .loss_functions import *
from .data.dataloader_graphnet import predict_fluxes

class LOCI():
    """
    Perform Local Intensity Scaling (LOCI, Schmidli et al 2006)

    LOCI works by 
        1. Defining a "wet_day_threshold" (here, this is equivalent to a "non-zero footprint value") so that there are the same number of non-zero values in the real footprints and the predicted footprints
        2. Appling a scaling factor so that the mean of the non-zero values in the corrected footprints is the same as the mean of the non-zero values in the real footprints
    
    Modes:
        - 2D: One threshold and scaling factor for the whole dataset
        - 3D: pixel-by-pixel (one threshold and scaling factor for each pixel)
        - mix: one threshold for the whole dataset, but scaling factor applied pixel-by-pixel
    """
    def __init__(self, mode="2D"):
        self.mode = mode

    def train(self, observed_precip, model_precip):
        """
        train - determine threshold and scaling factor from validation data

        Parameters:
        observed_precip (numpy.ndarray): Array of observed daily precipitation values. Equivalent to true footprints
        model_precip (numpy.ndarray): Array of model daily precipitation values. Equivalent to predicted footprints 
        """

        # Step 1: Determine the wet-day threshold (P_WET) for all footprint
        if self.mode=="2D" or self.mode=="mix":
            wet_day_frequency_observed = np.sum(observed_precip > 0)
            wet_day_threshold_model = np.percentile(model_precip, (100.0 * wet_day_frequency_observed) /  np.prod(np.shape(model_precip)))

            self.wet_day_threshold_model = wet_day_threshold_model
            

        # Step 2: Calculate the scaling factor (s) for all footprint
        if self.mode=="2D":
            wet_day_intensity_observed = np.mean(observed_precip[observed_precip >= wet_day_threshold_model])
            wet_day_intensity_model = np.mean(model_precip[model_precip >= wet_day_threshold_model])
            
            scaling_factor = (wet_day_intensity_observed - wet_day_threshold_model) / (wet_day_intensity_model - wet_day_threshold_model)

            self.scaling_factor = scaling_factor


        if self.mode=="3D" or self.mode=="mix":
            if self.mode=="3D":
                self.wet_day_threshold_model = np.zeros((observed_precip.shape[1], observed_precip.shape[2]))
            else:
                wet_day_threshold_model_here = self.wet_day_threshold_model

            self.scaling_factor = np.zeros((observed_precip.shape[1], observed_precip.shape[2]))

            for lat_idx in range(observed_precip.shape[1]):
                for lon_idx in range(observed_precip.shape[2]):
                    if self.mode=="3D":
                        # Step 1: Determine the wet-day threshold (P_WET) for each (latitude, longitude) pair
                        wet_day_frequency_observed = np.sum(observed_precip[:, lat_idx, lon_idx] > 0)
                        
                        wet_day_threshold_model_here = np.percentile(
                            model_precip[:, lat_idx, lon_idx],
                            (100.0 * wet_day_frequency_observed) / len(model_precip))
                        
                        self.wet_day_threshold_model[lat_idx, lon_idx] = wet_day_threshold_model_here
                    
                    # Step 2: Calculate the scaling factor (s) for each (latitude, longitude) pair
                    wet_day_intensity_observed = np.mean(observed_precip[:, lat_idx, lon_idx][observed_precip[:, lat_idx, lon_idx] >= wet_day_threshold_model_here])
                    
                    if np.sum(model_precip[:, lat_idx, lon_idx] >= wet_day_threshold_model_here) < 50:
                        wet_day_intensity_model = 0
                        scaling_factor = 0
                    else:
                        wet_day_intensity_model = np.mean(model_precip[:, lat_idx, lon_idx][model_precip[:, lat_idx, lon_idx] >= wet_day_threshold_model_here])

                        self.scaling_factor[lat_idx, lon_idx] = (wet_day_intensity_observed - wet_day_threshold_model_here) / (wet_day_intensity_model - wet_day_threshold_model_here)


       

    def adjust(self, observed_precip, model_precip):
        """
        adjust - applied trained parameters to validation data
        
        Parameters:
        observed_precip (numpy.ndarray): Array of observed daily precipitation values. Equivalent to true footprints
        model_precip (numpy.ndarray): Array of model daily precipitation values. Equivalent to predicted footprints 

        Returns:
        adjusted_precip: model_precip with threshold and scaling applied (ie corrected predicted footprints)
        """
        if self.mode=="2D":
            # Step 3: Perform LOCI adjustment to obtain the adjusted precipitation series
            self.adjusted_precip = np.maximum(self.wet_day_threshold_model + self.scaling_factor * (model_precip - self.wet_day_threshold_model), 0) 

        if self.mode=="3D" or self.mode=="mix":
            assert (observed_precip.shape[1], observed_precip.shape[2]) == (self.scaling_factor.shape), "arrays passed to adjust dont have the same lat/lon shape as arrays passed to train on. either used mode 2D or pass arrays with same domain"

            self.adjusted_precip = np.zeros_like(model_precip)
            if self.mode=="mix":
                wet_day_threshold_model_here = self.wet_day_threshold_model
            
            for lat_idx in range(observed_precip.shape[1]):
                for lon_idx in range(observed_precip.shape[2]):
                    if self.mode=="3D":
                        wet_day_threshold_model_here = self.wet_day_threshold_model[lat_idx, lon_idx]

                # Step 3: Perform LOCI adjustment for each (latitude, longitude) pair
                    self.adjusted_precip[:, lat_idx, lon_idx] = np.maximum(
                        wet_day_threshold_model_here + self.scaling_factor[lat_idx, lon_idx] * (model_precip[:, lat_idx, lon_idx] - wet_day_threshold_model_here),
                        0)
            
        #print(np.sum(self.adjusted_precip))
        self.adjusted_precip[self.adjusted_precip<=self.wet_day_threshold_model] = 0   
        #print(np.sum(self.adjusted_precip))  
        return self.adjusted_precip

def loci_adjustment(observed_precip, model_precip, mode="2D"):
    """
    Perform Local Intensity Scaling (LOCI) adjustment on model precipitation data, parameters are defined on the same data that is adjusted later. Equivalent to using class LOCI with the same data in train and adjust.

    Parameters:
        observed_precip (numpy.ndarray): Array of observed daily precipitation values.
        model_precip (numpy.ndarray): Array of model daily precipitation values.

    Returns:
        numpy.ndarray: Adjusted daily precipitation series.
    """
    # Step 1: Determine the wet-day threshold (P_WET) for all footprint
    if mode=="2D" or mode=="mix":
        wet_day_frequency_observed = np.sum(observed_precip > 0)
        wet_day_threshold_model = np.percentile(model_precip, (100.0 * wet_day_frequency_observed) /  np.prod(np.shape(model_precip)))
        

    # Step 2: Calculate the scaling factor (s) for all footprint
    if mode=="2D":
        wet_day_intensity_observed = np.mean(observed_precip[observed_precip >= wet_day_threshold_model])
        wet_day_intensity_model = np.mean(model_precip[model_precip >= wet_day_threshold_model])
        
        scaling_factor = (wet_day_intensity_observed - wet_day_threshold_model) / (wet_day_intensity_model - wet_day_threshold_model)

        # Step 3: Perform LOCI adjustment to obtain the adjusted precipitation series
        adjusted_precip = np.maximum(wet_day_threshold_model + scaling_factor * (model_precip - wet_day_threshold_model), 0)


    if mode=="3D" or mode=="mix":
        adjusted_precip = np.zeros_like(model_precip)
        for lat_idx in range(observed_precip.shape[1]):
            for lon_idx in range(observed_precip.shape[2]):
                if mode=="3D":
                    # Step 1: Determine the wet-day threshold (P_WET) for each (latitude, longitude) pair
                    wet_day_frequency_observed = np.sum(observed_precip[:, lat_idx, lon_idx] > 0)
                    wet_day_threshold_model = np.percentile(
                        model_precip[:, lat_idx, lon_idx],
                        (100.0 * wet_day_frequency_observed) / len(model_precip)
                    )
                
                # Step 2: Calculate the scaling factor (s) for each (latitude, longitude) pair
                wet_day_intensity_observed = np.mean(observed_precip[:, lat_idx, lon_idx][observed_precip[:, lat_idx, lon_idx] >= wet_day_threshold_model])
                if np.sum(model_precip[:, lat_idx, lon_idx] >= wet_day_threshold_model) == 0:
                    wet_day_intensity_model = 0
                    scaling_factor = 0
                else:
                    wet_day_intensity_model = np.mean(model_precip[:, lat_idx, lon_idx][model_precip[:, lat_idx, lon_idx] >= wet_day_threshold_model])
                    scaling_factor = (wet_day_intensity_observed - wet_day_threshold_model) / (wet_day_intensity_model - wet_day_threshold_model)
                    
                # Step 3: Perform LOCI adjustment for each (latitude, longitude) pair
                adjusted_precip[:, lat_idx, lon_idx] = np.maximum(
                    wet_day_threshold_model + scaling_factor * (model_precip[:, lat_idx, lon_idx] - wet_day_threshold_model),
                    0
                )


    adjusted_precip[adjusted_precip<=wet_day_threshold_model] = 0

    return adjusted_precip

    
def apply_threshold(observed_precip, model_precip, to_correct=None):
    """
    Perform only the threshold step from the Local Intensity Scaling (LOCI) method.

    Parameters:
        observed_precip (numpy.ndarray): Array of observed daily precipitation values.
        model_precip (numpy.ndarray): Array of model daily precipitation values.
        to_correct - optional. Pass array of test data to be corrected with the threshold determined on the validation data

    Returns:
        - If to_correct=None, returns threshold
        - else, adjusts test data passed and returns it
    """
    wet_day_frequency_observed = np.sum(observed_precip > 0)
    wet_day_threshold_model = np.percentile(model_precip, (100.0 * wet_day_frequency_observed) /  np.prod(np.shape(model_precip)))

    if to_correct is not None:
        corrected=np.copy(to_correct)
        corrected[corrected<=wet_day_threshold_model] = 0
        return corrected
    else:
        return wet_day_threshold_model


def corrective_layer(truths, preds, size=100, to_correct=None, return_models=True, degree=2):
    """
    Applies a quadratic polynomial correction (on a pixel-by-pixel basis, interpolates quadratic polynomial for error based on the predicted value)

    Parameters:
        observed_precip (numpy.ndarray): Array of observed daily precipitation values.
        model_precip (numpy.ndarray): Array of model daily precipitation values.
        to_correct - optional. Pass array of test data to be corrected with the threshold determined on the validation data

    Returns:
        - If to_correct=None, returns only trained models
        - else, returns corrected data, and models if return_models=True
    """
    models = []
    if to_correct is not None:
        corrected=np.zeros_like(to_correct)
    try:
        for lat_idx in range(size):
            for lon_idx in range(size):
                model = np.poly1d(np.polyfit(np.sort(preds[:, lat_idx, lon_idx]), np.sort(truths[:, lat_idx, lon_idx]) -  np.sort(preds[:, lat_idx, lon_idx]), degree))
                if to_correct is not None:
                    corrected[:, lat_idx, lon_idx] = to_correct[:, lat_idx, lon_idx] + model(to_correct[:, lat_idx, lon_idx])

        models.append(model)
    except:
        print(f"warning, there was an error with the fitter. returning zeros")

    if to_correct is not None:
        if return_models:
            return models, corrected
        else:
            return corrected
    else:
        return models

def corrective_layer_footprint(truths, preds, size=100, to_correct=None, degree=2, return_models=True):
    """
    Applies a quadratic polynomial correction (on a full footprint basis, interpolates quadratic polynomial for error based on the predicted value)

    Parameters:
        observed_precip (numpy.ndarray): Array of observed daily precipitation values.
        model_precip (numpy.ndarray): Array of model daily precipitation values.
        to_correct - optional. Pass array of test data to be corrected with the threshold determined on the validation data

    Returns:
        - If to_correct=None, returns only trained models
        - else, returns corrected data, and models if return_models=True
    """
    if to_correct is not None:
        corrected=np.zeros_like(to_correct)
    try:
        model = np.poly1d(np.polyfit(np.sort(preds.flatten()), np.sort(truths.flatten()) -  np.sort(preds.flatten()), degree))

        if to_correct is not None:
            correction = np.reshape(model(to_correct.flatten()), np.shape(to_correct))
            corrected= to_correct + correction
    except:
        print(f"warning, there was an error with the fitter. returning zeros")

    if to_correct is not None:
        if return_models:
            return model, corrected
        else:
            return corrected
    else:
        return model


# eval funs
def mae(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions))

def checkerboard(boardsize, squaresize=1):
    squaresize=(squaresize, squaresize)
    boardsize = (boardsize,boardsize)
    return np.fromfunction(lambda i, j: (i//squaresize[0])%2 != (j//squaresize[1])%2, boardsize).astype(int)

def mean_bias(truths, preds):
    return np.mean(truths-preds)
def nmae(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions))/np.mean(y_true)


def get_loss(model_name, directory=None):
    print(f"loading last checkpoint for {model_name}")
    if directory is None:
        # change to default directory!
        directory="/user/work/ef17148/GCN/graphnet/graph_weather/trained_satellite_models_fixedmet/"
    files = sorted(glob.glob(glob.escape(f"{directory}{model_name}/{model_name}_")+"*.pt"), key=getint)
    assert len(files)>0, f"no files found for model name {model_name}"
    checkpoint_to_load = files[-1] 
        
    checkpoint = torch.load(checkpoint_to_load, map_location=torch.device('cpu'))
    loss = checkpoint["loss"]
    return loss

def getint(name):
    num = name.split('_')[-1]
    num = num.split('.')[0]
    return int(num)



class ModelEv():
    """
    Run full model evaluation, applying different bias correction approaches

    Default is tuning bias correction on Jan-March, and correct the rest. Pass a regex string or a list of months to validation_set and test_set to change this
    """
    def __init__(self, model_name, size=100, validation_set="0[1-3]", test_set="*", check_validation_overlap=True, path_to_files="/user/work/ef17148/GCN/graphnet/graph_weather/trained_satellite_models_fixedmet/"):
        self.model_name=model_name
        self.size=size

        if "[" in model_name and "[[]" not in model_name:
            model_name = model_name.replace("[","%temp%").replace("]", "[]]").replace("%temp%", "[[]")

        print(f"loading data from {path_to_files}{model_name}/predictions/preds_2016*.nc")
        print(f"using validation data: {validation_set} and test data: {test_set}. Checking overlap between the two: {check_validation_overlap}")
        # load validation data
        if type(validation_set) is list:
            validation_files = []
            for date in validation_set:
                validation_files = validation_files + glob.glob(f"{path_to_files}{model_name}/predictions/preds_2016{date}.nc")
        elif type(validation_set) is str:
            validation_files = glob.glob(f"{path_to_files}{model_name}/predictions/preds_2016{validation_set}.nc")
        self.preds_validation = xr.open_mfdataset(validation_files)
        self.preds_validation.load()

        self.preds_validation.trans_predictions.values = self.preds_validation.trans_predictions.where(self.preds_validation.trans_predictions<0.1, other=0)


        # load test data
        if type(test_set) is list:
            test_files = []
            for date in test_set:
                test_files = test_files + glob.glob(f"{path_to_files}{model_name}/predictions/preds_2016{date}.nc")
        elif type(test_set) is str:
            test_files = glob.glob(f"{path_to_files}{model_name}/predictions/preds_2016{test_set}.nc")
        self.preds_test = xr.open_mfdataset(test_files)
        if check_validation_overlap:
            # make sure that there is no overlap between the test and the validation sets
            self.preds_test = self.preds_test.sel(time=list(set(self.preds_test.time.values)-set(self.preds_validation.time.values)))
        
        self.preds_test.load()

        self.preds_test.trans_predictions.values = self.preds_test.trans_predictions.where(self.preds_test.trans_predictions<0.1, other=0)

        if size < len(self.preds_validation.lat):
            print("size passed is smaller than actual array size. cutting array to match size parameter")
            centre = int(len(self.preds_validation.lat)/2)
            self.preds_validation = self.preds_validation.sel(lat=slice(centre-int(size/2), centre+int(size/2)-1), lon=slice(centre-int(size/2), centre+int(size/2)-1))
            self.preds_test = self.preds_test.sel(lat=slice(centre-int(size/2), centre+int(size/2)-1), lon=slice(centre-int(size/2), centre+int(size/2)-1))

        self.preds_test = self.preds_test.sortby("time")
        self.preds_validation = self.preds_validation.sortby("time")

    def get_adjusted_models(self, which="all"):
        observed_precip_val = np.copy(self.preds_validation.fp.values)
        model_precip_val = np.copy(self.preds_validation.trans_predictions.values)
        observed_precip_test = np.copy(self.preds_test.fp.values)
        model_precip_test = np.copy(self.preds_test.trans_predictions.values)
        model_precip = np.copy(model_precip_test)
        
        if which=="thr":
            adjusted_precipitation_thr = apply_threshold(observed_precip_val, model_precip_val, to_correct=model_precip_test)

            self.variations = {"preds":model_precip, "thr":adjusted_precipitation_thr}

        if which=="all":
            stloci = LOCI(mode="mix")
            stloci.train(observed_precip_val, model_precip_val)
            adjusted_precipitation_mix = stloci.adjust(observed_precip_test, model_precip_test)
            stloci = LOCI(mode="2D")
            stloci.train(observed_precip_val, model_precip_val)
            adjusted_precipitation = stloci.adjust(observed_precip_test, model_precip_test)
            stloci = LOCI(mode="3D")
            stloci.train(observed_precip_val, model_precip_val)
            adjusted_precipitation_3D = stloci.adjust(observed_precip_test, model_precip_test)

            adjusted_precipitation_bias_pbp = corrective_layer(observed_precip_val, model_precip_val, to_correct=model_precip_test, return_models=False, size=self.size)

            adjusted_precipitation_bias_pbp_linear = corrective_layer(observed_precip_val, model_precip_val, to_correct=model_precip_test, return_models=False, size=self.size, degree=1)


            adjusted_precipitation_bias = corrective_layer_footprint(observed_precip_val, model_precip_val, to_correct=model_precip_test, return_models=False, size=self.size)

            adjusted_precipitation_bias_linear = corrective_layer_footprint(observed_precip_val, model_precip_val, to_correct=model_precip_test, return_models=False, size=self.size, degree=1)

            adjusted_precipitation_thr = apply_threshold(observed_precip_val, model_precip_val, to_correct=model_precip_test)
            adjusted_precipitation_bias_thr = apply_threshold(observed_precip_val, model_precip_val, to_correct=adjusted_precipitation_bias)
            adjusted_precipitation_bias_l_thr = apply_threshold(observed_precip_val, model_precip_val, to_correct=adjusted_precipitation_bias_linear)


            self.variations = {"preds":model_precip, "thr":adjusted_precipitation_thr,"bias_pbp_q":adjusted_precipitation_bias_pbp, "bias_pbp_l":adjusted_precipitation_bias_pbp_linear, "bias_q":adjusted_precipitation_bias, "bias_l":adjusted_precipitation_bias_linear,"bias_thr":adjusted_precipitation_bias_thr, "bias_thr_l":adjusted_precipitation_bias_l_thr, "2D":adjusted_precipitation,"3D":adjusted_precipitation_3D,"mix":adjusted_precipitation_mix}

        if which=="small":
            adjusted_precipitation_bias_pbp = corrective_layer(observed_precip_val, model_precip_val, to_correct=model_precip_test, return_models=False, size=self.size)

            adjusted_precipitation_bias_pbp_linear = corrective_layer(observed_precip_val, model_precip_val, to_correct=model_precip_test, return_models=False, size=self.size, degree=1)


            adjusted_precipitation_bias = corrective_layer_footprint(observed_precip_val, model_precip_val, to_correct=model_precip_test, return_models=False, size=self.size)

            adjusted_precipitation_bias_linear = corrective_layer_footprint(observed_precip_val, model_precip_val, to_correct=model_precip_test, return_models=False, size=self.size, degree=1)

            adjusted_precipitation_thr = apply_threshold(observed_precip_val, model_precip_val, to_correct=model_precip_test)
            adjusted_precipitation_bias_thr = apply_threshold(observed_precip_val, model_precip_val, to_correct=adjusted_precipitation_bias)
            adjusted_precipitation_bias_l_thr = apply_threshold(observed_precip_val, model_precip_val, to_correct=adjusted_precipitation_bias_linear)


            self.variations = {"preds":model_precip, "thr":adjusted_precipitation_thr,"bias_pbp_q":adjusted_precipitation_bias_pbp, "bias_pbp_l":adjusted_precipitation_bias_pbp_linear, "bias_q":adjusted_precipitation_bias, "bias_l":adjusted_precipitation_bias_linear,"bias_thr":adjusted_precipitation_bias_thr, "bias_thr_l":adjusted_precipitation_bias_l_thr}            

        self.variations_meanings = {
            "preds":"model predictions", 
            "thr":"LOCI threshold applied",
            "bias_pbp_q":"quadratic bias correction, applied pixel by pixel", 
            "bias_pbp_l":"linear bias correction, applied pixel by pixel", 
            "bias_q":"quadratic bias correction, applied to all the data",
            "bias_l":"linear bias correction, applied to all the data",
            "bias_thr":"quadratic bias correction, applied to all the data, plus LOCI threshold", 
            "bias_thr_l":"linear bias correction, applied to all the data, plus LOCI threshold",
            "2D":"2D LOCI correction",
            "3D":"3D LOCI correction",
            "mix":"mixed LOCI correction",
        }

    
    def segmentation_evaluation(self):
        ious = {}
        dices = {}
        accuracies = {}
        for name in self.variations:
            ious[name] = intersection_over_union(self.preds_test.fp.values, self.variations[name])
            dices[name] = dice_similarity(self.preds_test.fp.values, self.variations[name])
            accuracies[name] = accuracy(self.preds_test.fp.values, self.variations[name])

        self.segmentation_metrics = {"IoU":ious, "Dice":dices, "Acc":accuracies} 

    def get_fluxes(self, fluxes="def"):
        print(type(fluxes))
        if type(fluxes) is str and fluxes=="def":
            fluxes = np.ones((self.size,self.size))
        observed_precip = np.copy(self.preds_test.fp.values)
        self.pred_fluxes = {}
        for name in self.variations:
            true_flux, pred_flux = predict_fluxes(observed_precip, self.variations[name], fluxes)
            self.pred_fluxes[name] = pred_flux
        self.true_flux=true_flux
        
    def evaluate(self, verbose=True):
        print(f"evaluation for model {self.model_name}")
        #metrics=["adj", "MAE", "IoU", "Dice", "Acc", "R2", "Flux_mae", "Bias", "NMAE"]
        observed_precip = np.copy(self.preds_test.fp.values)
        all_variations = []
        for name in self.variations:
            this_metric = {"adj":name}
            this_metric["MAE"] = mae(observed_precip, self.variations[name])
            this_metric["MSE"] = mean_squared_error(observed_precip.flatten(), self.variations[name].flatten())
            if hasattr(self, "segmentation_metrics"):
                for seg_metric in self.segmentation_metrics:
                    this_metric[seg_metric] = round(100*np.mean(self.segmentation_metrics[seg_metric][name]),2)
            if hasattr(self, "pred_fluxes"):
                try:
                    this_metric["r2"] = round(100*r2_score(self.true_flux, self.pred_fluxes[name]),2)
                    this_metric["flux_mae"]=round(mean_absolute_error(self.true_flux, self.pred_fluxes[name]),2)
                    this_metric["bias"]=round(mean_bias(self.true_flux, self.pred_fluxes[name]),4)
                    this_metric["NMAE"]=round(nmae(self.true_flux, self.pred_fluxes[name]),4)
                except ValueError:
                    this_metric["r2"] = np.nan
                    this_metric["flux_mae"]= np.nan
                    this_metric["bias"]= np.nan
                    this_metric["NMAE"]= np.nan               
        
            all_variations.append(this_metric)

        self.evaluation_results = pd.DataFrame(all_variations)
        if verbose: 
            print(self.evaluation_results)

        if verbose:
            print(f"Column:          Best scoring method:          Score:")
            for column in self.evaluation_results.columns[1:]:
                if column in ["IoU", "Dice", "Acc", "r2"]:
                    best_row = self.evaluation_results[column].idxmax()
                    best_value = self.evaluation_results[column].max()
                elif column in ["bias"]:
                    best_row = abs(self.evaluation_results[column]).idxmin()
                    best_value = abs(self.evaluation_results[column]).min()                    
                else:
                    best_row = self.evaluation_results[column].idxmin()  # Find the row with the highest value in the column
                    best_value = self.evaluation_results[column].min()  # Get the highest value itself

                row_name = self.evaluation_results.adj[best_row]  # Get the corresponding row name

                print(f"{column}              {row_name}               {best_value}")
            
    def full_ev(self, verbose=True, which="small"):
        self.get_adjusted_models(which)
        self.segmentation_evaluation()
        self.get_fluxes()
        self.evaluate(verbose=verbose)

