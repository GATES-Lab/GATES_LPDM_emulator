from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import sklearn.preprocessing as preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

from ..loss_functions import *

from sklearn.preprocessing import MinMaxScaler
import copy


class FootprintsDataset(Dataset):
    """
    Creates dataset to pass to the model, applying any transformations specified. 
    Parameters:
    - inputs: array of shape (samples, sizexsize, features) with extracted inputs
    - fp: array of shape (samples, sizexsize) with footprint
    - input_names: list of dictionaries with variable names and info. each variable entry should have at least {"var":var_name} format, or {"var":var_name, "level":level} if required
    - input_transforms: list containing all transforms to apply to the inputs. Current valid options are:
        - clever_transform (standardise each variable separately across all levels and times)
        - clever_transform_3(standardise each variable and level separately across all times)
    - output_transforms: list containing all transforms to apply to the outputs, in order. Current valid options are:
        - boxcox (apply pixel-wise boxcox)
        - boxcox_all (apply boxcox to all the data)
        - mu-law (apply mu-law enconding algorithm)
        - logv3 (take log10 and shift so mean of non-zero elements is zero)
    - transform_parameters: pass optional parameters to transform. Only used during training (during test, only parameters in test_mode are considering). Pass in format {"transform_name":{"param1":value, "param2":value}}
    
    - test_mode: dict. If training, leave empty. If testing (ie applying existing parameters and/or already fitted models), pass a dictionary or the transform_parameters of another dataset

    Example:
    training_ds = FootprintsDataset(inputs, data.fp_data, input_transforms=["clever_transform"],output_transforms=["boxcox_all"], input_names=names)
    testing_ds = FootprintsDataset(test_inputs, test_data.fp_data, input_transforms=["clever_transform"],output_transforms=["boxcox_all"], input_names=names, test_mode=training_ds.transform_parameters))

    Functions:
    - inverse_transform(predictions): provides the footprints reconverted to the original space, inverting any previously applied transforms
    - add_prototypes(prototypes) : unfinished! Takes prototypes of same shape as footprints, applies to them the same transform applied to the fps and appends them to the inputs array.

    """
    def __init__(self, inputs, fp, input_transforms = [], output_transforms = [], transform_parameters = {}, test_mode={}, input_names=[], size=None, full_land_cover=False):
        #super().__init__()
        self.inputs = np.copy(inputs)
        self.fp = np.copy(fp)
        if size is None:
            self.size = [int(np.sqrt(np.shape(fp)[-1])), int(np.sqrt(np.shape(fp)[-1]))]
            print(f"assuming this is a square dataset of size {self.size[0]} x {self.size[1]}!")
        elif type(size) is int:
            assert size == int(np.sqrt(np.shape(fp)[-1])), f"you passed size as an int, which normally implies a square dataset of size {size} x {size}, but the data size doesnt match (data would have size {int(np.sqrt(np.shape(fp)[-1]))} x {int(np.sqrt(np.shape(fp)[-1]))}). are you sure the data is square and of the size you pass?"
            self.size = [size, size]
        elif len(size) == 2 and (type(size) is tuple or type(size) is list):
            self.size = size
            print(f"assuming a non-square domain of size {self.size}")
        else:
            raise ValueError("you passed a size parameter of an unknown format. pass size=None if the dataset is square, or a list/tuple")

        self.input_transforms= copy.deepcopy(input_transforms)
        self.output_transforms= copy.deepcopy(output_transforms)
        self.input_names = copy.deepcopy(input_names)
        self.test_mode = copy.deepcopy(test_mode)
        self.transform_parameters = copy.deepcopy(transform_parameters)
        self.prototypes_added = False
        
        print(transform_parameters)
        print(self.transform_parameters)


        # note that _Transform is the base class and will raise a not_implemented error if used
        self.valid_input_transforms = {
            "clever_transform":{"params":["transformers"], "fun":_CleverTransform}, 
            "clever_transform_2":{"params":["transformers"], "fun":_CleverTransform2}, 
            "clever_transform_3":{"params":["transformers"], "fun":_CleverTransform3},
            "standardise":{"params":["transformers"], "fun":_Transform}, 
            "scale":{"params":["inputs_min", "inputs_max"], "fun":_Transform}}

        self.valid_output_transforms = {
            "boxcox":{"params":["transformers"], "fun":_Boxcox},
            "boxcox_all":{"params":["transformers"], "fun":_BoxcoxAll}, 
            "distance_boxcox":{"params":["transformers", "distances", "zero_shift"], "fun":_DistanceBoxcox},
            "mu-law":{"params":["scale", "mu"], "fun":_MuLaw}, 
            "logv1":{"params":["fp_mean", "fp_var"], "fun":_Transform}, 
            "logv2":{"params":[], "fun":_Transform}, 
            "logv3":{"params":["logged_mean"], "fun":_LogV3},
            "logv4":{"params":[], "fun":_LogV4},
            "logv3e":{"params":["logged_mean"], "fun":_LogV3e},
            "scale":{"params":["output_minmax"], "fun":_Transform},
            "mixed_log":{"params":[], "fun":_MixedLog}}


        ## assert that only valid input and output transforms have been passed
        assert set(self.input_transforms).issubset(self.valid_input_transforms.keys()), f"You passed some input transforms that are not in the list of valid transforms. \n The valid transforms are {list(self.valid_input_transforms.keys())}. \n The following transforms you passed but are not allowed: {set(self.input_transforms) - set(self.valid_input_transforms.keys())}"

        assert set(self.output_transforms).issubset(self.valid_output_transforms.keys()), f"You passed some input transforms that are not in the list of valid transforms. \n The valid transforms are {list(self.valid_output_transforms.keys())}. \n The following transforms you passed but are not allowed: {set(self.output_transforms) - set(self.valid_output_transforms.keys())}"


        if len(self.test_mode)>0:
            self.mode="test"
            # if on test mode, assert that all necessary parameters for the provided input and output transforms have been passed
            assert all([set(self.valid_input_transforms[x]["params"]).issubset(self.test_mode[x]) for x in self.input_transforms]), "You did not pass all the necessary parameters to test_mode for the input transformations!"

            assert all([set(self.valid_output_transforms[x]["params"]).issubset(self.test_mode[x]) for x in self.output_transforms]), "You did not pass all the necessary parameters to test_mode for the output transformations!"

        else:
            self.mode="train"        

        self.input_transforms = {key: None for key in self.input_transforms}
        for transform in self.input_transforms:
            self.input_transforms[transform]  = self.valid_input_transforms[transform]["fun"](self)
            self.input_transforms[transform].transform()

        self.output_transforms = {key: None for key in self.output_transforms}
        for transform in self.output_transforms:
            if transform in self.transform_parameters:
                print(self.transform_parameters)
                self.output_transforms[transform]  = self.valid_output_transforms[transform]["fun"](self, **self.transform_parameters[transform])
            else:
                self.output_transforms[transform]  = self.valid_output_transforms[transform]["fun"](self)

            self.fp = self.output_transforms[transform].transform(self.fp)


        if type(self.inputs) != torch.Tensor:
            self.inputs = torch.tensor(self.inputs, dtype=torch.float)
        if type(self.fp) != torch.Tensor:
            self.fp = torch.tensor(self.fp, dtype=torch.float)
            self.fp_numpy = self.fp.detach().numpy()

        if self.fp.ndim==2:
            self.fp = self.fp[:,:,None]

    def inverse_transform(self, predictions, return_transformed=True):
        self.predictions=predictions
        if type(self.predictions) == torch.Tensor:
            self.predictions = self.predictions.detach().numpy()
            
        for transform in self.output_transforms:
            self.transformed_predictions = self.output_transforms[transform].inverse_transform(self.predictions)
        if return_transformed:
            return self.transformed_predictions
    
    def add_prototypes(self, prototypes):
        """
        appends footprint prototypes to the input data
        """

        assert list(np.shape(self.fp)) == list(np.shape(prototypes)), f"The footprints and the prototypes should have the same size, but right now they have shapes {list(np.shape(self.fp))} and {list(np.shape(prototypes))} respectively"
        
        if self.prototypes_added:
            print("Careful, prototypes have already been added! replacing them with the newly passed ones")


        self.prototypes = prototypes
        for transform in self.output_transforms:
            self.prototypes = self.output_transforms[transform].transform(self.prototypes)

        if type(self.prototypes) != torch.Tensor:
            self.prototypes = torch.tensor(self.prototypes, dtype=torch.float)

        if not self.prototypes_added:
            self.inputs = torch.cat([self.inputs, self.prototypes[:,:,None]], dim=2)
            self.prototypes_added = True
            if len(self.input_names)>0:
                self.input_names.append({"var":"prototypes", "type":"not met"})

        elif self.prototypes_added:
            self.inputs = torch.cat([self.inputs[:,:,:-1], self.prototypes[:,:,None]], dim=2)
        
    def evaluate(self):
        """
        output metrics (normalised absolute mean, MSE, accuracy, IOU) for the predictions transformed back to the original data space
        """
        assert hasattr(self, "transformed_predictions"), "only works currently for transformed/normalised outputs!"

        assert np.shape(self.transformed_predictions) == np.shape(self.fp_untransformed), "predictions and fp have to have the same shape to evaluate errors"

        print(f"nans in transformed_predictions: {np.sum(np.isnan(self.transformed_predictions))} ")
        self.transformed_predictions = np.nan_to_num(self.transformed_predictions)
        print(f"nans in transformed_predictions after nan to num: {np.sum(np.isnan(self.transformed_predictions))} ")
        print(f"nans in fp_untransformed: {np.sum(np.isnan(self.fp_untransformed))} ")
        metrics = {}
        metrics["NMAE"] = NMAE_nans(self.transformed_predictions, self.fp_untransformed)
        metrics["MSE"] = mean_squared_error(self.transformed_predictions, np.nan_to_num(self.fp_untransformed))
        metrics["Accuracy"] = accuracy(self.transformed_predictions, np.nan_to_num(self.fp_untransformed), threshold=5e-5)
        metrics["IOU"] = intersection_over_union(self.transformed_predictions, np.nan_to_num(self.fp_untransformed), threshold=5e-5)

        print("evaluation metrics:", metrics)
        return metrics

    def predict_fluxes(self, flux, units_transform = "default"):
        ## convolute predicted footprints and fluxes, returns two np arrays, one with the true flux and one with the emulated flux, of shape (n_footprints,)
        ## flux is an array, regridded and cut to the same resolution and size of the footprints
        ## units_transform can be None (use fluxes directly), "default" (performs flux*1e3 / CH4molarmass) or another function (which should return an array of the same shape as the original flux)
        try:
            self.transformed_predictions
        except AttributeError:
            print("only works currently for transformed/normalised outputs!")

        if units_transform != None:
            if units_transform == "default":
                # only if the flux is in kg m s
                molarmass = 16.0425
                flux = flux*1e3 / molarmass
            else:
                flux = units_transform(flux)

        true_concentration = np.reshape(np.nan_to_num(self.fp_untransformed), (len(self.fp_untransformed), self.size[0], self.size[1]))*flux
        self.true_flux = np.sum(true_concentration, axis = (1,2))
        pred_concentration = np.reshape(self.transformed_predictions, (len(self.transformed_predictions), self.size[0], self.size[1]))*flux
        self.pred_flux = np.sum(pred_concentration, axis = (1,2))
        
        return self.true_flux, self.pred_flux


    def evaluate_flux(self, mode="uniform"):
        def checkerboard(boardsize, squaresize=1):
            if type(boardsize) is int:
                boardsize= (boardsize,boardsize)
            if type(squaresize) is int:
                squaresize=(squaresize, squaresize)
            return np.fromfunction(lambda i, j: (i//squaresize[0])%2 != (j//squaresize[1])%2, boardsize).astype(int)


        if mode=="uniform":
            flux = np.ones((self.size[0], self.size[1]))
        elif mode=="checkerboard_10":
            flux = checkerboard(self.size, 10)
        elif mode=="checkerboard_5":
            flux = checkerboard(self.size, 5)
        elif mode=="checkerboard_1":
            flux = checkerboard(self.size, 1)


        if mode in ["uniform","checkerboard_10","checkerboard_5", "checkerboard_1"]:
            self.predict_fluxes(flux, units_transform = None)


        metrics = {"MAE":mean_absolute_error(self.true_flux, self.pred_flux), "R2": r2_score(self.true_flux, self.pred_flux)}
        print("flux metrics:", metrics)
        return metrics
    
    def plot_footprints(self, idx):
        """
        visualise footprints and predictions for the footprints at index idx
        idx can be an int or a list of ints
        plots graph of size (len(idx), 4) with true footprint in original space and transformed space, and prediction in both spaces
        """

        assert type(idx) is int or type(idx) is list, "idx should be an int or list of ints"

        if type(idx) is int:
            idx=[idx]

        if hasattr(self, "transformed_predictions"): 
            plots = 4
        else:
            plots=2

        fig, ax = plt.subplots((len(idx)), plots, figsize=(plots*4,plots*len(idx)))
        if len(idx)==1:
            ax = ax[None,:]
        for idx_n, index in enumerate(idx):
            ax[idx_n,0].imshow(np.reshape(self.fp_untransformed[index], (self.size[0], self.size[1])), origin="lower")
            ax[idx_n,1].imshow(np.reshape(self.fp_numpy[index], (self.size[0], self.size[1])), origin="lower")
            if plots==4:
                ax[idx_n,2].imshow(np.reshape(self.transformed_predictions[index], (self.size[0], self.size[1])), origin="lower")
                ax[idx_n,3].imshow(np.reshape(self.predictions[index], (self.size[0], self.size[1])), origin="lower")

            ax[idx_n,0].set_ylabel(f"fp at index {index}")

        ax[0,0].set_title("True Footprint \n original space")
        ax[0,1].set_title("True Footprint \n transformed space")
        if plots==4:
            ax[0,2].set_title("Predicted Footprint \n original space")
            ax[0,3].set_title("Predicted Footprint \n transformed space")

        for axis in ax.flatten():
            axis.tick_params(left = False, bottom = False, labelbottom=False, labelleft=False) 
            #axis.tick_params(axis='y', colors='white')   

        fig.patch.set_facecolor('white')
            
    def plot_flux(self, window_n, window_size=100):
        
        # plot fluxes at a particular window in time (determined by window_size and window_n). 
        # Plot true flux + pred flux

        fig = plt.figure(figsize=(15,5))
        plt.plot(self.true_flux[window_size*window_n:window_size*(1+window_n)], label="truth")
        plt.plot(self.pred_flux[window_size*window_n:window_size*(1+window_n)], label="preds")

        plt.legend()

    
    def __len__(self):
        return self.inputs.size()[0]

    def __getitem__(self, item):
        return self.inputs[item,:,:], self.fp[item,:,:]  
    


class BoundaryDataset(Dataset):
    """
    Creates dataset to pass to the model, applying any transformations specified. 
    Parameters:
    - inputs: array of shape (samples, sizexsize, features) with extracted inputs
    - fp: array of shape (samples, sizexsize) with footprint
    - input_names: list of dictionaries with variable names and info. each variable entry should have at least {"var":var_name} format, or {"var":var_name, "level":level} if required
    - input_transforms: list containing all transforms to apply to the inputs. Current valid options are:
        - clever_transform (standardise each variable separately across all levels and times)
        - clever_transform_3(standardise each variable and level separately across all times)
    - output_transforms: list containing all transforms to apply to the outputs, in order. Current valid options are:
        - boxcox (apply pixel-wise boxcox)
        - boxcox_all (apply boxcox to all the data)
        - mu-law (apply mu-law enconding algorithm)
        - logv3 (take log10 and shift so mean of non-zero elements is zero)
    - transform_parameters: pass optional parameters to transform. Only used during training (during test, only parameters in test_mode are considering). Pass in format {"transform_name":{"param1":value, "param2":value}}
    
    - test_mode: dict. If training, leave empty. If testing (ie applying existing parameters and/or already fitted models), pass a dictionary or the transform_parameters of another dataset

    Example:
    training_ds = FootprintsDataset(inputs, data.fp_data, input_transforms=["clever_transform"],output_transforms=["boxcox_all"], input_names=names)
    testing_ds = FootprintsDataset(test_inputs, test_data.fp_data, input_transforms=["clever_transform"],output_transforms=["boxcox_all"], input_names=names, test_mode=training_ds.transform_parameters))

    Functions:
    - inverse_transform(predictions): provides the footprints reconverted to the original space, inverting any previously applied transforms
    - add_prototypes(prototypes) : unfinished! Takes prototypes of same shape as footprints, applies to them the same transform applied to the fps and appends them to the inputs array.

    """
    def __init__(self, inputs,baseline_inputs, outputs, use_baselines = False,input_transforms = [], transform_parameters = {}, test_mode={}, input_names=[]):
    #def __init__(self, inputs, fp, input_transforms = [], output_transforms = [], transform_parameters = {}, test_mode={}, input_names=[], size=None, full_land_cover=False):
        #super().__init__()
        self.inputs = np.copy(inputs)
        self.baseline_inputs = baseline_inputs # Nawid - Input related to the baselinevalues
        self.outputs = outputs
        '''
        if size is None:
            self.size = [int(np.sqrt(np.shape(fp)[-1])), int(np.sqrt(np.shape(fp)[-1]))]
            print(f"assuming this is a square dataset of size {self.size[0]} x {self.size[1]}!")
        elif type(size) is int:
            assert size == int(np.sqrt(np.shape(fp)[-1])), f"you passed size as an int, which normally implies a square dataset of size {size} x {size}, but the data size doesnt match (data would have size {int(np.sqrt(np.shape(fp)[-1]))} x {int(np.sqrt(np.shape(fp)[-1]))}). are you sure the data is square and of the size you pass?"
            self.size = [size, size]
        elif len(size) == 2 and (type(size) is tuple or type(size) is list):
            self.size = size
            print(f"assuming a non-square domain of size {self.size}")
        else:
            raise ValueError("you passed a size parameter of an unknown format. pass size=None if the dataset is square, or a list/tuple")
        '''

        self.input_transforms= copy.deepcopy(input_transforms)
        '''
        self.output_transforms= copy.deepcopy(output_transforms)
        '''
        self.input_names = copy.deepcopy(input_names)
        self.test_mode = copy.deepcopy(test_mode)
        self.transform_parameters = copy.deepcopy(transform_parameters)
        
        print(transform_parameters)
        print(self.transform_parameters)

        # note that _Transform is the base class and will raise a not_implemented error if used
        self.valid_input_transforms = {
            "clever_transform":{"params":["transformers"], "fun":_CleverTransform}, 
            "clever_transform_2":{"params":["transformers"], "fun":_CleverTransform2}, 
            "clever_transform_3":{"params":["transformers"], "fun":_CleverTransform3},
            "standardise":{"params":["transformers"], "fun":_Transform}, 
            "scale":{"params":["inputs_min", "inputs_max"], "fun":_Transform}}

        ## assert that only valid input and output transforms have been passed
        assert set(self.input_transforms).issubset(self.valid_input_transforms.keys()), f"You passed some input transforms that are not in the list of valid transforms. \n The valid transforms are {list(self.valid_input_transforms.keys())}. \n The following transforms you passed but are not allowed: {set(self.input_transforms) - set(self.valid_input_transforms.keys())}"
        '''
        assert set(self.output_transforms).issubset(self.valid_output_transforms.keys()), f"You passed some input transforms that are not in the list of valid transforms. \n The valid transforms are {list(self.valid_output_transforms.keys())}. \n The following transforms you passed but are not allowed: {set(self.output_transforms) - set(self.valid_output_transforms.keys())}"
        '''
        if len(self.test_mode)>0:
            self.mode="test"
            # if on test mode, assert that all necessary parameters for the provided input and output transforms have been passed
            assert all([set(self.valid_input_transforms[x]["params"]).issubset(self.test_mode[x]) for x in self.input_transforms]), "You did not pass all the necessary parameters to test_mode for the input transformations!"
            '''
            assert all([set(self.valid_output_transforms[x]["params"]).issubset(self.test_mode[x]) for x in self.output_transforms]), "You did not pass all the necessary parameters to test_mode for the output transformations!"
            '''
        else:
            self.mode="train"        

        self.input_transforms = {key: None for key in self.input_transforms}
        for transform in self.input_transforms:
            self.input_transforms[transform]  = self.valid_input_transforms[transform]["fun"](self)
            self.input_transforms[transform].transform()

        '''
        self.output_transforms = {key: None for key in self.output_transforms}
        for transform in self.output_transforms:
            if transform in self.transform_parameters:
                print(self.transform_parameters)
                self.output_transforms[transform]  = self.valid_output_transforms[transform]["fun"](self, **self.transform_parameters[transform])
            else:
                self.output_transforms[transform]  = self.valid_output_transforms[transform]["fun"](self)

            self.fp = self.output_transforms[transform].transform(self.fp)
        '''
        # Repeat the 2D array along the second axis to match the shape of the 3D array
        # The result should have shape (2, 4, 3)
        
        baseline_expanded = np.repeat(self.baseline_inputs[:, np.newaxis, :], self.inputs.shape[1], axis=1)

        if use_baselines:
            print('USING BASELINES')
            self.inputs = np.concatenate((self.inputs,baseline_expanded),axis=-1)

        if type(self.inputs) != torch.Tensor:
            self.inputs = torch.tensor(self.inputs, dtype=torch.float)
            #self.inputs = torch.tensor(self.inputs, dtype=torch.float,requires_grad=False)
            # requires grad or false
            #self.inputs.requires_grad = False;
        
        if type(self.outputs) != torch.Tensor:
            self.outputs = torch.tensor(self.outputs, dtype=torch.float)

            #self.outputs = torch.tensor(self.outputs, dtype=torch.float,requires_grad=False)
        
    
    def inverse_transform(self, predictions, return_transformed=True):
        self.predictions=predictions
        if type(self.predictions) == torch.Tensor:
            self.predictions = self.predictions.detach().numpy()
            
        for transform in self.output_transforms:
            self.transformed_predictions = self.output_transforms[transform].inverse_transform(self.predictions)
        if return_transformed:
            return self.transformed_predictions
    
    def __len__(self):
        return self.inputs.size()[0]

    def __getitem__(self, item):
        return self.inputs[item,:,:], self.outputs[item,:]  

class FootprintsDatasetNANV3(FootprintsDataset):
    """
    LATEST VERSION
    prepare the inputs and outputs as a dataset to pass to the model, returns a mask of nans
    """

    def __init__(self, inputs, fp, input_transforms = [], output_transforms = [], transform_parameters = {}, test_mode={}, input_names=[],full_land_cover=False, returning="original_footprints", size=None):
        super().__init__(inputs, fp, input_transforms, output_transforms, transform_parameters, test_mode, input_names, size=size)

        if returning=="original_footprints":
            self.original_fp = torch.tensor(self.fp_untransformed, dtype=torch.float)
            self.original_fp = torch.nan_to_num(self.original_fp)

        elif returning == "transformed_footprints":
            self.original_fp = fp
        if self.original_fp.ndim==2:
            self.original_fp = self.original_fp[:,:,None]
        
        print("preparing nanmask")
        self.nanmask= ~torch.isnan(self.fp)
        print(f"there were {torch.sum(~self.nanmask)} nans")
        self.fp = torch.nan_to_num(self.fp)
        print(f"after, there are {torch.sum(torch.isnan(self.fp))} nans in the footprint (should be zero!)")

        


    def __getitem__(self, item):
        return self.inputs[item,:,:], self.fp[item,:,:], self.original_fp[item,:,:], self.nanmask[item,:,:]
        

class FootprintsDatasetV3(FootprintsDataset):
    """
    LATEST VERSION
    prepare the inputs and outputs as a dataset to pass to the model
    """

    def __init__(self, inputs, fp, input_transforms = [], output_transforms = [], transform_parameters = {}, test_mode={}, input_names=[],full_land_cover=False, returning="original_footprints", size=None):
        super().__init__(inputs, fp, input_transforms, output_transforms, transform_parameters, test_mode, input_names, size=size)

        if returning=="original_footprints":
            self.original_fp = torch.tensor(self.fp_untransformed, dtype=torch.float)
        elif returning == "transformed_footprints":
            self.original_fp = fp
        if self.original_fp.ndim==2:
            self.original_fp = self.original_fp[:,:,None]

    def __getitem__(self, item):
        return self.inputs[item,:,:], self.fp[item,:,:], self.original_fp[item,:,:]
    
class _Transform:
    """
    Base class for input and output transforms of data
    If transforming inputs
        - save untransformed inputs self.parent.inputs_untransformed = np.copy(self.parent.inputs)
        - keep transformed inputs in self.inputs
    If transforming outputs:
        - save untransformed footprint self.parent.fp_untransformed = np.copy(self.parent.fp)
        - keep transformed fp in self.fp   
    If inverse transforming outputs:
        - save original predictions self.parent.predictions = predictions
        - put transformed predictions under self.parent.transformed_predictions

    """
    def __init__(self, parent):
        self.parent = parent
    
    def transform(self):
        raise NotImplementedError("this transform has not been yet implemented!")

    def inverse_transform(self):
        raise NotImplementedError("this inverse transform has not been yet implemented!")

class _CleverTransform:
    """
    Apply a standard scaler to each variable across all levels and all timesteps

    If train, save scalers in a dictionary. Format example: {variable_name:scaler, variable_name:scaler, ...}

    If test, apply these

    If inverse_transform, apply saved transforms (either from test object or previous transforms) to transform the data back
    """
    def __init__(self, parent):
        self.parent = parent
        assert len(self.parent.input_names)>0, "Pass the input names to do a clever transform"
        self.parent.inputs_untransformed = np.copy(self.parent.inputs)

        self.varnames = np.array([x["var"] for x in self.parent.input_names])

        if self.parent.mode=="train":
            self.transformers = {}
        if self.parent.mode=="test":
            self.transformers = self.parent.test_mode["clever_transform"]["transformers"]

    def transform(self):
        for varname in np.unique(self.varnames):
            var_locations = np.where(self.varnames==varname)[0]
            if self.parent.mode=="train":
                scaler = preprocessing.StandardScaler()
                scaler.fit(self.parent.inputs[:,:,var_locations].flatten
            ().reshape(-1, 1))
                self.transformers[varname]=scaler 
            if self.parent.mode=="test":
                scaler = self.transformers[varname]

            shape = np.shape(self.parent.inputs[:,:,var_locations])
            self.parent.inputs[:,:,var_locations] = np.reshape(scaler.transform(self.parent.inputs[:,:,var_locations].flatten().reshape(-1, 1)), shape)  
    
        self.parent.transform_parameters["clever_transform"] = {}
        self.parent.transform_parameters["clever_transform"]["transformers"] = self.transformers  

class _CleverTransform2:
    """
    Apply a standard scaler to each variable and level, across all timesteps

    If train, save scalers in a dictionary, with subdictionaries for each level if needed. Format example: {variable_name:{levelA:scaler, levelB:scaler ...}, variable_name:scaler, ...}

    If test, apply these
    """
    def __init__(self, parent):
        self.parent = parent
        assert len(self.parent.input_names)>0, "Pass the input names to do a clever transform"
        self.parent.inputs_untransformed = np.copy(self.parent.inputs)

        self.varnames = np.array([x["var"] for x in self.parent.input_names])

        if self.parent.mode=="train":
            self.transformers = {}
        if self.parent.mode=="test":
            self.transformers = self.parent.test_mode["clever_transform_2"]["transformers"]

        self.vars_to_ignore = ["sin_time_year", "cos_time_year"]

    def transform(self):
        for varname in np.unique(self.varnames):
            if varname not in self.vars_to_ignore:
                var_locations = np.where(self.varnames==varname)[0]
                # apply below if variable has levels
                if "level" in self.parent.input_names[var_locations[0]]:
                    levels_dict = {}
                    levels = np.array([self.parent.input_names[x]["level"] for x in var_locations])
                    for level in np.unique(levels):
                        levels_locations = np.where(levels==level)[0]
                        if self.parent.mode=="train":
                            scaler = preprocessing.StandardScaler()
                            scaler.fit(self.parent.inputs[:,:,var_locations[levels_locations]].flatten
                        ().reshape(-1, 1))
                            levels_dict[level]=scaler
                        if self.parent.mode=="test":
                            scaler = self.transformers[varname][level]
                            
                        shape = np.shape(self.parent.inputs[:,:,var_locations[levels_locations]])
                        self.parent.inputs[:,:,var_locations[levels_locations]] = np.reshape(scaler.transform(self.parent.inputs[:,:,var_locations[levels_locations]].flatten
                        ().reshape(-1, 1)), shape)

                    if self.parent.mode=="train":
                        self.transformers[varname]=levels_dict

                # apply below if variable has no levels
                else:
                    if self.parent.mode=="train":
                        scaler = preprocessing.StandardScaler()
                        scaler.fit(self.parent.inputs[:,:,var_locations].flatten
                    ().reshape(-1, 1))
                        self.transformers[varname]=scaler 
                    if self.parent.mode=="test":
                        scaler = self.transformers[varname]

                    shape = np.shape(self.parent.inputs[:,:,var_locations])
                    self.parent.inputs[:,:,var_locations] = np.reshape(scaler.transform(self.parent.inputs[:,:,var_locations].flatten().reshape(-1, 1)), shape)    
            else:
                print(f"ignoring var {varname} and leaving as is")

        self.parent.transform_parameters["clever_transform_2"] = {}
        self.parent.transform_parameters["clever_transform_2"]["transformers"] = self.transformers      


class _CleverTransform3:
    """
    SAME as clever transform 2 but applies minmax to land cover and ignores full landcover 
    Apply a standard scaler to each variable and level, across all timesteps

    If train, save scalers in a dictionary, with subdictionaries for each level if needed. Format example: {variable_name:{levelA:scaler, levelB:scaler ...}, variable_name:scaler, ...}

    If test, apply these
    """
    def __init__(self, parent):
        self.parent = parent
        assert len(self.parent.input_names)>0, "Pass the input names to do a clever transform"
        self.parent.inputs_untransformed = np.copy(self.parent.inputs)

        self.varnames = np.array([x["var"] for x in self.parent.input_names])

        if self.parent.mode=="train":
            self.transformers = {}
        if self.parent.mode=="test":
            self.transformers = self.parent.test_mode["clever_transform_3"]["transformers"]

        self.vars_to_ignore = ["sin_time_year", "cos_time_year"]

    def transform(self):
        for varname in np.unique(self.varnames):
            if varname not in self.vars_to_ignore and "land_cover" not in varname:
                var_locations = np.where(self.varnames==varname)[0]
                # apply below if variable has levels
                if "level" in self.parent.input_names[var_locations[0]]:
                    levels_dict = {}
                    levels = np.array([self.parent.input_names[x]["level"] for x in var_locations])
                    for level in np.unique(levels):
                        levels_locations = np.where(levels==level)[0]
                        if self.parent.mode=="train":
                            scaler = preprocessing.StandardScaler()
                            scaler.fit(self.parent.inputs[:,:,var_locations[levels_locations]].flatten
                        ().reshape(-1, 1))
                            levels_dict[level]=scaler
                        if self.parent.mode=="test":
                            scaler = self.transformers[varname][level]
                            
                        shape = np.shape(self.parent.inputs[:,:,var_locations[levels_locations]])
                        self.parent.inputs[:,:,var_locations[levels_locations]] = np.reshape(scaler.transform(self.parent.inputs[:,:,var_locations[levels_locations]].flatten
                        ().reshape(-1, 1)), shape)

                    if self.parent.mode=="train":
                        self.transformers[varname]=levels_dict

                # apply below if variable has no levels
                else:
                    if self.parent.mode=="train":
                        scaler = preprocessing.StandardScaler()
                        scaler.fit(self.parent.inputs[:,:,var_locations].flatten
                    ().reshape(-1, 1))
                        self.transformers[varname]=scaler 
                    if self.parent.mode=="test":
                        scaler = self.transformers[varname]

                    shape = np.shape(self.parent.inputs[:,:,var_locations])
                    self.parent.inputs[:,:,var_locations] = np.reshape(scaler.transform(self.parent.inputs[:,:,var_locations].flatten().reshape(-1, 1)), shape)    
            else:
                if varname == "land_cover" or varname=="landcover":
                    var_locations = np.where(self.varnames==varname)[0]
                    print("landcover  bit!")
                    print(np.max(self.parent.inputs[:,:,var_locations]))
                    if self.parent.mode=="train":
                        scaler = preprocessing.MinMaxScaler()
                        scaler.fit(self.parent.inputs[:,:,var_locations].flatten
                    ().reshape(-1, 1))
                        self.transformers[varname]=scaler 
                    if self.parent.mode=="test":
                        scaler = self.transformers[varname]

                    shape = np.shape(self.parent.inputs[:,:,var_locations])
                    self.parent.inputs[:,:,var_locations] = np.reshape(scaler.transform(self.parent.inputs[:,:,var_locations].flatten().reshape(-1, 1)), shape)            
                    print(np.max(self.parent.inputs[:,:,var_locations]))   
                    print(var_locations)    

                else:
                    print(f"ignoring var {varname} and leaving as is")

        self.parent.transform_parameters["clever_transform_3"] = {}
        self.parent.transform_parameters["clever_transform_3"]["transformers"] = self.transformers      



class _Boxcox(_Transform):
    """
    Apply boxcox + standardisation to all pixels in image. 
    Note: Does NOT translate across domain sizes (ie fp size in training has to be the same as fp size in testing)

    If train, train and transform

    If test, use trained object to transform
    """
    def __init__(self, parent):
        self.parent = parent
        print("init boxcox")
        self.parent.fp_untransformed = np.copy(self.parent.fp)

        if self.parent.mode=="train":
            self.boxcox = preprocessing.PowerTransformer(method='box-cox', standardize=True)
            self.boxcox.fit(0.0000001+np.squeeze(self.parent.fp))

        elif self.parent.mode=="test":
            self.boxcox =  self.parent.test_mode["boxcox"]["transformers"] 

        self.parent.transform_parameters["boxcox"] = {}
        self.parent.transform_parameters["boxcox"]["transformers"] = self.boxcox

    def transform(self, fp):
        print("transforming")
        fp = self.boxcox.transform(0.0000001+np.squeeze(fp))  
        return fp    

    def inverse_transform(self, predictions):
        self.parent.predictions = predictions
        transformed_predictions=self.boxcox.inverse_transform(self.parent.predictions)-0.0000001
        return transformed_predictions

class _DistanceBoxcox(_Transform):
    """
    Apply boxcox + standardisation. A boxcox tranformer is fit for all non-zero datapoints that are at the same distance from the centre (distance is euclidean in pixels, rounded to nearest int)
    Note: Does NOT translate across domain sizes

    Parameter zero_shift indicates the shift to apply to original zeros - as these aren't transformed, they will remain as zeros and therefore need to be moved "out of the way" so they are not confused with the original 

    If train, train and transform

    If test, use trained object to transform
    """
    def __init__(self, parent, zero_shift=-5):
        self.parent = parent
        
        print("note that this WILL NOT work for non-square data!")
        print("init distance boxcox")
        self.parent.fp_untransformed = np.copy(self.parent.fp)
        self.zero_shift=zero_shift

        if self.parent.mode=="train":
            centre = int(self.parent.size[0]/2)
            # calculate distances between all cells and measurement cell
            X, Y = np.meshgrid(np.arange(self.parent.size[0]), np.arange(self.parent.size[0]))
            self.distances = np.sqrt((centre - X)**2 + (centre - Y)**2)
            self.distances = self.distances.flatten()
            # calculate boxcox for all datapoints that are at the same distance from the release point
            # distances are rounded for smoothness
            self.transformers = {}
            for dist in np.unique(self.distances.round(decimals=0)):
                dist_idxs = np.where(self.distances.round(decimals=0)==dist)
                pt = preprocessing.PowerTransformer(method="box-cox", standardize=True)
                pt.fit(self.parent.fp[:,dist_idxs].flatten()[self.parent.fp[:,dist_idxs].flatten()>0].reshape(-1, 1))
                self.transformers[dist] = pt

        elif self.parent.mode=="test":
            self.transformers =  self.parent.test_mode["distance_boxcox"]["transformers"] 
            self.distances = self.parent.test_mode["distance_boxcox"]["distances"] 
            self.zero_shift = self.parent.test_mode["distance_boxcox"]["zero_shift"] 

        self.parent.transform_parameters["distance_boxcox"] = {}
        self.parent.transform_parameters["distance_boxcox"]["transformers"] = self.transformers
        self.parent.transform_parameters["distance_boxcox"]["distances"] = self.distances
        self.parent.transform_parameters["distance_boxcox"]["zero_shift"] = self.zero_shift

    def transform(self, fp):
        print("transforming")
        fp_untransformed = np.copy(fp)
        fp = np.zeros_like(fp) + self.zero_shift

        for dist in np.unique(self.distances.round(decimals=0)):
            dist_idxs = np.where(self.distances.round(decimals=0)==dist)[0]
            pt = self.transformers[dist]
            for w in dist_idxs:
                non_zero_idxs = np.squeeze(fp_untransformed[:,w]>0)
                if np.sum(non_zero_idxs)>0:
                    fp[:,w][non_zero_idxs] = np.squeeze(pt.transform(fp_untransformed[:,w][non_zero_idxs].reshape(-1, 1)))    
        return fp    

    def inverse_transform(self, predictions):
        self.parent.predictions = predictions
        transformed_predictions = np.zeros_like(predictions)

        for dist in np.unique(self.distances.round(decimals=0)):
            dist_idxs = np.where(self.distances.round(decimals=0)==dist)[0]
            pt = self.transformers[dist]
            for w in dist_idxs:
                if self.zero_shift>0:
                    non_zero_idxs = np.squeeze(predictions[:,w]<0.75*self.zero_shift)
                if self.zero_shift<0:
                    non_zero_idxs = np.squeeze(predictions[:,w]>0.75*self.zero_shift)

                if np.sum(non_zero_idxs)>0:
                    transformed_predictions[:,w][non_zero_idxs] = np.squeeze(pt.inverse_transform(predictions[:,w][non_zero_idxs].reshape(-1, 1)))                

        return transformed_predictions

class _BoxcoxAll(_Transform):
    """
    Apply boxcox + standardisation to all pixels in image. 
    Note: Translates across domain sizes - training dataset can have different fp size to testing

    If train, train and transform

    If test, use trained object to transform
    """
    def __init__(self, parent):
        self.parent = parent
        print("init boxcox all")
        self.parent.fp_untransformed = np.copy(self.parent.fp)

        if self.parent.mode=="train":
            self.boxcox = preprocessing.PowerTransformer(method='box-cox', standardize=True)
            self.boxcox.fit(0.0000001+np.squeeze(self.parent.fp.flatten()).reshape(-1, 1))

        elif self.parent.mode=="test":
            self.boxcox =  self.parent.test_mode["boxcox_all"]["transformers"]

        self.parent.transform_parameters["boxcox_all"] = {}
        self.parent.transform_parameters["boxcox_all"]["transformers"] = self.boxcox

    def transform(self, fp):
        print("transforming")
        fp = self.boxcox.transform(0.0000001+np.squeeze(fp.flatten()).reshape(-1, 1))
        fp = np.reshape(fp, np.shape(self.parent.fp_untransformed))   
        return fp

    def inverse_transform(self, predictions):
        self.parent.predictions = predictions
        transformed_predictions=self.boxcox.inverse_transform(self.parent.predictions.flatten().reshape(-1, 1))-0.0000001
        transformed_predictions = np.reshape(transformed_predictions, np.shape(self.parent.predictions)) 
        return transformed_predictions

class _LogFootNet(_Transform):
    """
    Takes log of (output data+0.0000001) (ofset is needed so data is strictly positive) and shifts so mean(logged(non-zero values)) = 0
    Note: translates across domain sizes
    """

    def __init__(self, parent):
        self.parent = parent
        print("init logv3")
        self.parent.fp_untransformed = np.copy(self.parent.fp)    

        if self.parent.mode=="train":
            self.logged_mean = np.mean(np.log10(self.parent.fp.flatten()[self.parent.fp.flatten()>0]))
        if self.parent.mode=="test":
            self.logged_mean = self.parent.test_mode["logv3"]["logged_mean"]
        
        self.parent.transform_parameters["logv3"] = {}
        self.parent.transform_parameters["logv3"]["logged_mean"] = self.logged_mean

    def transform(self, fp):
        logged = np.log10(self.parent.fp+0.0000001)
        fp = logged + abs(self.logged_mean) 
        return fp
    
    def inverse_transform(self, predictions):
        self.parent.predictions = predictions
        transformed_predictions = 10**(predictions - abs(self.logged_mean))-0.0000001      
        return transformed_predictions   
    
class _LogV3e(_Transform):
    """
    Takes log of (output data+0.0000001) (ofset is needed so data is strictly positive) and shifts so mean(logged(non-zero values)) = 0
    Note: translates across domain sizes
    """

    def __init__(self, parent):
        self.parent = parent
        print("init logv3")
        self.parent.fp_untransformed = np.copy(self.parent.fp)    

        if self.parent.mode=="train":
            self.logged_mean = np.mean(np.log(self.parent.fp.flatten()[self.parent.fp.flatten()>0]))
        if self.parent.mode=="test":
            self.logged_mean = self.parent.test_mode["logv3e"]["logged_mean"]
        
        self.parent.transform_parameters["logv3e"] = {}
        self.parent.transform_parameters["logv3e"]["logged_mean"] = self.logged_mean

    def transform(self, fp):
        logged = np.log(self.parent.fp+0.0000001)
        fp = logged + abs(self.logged_mean) 
        return fp
    
    def inverse_transform(self, predictions):
        self.parent.predictions = predictions
        transformed_predictions = np.exp(predictions - abs(self.logged_mean))-0.0000001      
        return transformed_predictions        
    
class _LogV3(_Transform):
    """
    Takes log of (output data+0.0000001) (ofset is needed so data is strictly positive) and shifts so mean(logged(non-zero values)) = 0
    Note: translates across domain sizes
    """

    def __init__(self, parent):
        self.parent = parent
        print("init logv3")
        self.parent.fp_untransformed = np.copy(self.parent.fp)    

        if self.parent.mode=="train":
            self.logged_mean = np.mean(np.log10(self.parent.fp.flatten()[self.parent.fp.flatten()>0]))
        if self.parent.mode=="test":
            self.logged_mean = self.parent.test_mode["logv3"]["logged_mean"]
        
        self.parent.transform_parameters["logv3"] = {}
        self.parent.transform_parameters["logv3"]["logged_mean"] = self.logged_mean

    def transform(self, fp):
        logged = np.log10(self.parent.fp+0.0000001)
        fp = logged + abs(self.logged_mean) 
        return fp
    
    def inverse_transform(self, predictions):
        self.parent.predictions = predictions
        transformed_predictions = 10**(predictions - abs(self.logged_mean))-0.0000001      
        return transformed_predictions        

class _LogV4(_Transform):
    """
    Takes log of output data where non-zero, and offsets by minimum value so its above
    Note: translates across domain sizes
    """

    def __init__(self, parent, minimum_oom=5):
        self.parent = parent
        print("init logv4")
        self.parent.fp_untransformed = np.copy(self.parent.fp)    
        self.minimum_oom = minimum_oom
        
        self.parent.transform_parameters["logv4"] = {}

    def transform(self, fp):
        logged = np.copy(self.parent.fp)
        logged[logged>0] = self.minimum_oom+np.log10(logged[logged>0])
        logged[logged<0] = 0 
        return logged
    
    def inverse_transform(self, predictions):
        transformed_predictions = np.copy(predictions)
        self.parent.predictions = predictions
        
        # if using relu this shouldnt be necessary but just in case
        transformed_predictions[transformed_predictions <0] = 0
        transformed_predictions[transformed_predictions>0] = 10**(transformed_predictions[transformed_predictions>0]-self.minimum_oom)
        return transformed_predictions        


class _MixedLog(_Transform):

    def __init__(self, parent):
        self.parent = parent
        print("init mixed log")
        self.parent.fp_untransformed = np.copy(self.parent.fp)    

    def transform(self, fp):
        c1 = 0
        self.c2 = -1.5
        self.min_val = 1e-6
        self.maxval = 1
        x = c1*np.minimum(fp, self.maxval) + self.c2*((np.log(np.maximum(fp, self.min_val))-np.log(self.min_val))/np.log(self.min_val))
        return x

    def inverse_transform(self, predictions):
        self.parent.predictions = predictions
        transformed_predictions = np.copy(predictions)
        transformed_predictions[predictions>0] = np.exp((np.log(self.min_val)/self.c2)*predictions[predictions>0] + np.log(self.min_val))  
        return transformed_predictions        

class _MuLaw(_Transform):
    """
    Applies mu-law encoding, a signal compression algorithm (https://en.wikipedia.org/wiki/M-law_algorithm)
    Formula is F(x)=sgn(x) ln(1 + mu*abs(x)/scale) / ln(1+mu)
    
    Parameter scale can be any positive value - scale=1 means no scaling is done to the data, scaling="max" rescales the data to the 0-1 range

    Translates across domain sizes



    """
    def __init__(self, parent, mu=256, scale=1):
        self.parent = parent
        print("init mulaw")
        self.parent.fp_untransformed = np.copy(self.parent.fp) 
        if self.parent.mode=="train":
            self.mu = mu
            self.scale = scale
        if self.parent.mode=="test":
            self.scale = self.parent.test_mode["mu-law"]["scale"]
            self.mu = self.parent.test_mode["mu-law"]["mu"]

        if self.scale=="max":
            self.scale = np.max(self.parent.fp)

        self.parent.transform_parameters["mu-law"] = {}
        self.parent.transform_parameters["mu-law"]["mu"] = self.mu
        self.parent.transform_parameters["mu-law"]["scale"] = self.scale       
    
    def transform(self, fp):
        fp = np.sign(fp)*np.log(1+self.mu*(np.abs(fp/self.scale)))/(np.log(1+self.mu))
        return fp
    
    def inverse_transform(self, predictions):
        self.parent.predictions = predictions
        transformed_predictions = self.scale * np.sign(predictions) * ((1+self.mu)**(np.abs(predictions))-1)/self.mu 
        return transformed_predictions   


def plot_footprints(idx, original_fps=None, transformed_fps=None, predictions=None, transformed_predictions=None, size=30, log=False):
    """
    visualise footprints and predictions for the footprints at index idx
    idx can be an int or a list of ints
    pass any combination of original_fps, transformed_fps, predictions and transformed_predictions to plot
    """

    assert type(idx) is int or type(idx) is list, "idx should be an int or list of ints"

    if type(idx) is int:
        idx=[idx]
    
    to_plot = (original_fps is not None) + (transformed_fps is not None) + (predictions is not None) + (transformed_predictions is not None)

    assert to_plot>0, "You passed none of the four original_fps, transformed_fps, predictions or transformed_predictions so nothing will be plotted!"
    fig, ax = plt.subplots((len(idx)), to_plot, figsize=(4*to_plot,4*len(idx)))
    if to_plot==1:
        ax = np.array([ax])
    if len(idx)==1:
        ax = ax[None,:]
    
    for idx_n, index in enumerate(idx):
        ax_counter = 0
        if original_fps is not None:
            if log:
                ax[idx_n,ax_counter].imshow(np.reshape(np.log10(original_fps[index]), (size, size)), origin="lower")
            else:
                ax[idx_n,ax_counter].imshow(np.reshape(original_fps[index], (size, size)), origin="lower")
            ax_counter+=1
        if transformed_predictions is not None:
            if log:
                ax[idx_n,ax_counter].imshow(np.reshape(np.log10(transformed_predictions[index]),(size, size)), origin="lower")
            else:
                ax[idx_n,ax_counter].imshow(np.reshape(transformed_predictions[index],(size, size)), origin="lower")
            ax_counter+=1
        if transformed_fps is not None:
            ax[idx_n,ax_counter].imshow(np.reshape(transformed_fps[index], (size, size)), origin="lower")
            ax_counter+=1
        if predictions is not None:
            ax[idx_n,ax_counter].imshow(np.reshape(predictions[index], (size, size)), origin="lower")
            ax_counter+=1

        ax[idx_n,0].set_ylabel(f"fp at index {index}")

    ax_counter = 0
    if original_fps is not None:
        ax[0,ax_counter].set_title("True Footprint \n original space")
        ax_counter+=1
    if  transformed_predictions is not None:
        ax[0,ax_counter].set_title("Predicted Footprint \n original space")
        ax_counter+=1
    if transformed_fps is not None:
        ax[0,ax_counter].set_title("True Footprint \n transformed space")
        ax_counter+=1
    if predictions is not None:
        ax[0,ax_counter].set_title("Predicted Footprint \n transformed space")
        ax_counter+=1


    for axis in ax.flatten():
        axis.tick_params(left = False, bottom = False, labelbottom=False, labelleft=False) 
        #axis.tick_params(axis='y', colors='white')   

    fig.patch.set_facecolor('white')



def predict_fluxes(true_fp, pred_fp, flux, units_transform = "default"):
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


def predict_fluxes_3D(true_fp, pred_fp, flux, units_transform = "default"):
    ## convolute predicted footprints and fluxes, returns two np arrays, one with the true flux and one with the emulated flux, of shape (n_footprints,)
    ## flux is an array, regridded and cut to the same resolution and size of the footprints
    ## units_transform can be None (use fluxes directly), "default" (performs flux*1e3 / CH4molarmass) or another function (which should return an array of the same shape as the original flux)

    assert len(np.shape(flux))==3 and len(np.shape(true_fp))==3 and len(np.shape(pred_fp))==3, "the fps and fluxes all have to be 3D"

    if units_transform != None:
        if units_transform == "default":
            # for edgar
            molarmass = 16.0425
            flux = flux*1e3 / molarmass
        else:
            flux = units_transform(flux)
    true_concentration = true_fp*flux
    print(np.shape(true_flux), np.shape(flux), np.shape(true_concentration))
    true_flux = np.sum(true_concentration, axis = (1,2))
    pred_concentration = pred_fp*flux
    pred_flux = np.sum(pred_concentration, axis = (1,2))
    
    return true_flux, pred_flux
    
def NMAE(pred, target, mode="all"):
    pred = pred.detach().numpy() if torch.is_tensor(pred) else pred
    target = target.detach().numpy() if torch.is_tensor(target) else target
    if mode=="per cell":
        return np.mean(mean_absolute_error(target, pred, multioutput="raw_values")/(np.mean(target, axis=0)))
    if mode=="all":
        return np.mean(mean_absolute_error(target, pred)/(np.mean(target))) 

       

def NMAE_nans(pred, target, mode="all"):
    pred = pred.detach().numpy() if torch.is_tensor(pred) else pred
    target = target.detach().numpy() if torch.is_tensor(target) else target
    if mode=="per cell":
        return np.mean(np.nanmean(abs(target - pred), axis=0)/(np.nanmean(target, axis=0)))
    if mode=="all":
        if len(np.shape(target))==2:
            return np.nanmean(np.nanmean(abs(target - pred), axis=1)/(np.nanmean(target, axis=1)))
        else:
            return np.nanmean(np.nanmean(abs(target - pred), axis=(1,2))/(np.nanmean(target, axis=(1,2))))



