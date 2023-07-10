"""

The dataloader has to do a few things for the model to work correctly

1. Load the land-0sea mask, orography dataset, regridded from 0.1 to the correct resolution
2. Calculate the top-of-atmosphere solar radiation for each location at fcurrent time and 10 other
 times +- 12 hours
3. Add day-of-year, sin(lat), cos(lat), sin(lon), cos(lon) as well
3. Batch data as either in geometric batches, or more normally
4. Rescale between 0 and 1, but don't normalize

"""

#import const
#import numpy as np
#import pandas as pd
#import torchvision.transforms as transforms
#import xarray as xr
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import sklearn.preprocessing as preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler



class FootprintsDataset(Dataset):
    # should probably merge this with data? or with get inputs
    """
    options right now:
    - train mode 
        pass only data to set up/transform
    - test mode
        pass data + test_mode dictionary with the fitted transformers from the train set
    
    - standardise: apply (feature-mean)/std feature-wise 
    - scale: scale to -1, 1 range. If input_names is passed, the inputs with PBLH and temp grad (which have an exponential distribution rather than normal) are normalised with boxcox before scaling
    - transform output: if "same", do standardise, if "boxcox", do boxcox
    - scale output: scale to range (not implemented!!)
    """

    def __init__(self, inputs, fp, standardise=False, scale=False, transform_output=False, scale_output=False, test_mode={}, feature_dim=11, aux_dim=5, clever_transform=False, clever_transform_2=False, input_names=[], standardise_all=False, device="cpu", zeroing=False, binary_fp=None, transform_parameters={}):
        super().__init__()  
        self.inputs = inputs
        self.fp = fp
        self.standardise_inputs=False
        self.transform_output=False
        self.mode="train"
        self.scale_output=False
        self.zeroing=zeroing


        if len(test_mode)>0:
            self.mode="test"
            if clever_transform:
                assert len(input_names)>0, "Pass the input names to do a clever transform"
                self.clever = True
                self.inputs_untransformed = np.copy(inputs)
                varnames = np.array([x["var"] for x in input_names])
                try:
                    transformers = test_mode["clever_transformers"]
                    self.inputs = np.zeros_like(inputs)
                    for varname in np.unique(varnames):
                        #print(varname)
                        var_locations = np.where(varnames==varname)
                        scaler = transformers[varname]
                        shape = np.shape(inputs[:,:,var_locations])
                        self.inputs[:,:,var_locations] = np.reshape(scaler.transform(inputs[:,:,var_locations].flatten().reshape(-1, 1)), shape)
                except KeyError:
                    print("pass clever transformers")

            if clever_transform_2:
                assert len(input_names)>0, "Pass the input names to do a clever transform"
                self.clever = True
                self.inputs_untransformed = np.copy(inputs)
                varnames = np.array([x["var"] for x in input_names])
                try:
                    transformers = test_mode["clever_transformers"]
                    for varname in np.unique(varnames):
                        var_locations = np.where(varnames==varname)[0]
                        #print(varname, var_locations)
                        if "level" in input_names[var_locations[0]]:
                            levels = np.array([input_names[x]["level"] for x in var_locations])
                            for level in np.unique(levels):
                                levels_locations = np.where(levels==level)[0]
                                #print(levels_locations, var_locations, var_locations[levels_locations])
                                scaler = transformers[varname][level]
                                shape = np.shape(inputs[:,:,var_locations[levels_locations]])
                                self.inputs[:,:,var_locations[levels_locations]] = np.reshape(scaler.transform(inputs[:,:,var_locations[levels_locations]].flatten().reshape(-1, 1)), shape)
                        else:
                            #print("???")
                            shape = np.shape(inputs[:,:,var_locations])
                            #print(np.max(inputs[:,:,var_locations]), np.min(inputs[:,:,var_locations]))
                            scaler = transformers[varname]
                            self.inputs[:,:,var_locations] = np.reshape(scaler.transform(inputs[:,:,var_locations].flatten().reshape(-1, 1)), shape)
                            #print(np.max(inputs[:,:,var_locations]), np.min(inputs[:,:,var_locations]))


                except KeyError:
                    print("pass clever transformers")
           

            self.test_mode = test_mode
            if standardise:
                try:
                    self.standardise_inputs=True
                    self.inputs_untransformed = np.copy(inputs)
                    self.inputs[:,:,:feature_dim] = (self.inputs[:,:,:feature_dim]-test_mode["mean"])/test_mode["var"]
                    self.inputs_mean=test_mode["mean"]
                    self.inputs_var=test_mode["var"]
                except KeyError:
                    print("no info was passed to standardise inputs")
                    pass


            if standardise_all:
                try:

                    self.standardise_inputs=True
                    self.inputs_untransformed = np.copy(inputs)
                    scalers=test_mode["transformers"]
                    for n in range(np.shape(inputs)[-1]):  
                        shape = np.shape(inputs[:,:,n])
                        self.inputs[:,:,n] = np.reshape(scalers[n].transform(inputs[:,:,n].flatten().reshape(-1, 1)), shape)                       

                except KeyError:
                    print("no info was passed to standardise inputs")
                    pass

            if scale:
                try:
                    self.scale_inputs=True
                    self.inputs_untransformed = np.copy(inputs)
                    if len(input_names)>0:
                        rescale_differently = ["PBLH present", "PBLH past"]#, "temp grad"]
                        print(f"inputs {rescale_differently} will be tranformed to a normal distribution before rescaling")
                        for n, name in enumerate(input_names):
                            if name in rescale_differently:
                                self.inputs[:,:,n] = np.reshape(test_mode["input_boxcox"][name]["boxcox"].transform(test_mode["input_boxcox"][name]["shift"]+self.inputs_untransformed[:,:,n].flatten().reshape(-1, 1)), np.shape(self.inputs_untransformed[:,:,n]))   

                    self.inputs = -1 + (2*(self.inputs - test_mode["inputs_min"]))/(test_mode["inputs_max"] - test_mode["inputs_min"])                          
                except KeyError:
                    print("no info (or not enough info!) was passed to scale inputs")


            if transform_output=="same":
                self.transform_output="same"
                print("use this in weather mode only!!!")
                idx = 0
                try:
                    self.fp = (self.fp-test_mode["mean"][idx])/test_mode["var"][idx]           
                except KeyError:
                    print("no info was passed to standardise output")
                    pass 

            if transform_output=="boxcox":
                self.transform_output="boxcox"
                self.boxcox=test_mode["boxcox"]
                self.fp_untransformed = np.copy(fp)
                try:
                    self.fp = test_mode["boxcox"].transform(0.0000001+np.squeeze(self.fp))[:,:,None]
                    if zeroing:
                        self.fp[self.fp<0]=0
                        print("zeroing")
                except KeyError:
                    print("no info was passed to do boxcox on the output")
                    pass
            
            if transform_output=="mu-law":
                print("here!")
                self.transform_output="mu-law"
                self.fp_untransformed = np.copy(fp)
                self.transform_parameters=test_mode["transform_parameters"]
                assert "max" in list(self.transform_parameters.keys()) and "mu" in list(self.transform_parameters.keys()), "Transform parameters need to have max and mu"

                self.fp = np.log(1+self.transform_parameters["mu"]*(self.fp/self.transform_parameters["max"]))/(np.log(1+self.transform_parameters["mu"]))

                transform_parameters=self.transform_parameters
            


            if scale_output:
                self.scale_output=True
                try:
                    self.output_minmax=test_mode["output_minmax"]
                    self.fp = self.output_minmax.transform(np.squeeze(self.fp))[:,:,None]

                    #self.fp = -1 + (2*(self.fp - test_mode["output_min"]))/(test_mode["output_max"] - test_mode["output_min"])      
                except KeyError:
                    print("no info was passed to rescale the output")            
        else:
            if clever_transform:
                assert len(input_names)>0, "Pass the input names to do a clever transform"
                self.clever = True
                self.inputs_untransformed = np.copy(inputs)
                varnames = np.array([x["var"] for x in input_names])
                self.transformers = {}
                self.inputs = np.zeros_like(inputs)
                #print(varnames)
                for varname in np.unique(varnames):
                    #print(varname)
                    var_locations = np.where(varnames==varname)
                    scaler = preprocessing.StandardScaler()
                    shape = np.shape(inputs[:,:,var_locations])
                    #print(var_locations)
                    #print(np.shape(inputs[:,:,var_locations].flatten()))
                    self.inputs[:,:,var_locations] = np.reshape(scaler.fit_transform(inputs[:,:,var_locations].flatten().reshape(-1, 1)), shape)
                    self.transformers[varname]=scaler


            if clever_transform_2:
                assert len(input_names)>0, "Pass the input names to do a clever transform"
                self.clever = True
                self.inputs_untransformed = np.copy(inputs)
                varnames = np.array([x["var"] for x in input_names])
                self.transformers = {}
                self.inputs = np.zeros_like(inputs)
                #print(varnames)
                for varname in np.unique(varnames):
                    if "time" not in varname: # time variables are already normalised
                        var_locations = np.where(varnames==varname)[0]
                        if "level" in input_names[var_locations[0]]:
                            levels_dict = {}
                            levels = np.array([input_names[x]["level"] for x in var_locations])
                            for level in np.unique(levels):
                                levels_locations = np.where(levels==level)[0]
                                scaler = preprocessing.StandardScaler()
                                shape = np.shape(inputs[:,:,var_locations[levels_locations]])
                                self.inputs[:,:,var_locations[levels_locations]] = np.reshape(scaler.fit_transform(inputs[:,:,var_locations[levels_locations]].flatten
                                ().reshape(-1, 1)), shape)
                                #print(var_locations[levels_locations])
                                levels_dict[level]=scaler
                            self.transformers[varname]=levels_dict
                        else:
                            scaler = preprocessing.StandardScaler()
                            shape = np.shape(inputs[:,:,var_locations])
                            self.inputs[:,:,var_locations] = np.reshape(scaler.fit_transform(inputs[:,:,var_locations].flatten().reshape(-1, 1)), shape)
                            self.transformers[varname]=scaler          


            if standardise:
                self.standardise_inputs=True
                self.inputs_untransformed = np.copy(inputs)
                self.inputs_mean = np.mean(self.inputs[:,:,:feature_dim], axis=(0,1))
                self.inputs_var = np.var(self.inputs[:,:,:feature_dim], axis=(0,1))
                self.inputs[:,:,:feature_dim] = (self.inputs[:,:,:feature_dim]-self.inputs_mean)/self.inputs_var

            if standardise_all:
                self.standardise_inputs=True
                self.inputs_untransformed = np.copy(inputs)   
                self.transformers = []
                for n in range(np.shape(inputs)[-1]):                 
                    scaler = preprocessing.StandardScaler()
                    shape = np.shape(inputs[:,:,n])
                    self.inputs[:,:,n] = np.reshape(scaler.fit_transform(inputs[:,:,n].flatten().reshape(-1, 1)), shape)
                    self.transformers.append(scaler)               

            if scale:
                self.scale_inputs=True
                self.inputs_untransformed = np.copy(inputs)
                if len(input_names)>0:
                    rescale_differently = ["PBLH present", "PBLH past"]#, "temp grad"]
                    print(f"inputs {rescale_differently} will be tranformed to a normal distribution before rescaling")
                    self.input_boxcox = {}
                    for n, name in enumerate(input_names):
                        if name in rescale_differently:
                            pt = preprocessing.PowerTransformer(method='box-cox', standardize=False)
                            self.input_boxcox[name] = {"boxcox":pt, "shift":0.1+np.abs(np.min([0, np.min(self.inputs_untransformed[:,:,n])]))}
                            self.inputs[:,:,n] = np.reshape(self.input_boxcox[name]["boxcox"].fit_transform(self.input_boxcox[name]["shift"]+self.inputs_untransformed[:,:,n].flatten().reshape(-1, 1)), np.shape(self.inputs_untransformed[:,:,n]))                        
      
                #self.inputs_max = np.max(self.inputs, axis=(0,1))                
                #self.inputs_min = np.min(self.inputs, axis=(0,1)) 
                self.inputs_max = np.quantile(self.inputs, 0.997, axis=(0,1))                
                self.inputs_min = np.quantile(self.inputs, 0.003, axis=(0,1)) 
                # rescale all to -1, 1
                self.inputs = -1 + (2*(self.inputs - self.inputs_min))/(self.inputs_max - self.inputs_min)       

            if transform_output=="same":
                self.transform_output="same"
                print("use this in weather mode only!!!")
                idx = 0
                self.fp = (self.fp-self.inputs_mean[idx])/self.inputs_var[idx]

                #self.fp_mean = np.mean(self.fp, axis=(0,1))
                #self.fp_var = np.var(self.fp, axis=(0,1))            

            elif transform_output=="log":
                self.transform_output="log"
                self.fp_untransformed = np.copy(fp)
                logged = np.log(self.fp+1)
                self.logged=logged
                self.fp = (logged-np.mean(logged, axis=(0,1)))/np.var(logged, axis=(0,1))
                self.fp_mean = np.mean(logged, axis=(0,1))
                self.fp_var = np.var(logged, axis=(0,1))
            
            elif transform_output=="boxcox":
                self.transform_output="boxcox"
                self.fp_untransformed = np.copy(fp)
                pt = preprocessing.PowerTransformer(method='box-cox', standardize=True)
                self.boxcox = pt
                self.fp = self.boxcox.fit_transform(0.0000001+np.squeeze(self.fp))[:,:,None]
                if zeroing:
                    self.fp[self.fp<0]=0
                    print("zeroing")
                
            elif transform_output=="mu-law":
                self.transform_output="mu-law"
                self.fp_untransformed = np.copy(fp)       
                if "max" not in list(transform_parameters.keys()):
                    transform_parameters["max"] = 0.1
                if "mu" not in list(transform_parameters.keys()):
                    transform_parameters["mu"] = 256             
                self.fp = np.log(1+transform_parameters["mu"]*(self.fp/transform_parameters["max"]))/(np.log(1+transform_parameters["mu"]))

                self.transform_parameters=transform_parameters


            if scale_output:
                self.scale_output=True
                #self.output_max = np.quantile(self.fp, 0.997, axis=0)                
                #self.output_min = np.quantile(self.fp, 0.003, axis=0) 
                # rescale all to -1, 1
                #self.fp = -1 + (2*(self.fp - self.output_min))/(self.output_max - self.output_min)     
                self.output_minmax = MinMaxScaler(feature_range=(0,1))
                self.fp = self.output_minmax.fit_transform(np.squeeze(self.fp))[:,:,None]

        if type(self.inputs) != torch.Tensor:
            self.inputs = torch.tensor(self.inputs, dtype=torch.float)#, device=device)
        if type(self.fp) != torch.Tensor:
            self.fp = torch.tensor(self.fp, dtype=torch.float)#, device=device)
        if len(self.fp.size())==2:
            self.fp = self.fp[:,:,None]

        #print(self.inputs.size()[:-1], self.fp.size()[:-1])
        assert self.inputs.size()[:-1] == self.fp.size()[:-1], f"Shapes don't match up: inputs has shape {self.inputs.size()}, fp has shape {self.fp.size()}"


        if binary_fp is not None:
            print("adding binary fp", torch.squeeze(self.fp).size())

            self.include_binary_fp = True
            if type(binary_fp) != torch.Tensor:
                binary_fp = torch.tensor(binary_fp, dtype=torch.float)
            self.fp = torch.stack((torch.squeeze(self.fp), binary_fp))
            self.fp = self.fp.permute((1,2,0))
            #print(np.shape(self.fp))

    def __len__(self):
        return self.inputs.size()[0]

    def __getitem__(self, item):
        return self.inputs[item,:,:], self.fp[item,:,:]  
    
    def inverse_transform(self, predictions, remove_negs_before=True, remove_negs_after=True, remove_highs=False):
        if np.shape(predictions) != np.shape(self.fp_untransformed):
            print("careful, the predictions you inputted don't have the same shape as the footprints in this dataset!")
        if remove_highs==True and self.transform_output=="boxcox":
            predictions[predictions>3]=3
            predictions[predictions<0]=0
        self.predictions=predictions #.detach().numpy()
        if remove_negs_before==True and self.transform_output=="boxcox":
            predictions[predictions<0]=0
        if self.transform_output=="boxcox" and self.scale_output:
            #rescaled = (self.predictions-np.squeeze(self.test_mode["output_min"]))*(np.squeeze(self.test_mode["output_max"]) - np.squeeze(self.test_mode["output_min"]))/2
            #rescaled=np.squeeze(self.test_mode["output_min"]) + (self.predictions-np.squeeze(self.test_mode["output_min"]))*(np.squeeze(self.test_mode["output_max"]) - np.squeeze(self.test_mode["output_min"]))/2
            self.transformed_predictions=self.boxcox.inverse_transform(self.output_minmax.inverse_transform(predictions))-0.0000001
        elif self.transform_output=="boxcox" and not self.scale_output:
            self.transformed_predictions=self.boxcox.inverse_transform(self.predictions)-0.0000001
        if self.transform_output=="log":
            self.transformed_predictions=np.exp(((self.predictions)*self.fp_var+self.fp_mean))-1

        if self.transform_output=="mu-law":
            print("inverting")
            normalised = (1/self.transform_parameters["mu"])*(np.exp(predictions*(np.log(1+self.transform_parameters["mu"])))-1) 
            self.transformed_predictions = normalised*self.transform_parameters["max"]

        if remove_negs_after:
            self.transformed_predictions[self.transformed_predictions<0] = 0

        return self.transformed_predictions
    
    def prediction_error(self):
            try:
                self.transformed_predictions
            except AttributeError:
                print("only works currently for transformed/normalised outputs!")
            assert np.shape(self.transformed_predictions) == np.shape(self.fp_untransformed), "predictions and fp have to have the same shape to evaluate errors"
            nmaes = []
            for i in range(len(self.transformed_predictions)):
                nmaes.append(NMAE(self.transformed_predictions[i,:], self.fp_untransformed[i,:]))
            mse = mean_squared_error(self.transformed_predictions, self.fp_untransformed)

            print(f"NMAE: {np.mean(nmaes)}, MSE: {mse}")
            return np.mean(nmaes), mse

    def predict_fluxes(self, flux, units_transform = "default"):
        ## convolute predicted footprints and fluxes, returns two np arrays, one with the true flux and one with the emulated flux, of shape (n_footprints,)
        ## flux is an array, regridded and cut to the same resolution and size of the footprints
        ## units_transform can be None (use fluxes directly), "default" (performs flux*1e3 / CH4molarmass) or another function (which should return an array of the same shape as the original flux)
        try:
            self.transformed_predictions
        except AttributeError:
            print("only works currently for transformed/normalised outputs!")

        shape = int(np.sqrt(np.shape(self.transformed_predictions)[1])) 
        if units_transform != None:
            if units_transform == "default":
                molarmass = 16.0425
                flux = flux*1e3 / molarmass
            else:
                flux = units_transform(flux)
        true_concentration = np.reshape(self.fp_untransformed, (len(self.fp_untransformed), shape, shape))*flux
        self.true_flux = np.sum(true_concentration, axis = (1,2))
        pred_concentration = np.reshape(self.transformed_predictions, (len(self.transformed_predictions), shape, shape))*flux
        self.pred_flux = np.sum(pred_concentration, axis = (1,2))
        
        return self.true_flux, self.pred_flux


class SeqFootprintsDataset(Dataset):
    # should probably merge this with data? or with get inputs
    """
    options right now:
    - train mode 
        pass only data to set up/transform
    - test mode
        pass data + test_mode dictionary with the fitted transformers from the train set
    
    - standardise: apply (feature-mean)/std feature-wise 
    - scale: scale to -1, 1 range. If input_names is passed, the inputs with PBLH and temp grad (which have an exponential distribution rather than normal) are normalised with boxcox before scaling
    - transform output: if "same", do standardise, if "boxcox", do boxcox
    - scale output: scale to range (not implemented!!)
    """

    def __init__(self, inputs, fp, standardise=False, scale=False, transform_output=False, scale_output=False, test_mode={}, feature_dim=11, aux_dim=5, clever_transform=False, clever_transform_2=False, input_names=[], standardise_all=False, device="cpu", zeroing=False, binary_fp=None, transform_parameters={}):
        super().__init__()  
        self.inputs = inputs
        self.fp = fp
        self.standardise_inputs=False
        self.transform_output=False
        self.mode="train"
        self.scale_output=False
        self.zeroing=zeroing

        if len(test_mode)>0:
            self.mode="test"
            if clever_transform:
                assert len(input_names)>0, "Pass the input names to do a clever transform"
                self.clever = True
                self.inputs_untransformed = np.copy(inputs)
                varnames = np.array([x["var"] for x in input_names])
                try:
                    transformers = test_mode["clever_transformers"]
                    self.inputs = np.zeros_like(inputs)
                    for varname in np.unique(varnames):
                        #print(varname)
                        var_locations = np.where(varnames==varname)
                        scaler = transformers[varname]
                        shape = np.shape(inputs[:,:,var_locations])
                        self.inputs[:,:,var_locations] = np.reshape(scaler.transform(inputs[:,:,var_locations].flatten().reshape(-1, 1)), shape)
                except KeyError:
                    print("pass clever transformers")

            if clever_transform_2:
                assert len(input_names)>0, "Pass the input names to do a clever transform"
                self.clever = True
                self.inputs_untransformed = np.copy(inputs)
                varnames = np.array([x["var"] for x in input_names])
                try:
                    transformers = test_mode["clever_transformers"]
                    for varname in np.unique(varnames):
                        var_locations = np.where(varnames==varname)[0]
                        #print(varname, var_locations)
                        if "level" in input_names[var_locations[0]]:
                            levels = np.array([input_names[x]["level"] for x in var_locations])
                            for level in np.unique(levels):
                                levels_locations = np.where(levels==level)[0]
                                #print(levels_locations, var_locations, var_locations[levels_locations])
                                scaler = transformers[varname][level]
                                shape = np.shape(inputs[:,:,var_locations[levels_locations]])
                                self.inputs[:,:,var_locations[levels_locations]] = np.reshape(scaler.transform(inputs[:,:,var_locations[levels_locations]].flatten().reshape(-1, 1)), shape)
                        else:
                            #print("???")
                            shape = np.shape(inputs[:,:,var_locations])
                            #print(np.max(inputs[:,:,var_locations]), np.min(inputs[:,:,var_locations]))
                            scaler = transformers[varname]
                            self.inputs[:,:,var_locations] = np.reshape(scaler.transform(inputs[:,:,var_locations].flatten().reshape(-1, 1)), shape)
                            #print(np.max(inputs[:,:,var_locations]), np.min(inputs[:,:,var_locations]))


                except KeyError:
                    print("pass clever transformers")
           

            self.test_mode = test_mode
            if standardise:
                try:
                    self.standardise_inputs=True
                    self.inputs_untransformed = np.copy(inputs)
                    self.inputs[:,:,:feature_dim] = (self.inputs[:,:,:feature_dim]-test_mode["mean"])/test_mode["var"]
                    self.inputs_mean=test_mode["mean"]
                    self.inputs_var=test_mode["var"]
                except KeyError:
                    print("no info was passed to standardise inputs")
                    pass


            if standardise_all:
                try:

                    self.standardise_inputs=True
                    self.inputs_untransformed = np.copy(inputs)
                    scalers=test_mode["transformers"]
                    for n in range(np.shape(inputs)[-1]):  
                        shape = np.shape(inputs[:,:,n])
                        self.inputs[:,:,n] = np.reshape(scalers[n].transform(inputs[:,:,n].flatten().reshape(-1, 1)), shape)                       

                except KeyError:
                    print("no info was passed to standardise inputs")
                    pass

            if scale:
                try:
                    self.scale_inputs=True
                    self.inputs_untransformed = np.copy(inputs)
                    if len(input_names)>0:
                        rescale_differently = ["PBLH present", "PBLH past"]#, "temp grad"]
                        print(f"inputs {rescale_differently} will be tranformed to a normal distribution before rescaling")
                        for n, name in enumerate(input_names):
                            if name in rescale_differently:
                                self.inputs[:,:,n] = np.reshape(test_mode["input_boxcox"][name]["boxcox"].transform(test_mode["input_boxcox"][name]["shift"]+self.inputs_untransformed[:,:,n].flatten().reshape(-1, 1)), np.shape(self.inputs_untransformed[:,:,n]))   

                    self.inputs = -1 + (2*(self.inputs - test_mode["inputs_min"]))/(test_mode["inputs_max"] - test_mode["inputs_min"])                          
                except KeyError:
                    print("no info (or not enough info!) was passed to scale inputs")


            if transform_output=="same":
                self.transform_output="same"
                print("use this in weather mode only!!!")
                idx = 0
                try:
                    self.fp = (self.fp-test_mode["mean"][idx])/test_mode["var"][idx]           
                except KeyError:
                    print("no info was passed to standardise output")
                    pass 

            if transform_output=="boxcox":
                self.transform_output="boxcox"
                self.boxcox=test_mode["boxcox"]
                self.fp_untransformed = np.copy(fp)
                try:
                    self.fp = test_mode["boxcox"].transform(0.0000001+np.squeeze(self.fp))[:,:,None]
                    if zeroing:
                        self.fp[self.fp<0]=0
                        print("zeroing")
                except KeyError:
                    print("no info was passed to do boxcox on the output")
                    pass
            
            if transform_output=="mu-law":
                print("here!")
                self.transform_output="mu-law"
                self.fp_untransformed = np.copy(fp)
                self.transform_parameters=test_mode["transform_parameters"]
                assert "max" in list(self.transform_parameters.keys()) and "mu" in list(self.transform_parameters.keys()), "Transform parameters need to have max and mu"

                self.fp = np.log(1+self.transform_parameters["mu"]*(self.fp/self.transform_parameters["max"]))/(np.log(1+self.transform_parameters["mu"]))

                transform_parameters=self.transform_parameters
            


            if scale_output:
                self.scale_output=True
                try:
                    self.output_minmax=test_mode["output_minmax"]
                    self.fp = self.output_minmax.transform(np.squeeze(self.fp))[:,:,None]

                    #self.fp = -1 + (2*(self.fp - test_mode["output_min"]))/(test_mode["output_max"] - test_mode["output_min"])      
                except KeyError:
                    print("no info was passed to rescale the output")            
        else:
            if clever_transform:
                assert len(input_names)>0, "Pass the input names to do a clever transform"
                self.clever = True
                self.inputs_untransformed = np.copy(inputs)
                varnames = np.array([x["var"] for x in input_names])
                self.transformers = {}
                self.inputs = np.zeros_like(inputs)
                #print(varnames)
                for varname in np.unique(varnames):
                    #print(varname)
                    var_locations = np.where(varnames==varname)
                    scaler = preprocessing.StandardScaler()
                    shape = np.shape(inputs[:,:,var_locations])
                    #print(var_locations)
                    #print(np.shape(inputs[:,:,var_locations].flatten()))
                    self.inputs[:,:,var_locations] = np.reshape(scaler.fit_transform(inputs[:,:,var_locations].flatten().reshape(-1, 1)), shape)
                    self.transformers[varname]=scaler


            if clever_transform_2:
                assert len(input_names)>0, "Pass the input names to do a clever transform"
                self.clever = True
                self.inputs_untransformed = np.copy(inputs)
                varnames = np.array([x["var"] for x in input_names])
                self.transformers = {}
                self.inputs = np.zeros_like(inputs)
                #print(varnames)
                for varname in np.unique(varnames):
                    var_locations = np.where(varnames==varname)[0]
                    if "level" in input_names[var_locations[0]]:
                        levels_dict = {}
                        levels = np.array([input_names[x]["level"] for x in var_locations])
                        for level in np.unique(levels):
                            levels_locations = np.where(levels==level)[0]
                            scaler = preprocessing.StandardScaler()
                            shape = np.shape(inputs[:,:,var_locations[levels_locations]])
                            self.inputs[:,:,var_locations[levels_locations]] = np.reshape(scaler.fit_transform(inputs[:,:,var_locations[levels_locations]].flatten
                            ().reshape(-1, 1)), shape)
                            #print(var_locations[levels_locations])
                            levels_dict[level]=scaler
                        self.transformers[varname]=levels_dict
                    else:
                        scaler = preprocessing.StandardScaler()
                        shape = np.shape(inputs[:,:,var_locations])
                        self.inputs[:,:,var_locations] = np.reshape(scaler.fit_transform(inputs[:,:,var_locations].flatten().reshape(-1, 1)), shape)
                        self.transformers[varname]=scaler          


            if standardise:
                self.standardise_inputs=True
                self.inputs_untransformed = np.copy(inputs)
                self.inputs_mean = np.mean(self.inputs[:,:,:feature_dim], axis=(0,1))
                self.inputs_var = np.var(self.inputs[:,:,:feature_dim], axis=(0,1))
                self.inputs[:,:,:feature_dim] = (self.inputs[:,:,:feature_dim]-self.inputs_mean)/self.inputs_var

            if standardise_all:
                self.standardise_inputs=True
                self.inputs_untransformed = np.copy(inputs)   
                self.transformers = []
                for n in range(np.shape(inputs)[-1]):                 
                    scaler = preprocessing.StandardScaler()
                    shape = np.shape(inputs[:,:,n])
                    self.inputs[:,:,n] = np.reshape(scaler.fit_transform(inputs[:,:,n].flatten().reshape(-1, 1)), shape)
                    self.transformers.append(scaler)               

            if scale:
                self.scale_inputs=True
                self.inputs_untransformed = np.copy(inputs)
                if len(input_names)>0:
                    rescale_differently = ["PBLH present", "PBLH past"]#, "temp grad"]
                    print(f"inputs {rescale_differently} will be tranformed to a normal distribution before rescaling")
                    self.input_boxcox = {}
                    for n, name in enumerate(input_names):
                        if name in rescale_differently:
                            pt = preprocessing.PowerTransformer(method='box-cox', standardize=False)
                            self.input_boxcox[name] = {"boxcox":pt, "shift":0.1+np.abs(np.min([0, np.min(self.inputs_untransformed[:,:,n])]))}
                            self.inputs[:,:,n] = np.reshape(self.input_boxcox[name]["boxcox"].fit_transform(self.input_boxcox[name]["shift"]+self.inputs_untransformed[:,:,n].flatten().reshape(-1, 1)), np.shape(self.inputs_untransformed[:,:,n]))                        
      
                #self.inputs_max = np.max(self.inputs, axis=(0,1))                
                #self.inputs_min = np.min(self.inputs, axis=(0,1)) 
                self.inputs_max = np.quantile(self.inputs, 0.997, axis=(0,1))                
                self.inputs_min = np.quantile(self.inputs, 0.003, axis=(0,1)) 
                # rescale all to -1, 1
                self.inputs = -1 + (2*(self.inputs - self.inputs_min))/(self.inputs_max - self.inputs_min)       

            if transform_output=="same":
                self.transform_output="same"
                print("use this in weather mode only!!!")
                idx = 0
                self.fp = (self.fp-self.inputs_mean[idx])/self.inputs_var[idx]

                #self.fp_mean = np.mean(self.fp, axis=(0,1))
                #self.fp_var = np.var(self.fp, axis=(0,1))            

            elif transform_output=="log":
                self.transform_output="log"
                self.fp_untransformed = np.copy(fp)
                logged = np.log(self.fp+1)
                self.logged=logged
                self.fp = (logged-np.mean(logged, axis=(0,1)))/np.var(logged, axis=(0,1))
                self.fp_mean = np.mean(logged, axis=(0,1))
                self.fp_var = np.var(logged, axis=(0,1))
            
            elif transform_output=="boxcox":
                self.transform_output="boxcox"
                self.fp_untransformed = np.copy(fp)
                pt = preprocessing.PowerTransformer(method='box-cox', standardize=True)
                self.boxcox = pt
                self.fp = self.boxcox.fit_transform(0.0000001+np.squeeze(self.fp))[:,:,None]
                if zeroing:
                    self.fp[self.fp<0]=0
                    print("zeroing")
                
            elif transform_output=="mu-law":
                self.transform_output="mu-law"
                self.fp_untransformed = np.copy(fp)       
                if "max" not in list(transform_parameters.keys()):
                    transform_parameters["max"] = 0.1
                if "mu" not in list(transform_parameters.keys()):
                    transform_parameters["mu"] = 256             
                self.fp = np.log(1+transform_parameters["mu"]*(self.fp/transform_parameters["max"]))/(np.log(1+transform_parameters["mu"]))

                self.transform_parameters=transform_parameters


            if scale_output:
                self.scale_output=True
                #self.output_max = np.quantile(self.fp, 0.997, axis=0)                
                #self.output_min = np.quantile(self.fp, 0.003, axis=0) 
                # rescale all to -1, 1
                #self.fp = -1 + (2*(self.fp - self.output_min))/(self.output_max - self.output_min)     
                self.output_minmax = MinMaxScaler(feature_range=(0,1))
                self.fp = self.output_minmax.fit_transform(np.squeeze(self.fp))[:,:,None]


        self.present = []
        self.past = []
        self.notime = []
        for i, n in enumerate(input_names):
            if "time" in n.keys():
                if n["time"]=="t-0":
                    self.present.append(i)
                else:
                    self.past.append(i)
            else:
                self.notime.append(i)


        if type(self.inputs) != torch.Tensor:
            self.inputs = torch.tensor(self.inputs, dtype=torch.float)#, device=device)
        if type(self.fp) != torch.Tensor:
            self.fp = torch.tensor(self.fp, dtype=torch.float)#, device=device)
        if len(self.fp.size())==2:
            self.fp = self.fp[:,:,None]

        #print(self.inputs.size()[:-1], self.fp.size()[:-1])
        assert self.inputs.size()[:-1] == self.fp.size()[:-1], f"Shapes don't match up: inputs has shape {self.inputs.size()}, fp has shape {self.fp.size()}"


        if binary_fp is not None:
            print("adding binary fp", torch.squeeze(self.fp).size())

            self.include_binary_fp = True
            if type(binary_fp) != torch.Tensor:
                binary_fp = torch.tensor(binary_fp, dtype=torch.float)
            self.fp = torch.stack((torch.squeeze(self.fp), binary_fp))
            self.fp = self.fp.permute((1,2,0))
            #print(np.shape(self.fp))




    def __len__(self):
        return self.inputs.size()[0]

    def __getitem__(self, item):
        return self.inputs[item,:,self.present+self.notime], self.inputs[item,:,self.past+self.notime], self.fp[item,:,:]  
    
    def inverse_transform(self, predictions, remove_negs=True, remove_highs=False):
        if np.shape(predictions) != np.shape(self.fp_untransformed):
            print("careful, the predictions you inputted don't have the same shape as the footprints in this dataset!")
        if remove_highs==True and self.transform_output=="boxcox":
            predictions[predictions>3]=3
            predictions[predictions<0]=0
        self.predictions=predictions #.detach().numpy()
        if self.transform_output=="boxcox" and self.scale_output:
            #rescaled = (self.predictions-np.squeeze(self.test_mode["output_min"]))*(np.squeeze(self.test_mode["output_max"]) - np.squeeze(self.test_mode["output_min"]))/2
            #rescaled=np.squeeze(self.test_mode["output_min"]) + (self.predictions-np.squeeze(self.test_mode["output_min"]))*(np.squeeze(self.test_mode["output_max"]) - np.squeeze(self.test_mode["output_min"]))/2
            self.transformed_predictions=self.boxcox.inverse_transform(self.output_minmax.inverse_transform(predictions))-0.0000001
        elif self.transform_output=="boxcox" and not self.scale_output:
            self.transformed_predictions=self.boxcox.inverse_transform(self.predictions)-0.0000001
        if self.transform_output=="log":
            self.transformed_predictions=np.exp(((self.predictions)*self.fp_var+self.fp_mean))-1

        if self.transform_output=="mu-law":
            print("inverting")
            normalised = (1/self.transform_parameters["mu"])*(np.exp(predictions*(np.log(1+self.transform_parameters["mu"])))-1) 
            self.transformed_predictions = normalised*self.transform_parameters["max"]

        if remove_negs:
            self.transformed_predictions[self.transformed_predictions<0] = 0

        return self.transformed_predictions
    
    def prediction_error(self):
            try:
                self.transformed_predictions
            except AttributeError:
                print("only works currently for transformed/normalised outputs!")
            assert np.shape(self.transformed_predictions) == np.shape(self.fp_untransformed), "predictions and fp have to have the same shape to evaluate errors"
            nmaes = []
            for i in range(len(self.transformed_predictions)):
                nmaes.append(NMAE(self.transformed_predictions[i,:], self.fp_untransformed[i,:]))
            mse = mean_squared_error(self.transformed_predictions, self.fp_untransformed)

            print(f"NMAE: {np.mean(nmaes)}, MSE: {mse}")
            return np.mean(nmaes), mse

    def predict_fluxes(self, flux, units_transform = "default"):
        ## convolute predicted footprints and fluxes, returns two np arrays, one with the true flux and one with the emulated flux, of shape (n_footprints,)
        ## flux is an array, regridded and cut to the same resolution and size of the footprints
        ## units_transform can be None (use fluxes directly), "default" (performs flux*1e3 / CH4molarmass) or another function (which should return an array of the same shape as the original flux)
        try:
            self.transformed_predictions
        except AttributeError:
            print("only works currently for transformed/normalised outputs!")

        shape = int(np.sqrt(np.shape(self.transformed_predictions)[1])) 
        if units_transform != None:
            if units_transform == "default":
                molarmass = 16.0425
                flux = flux*1e3 / molarmass
            else:
                flux = units_transform(flux)
        true_concentration = np.reshape(self.fp_untransformed, (len(self.fp_untransformed), shape, shape))*flux
        self.true_flux = np.sum(true_concentration, axis = (1,2))
        pred_concentration = np.reshape(self.transformed_predictions, (len(self.transformed_predictions), shape, shape))*flux
        self.pred_flux = np.sum(pred_concentration, axis = (1,2))
        
        return self.true_flux, self.pred_flux



def NMAE(pred, target, mode="all"):
    pred = pred.detach().numpy() if torch.is_tensor(pred) else pred
    target = target.detach().numpy() if torch.is_tensor(target) else target
    if mode=="per cell":
        return np.mean(mean_absolute_error(target, pred, multioutput="raw_values")/(np.mean(target, axis=0)))
    if mode=="all":
        return np.mean(mean_absolute_error(target, pred)/(np.mean(target)))        


