from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import sklearn.preprocessing as preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler




class FootprintsDatasetV2(Dataset):
    """
    Creates dataset to pass to the model, applying any transformations specified. 
    Parameters:
    - inputs: array of shape (samples, sizexsize, features) with extracted inputs
    - fp: array of shape (samples, sizexsize) with footprint
    - input_names: dictionary with variable names
    - input_transforms: list containing all transforms to apply to the inputs. Current valid options are:
        - clever_transform (standardise each variable separately across all levels and times)
        - clever_transform_2(standardise each variable and level separately across all times)
    - output_transforms: list containing all transforms to apply to the outputs, in order. Current valid options are:
        - boxcox (apply pixel-wise boxcox)
        - boxcox_all (apply boxcox to all the data)
        - mu-law (apply mu-law enconding algorithm)
        - logv3 (take log10 and shift so mean of non-zero elements is zero)
    
    - test_mode: dict. If training, leave empty. If testing (ie applying existing parameters and/or already fitted models), pass a dictionary or the transform_parameters of another dataset

    Example:
    training_ds = FootprintsDatasetV2(inputs, data.fp_data, input_transforms=["clever_transform"],output_transforms=["boxcox_all"], input_names=names)
    testing_ds = FootprintsDatasetV2(test_inputs, test_data.fp_data, input_transforms=["clever_transform"],output_transforms=["boxcox_all"], input_names=names, test_mode=training_ds.transform_parameters))

    Functions:
    - inverse_transform(predictions): provides the footprints reconverted to the original space, inverting any previously applied transforms
    - add_prototypes(prototypes) : unfinished! Takes prototypes of same shape as footprints, applies to them the same transform applied to the fps and appends them to the inputs array.

    """
    def __init__(self, inputs, fp, input_transforms = [], output_transforms = [], transform_parameters = {}, test_mode={}, input_names=[]):
        print(transform_parameters, "!")
        self.inputs = inputs
        self.fp = fp
        self.input_transforms=input_transforms
        self.output_transforms=output_transforms
        self.input_names=input_names
        self.test_mode = test_mode
        self.transform_parameters = transform_parameters
        
        print(transform_parameters)
        print(self.transform_parameters)

        self.valid_input_transforms = {
            "clever_transform":{"params":["transformers"], "fun":_CleverTransform}, 
            "clever_transform_2":{"params":["transformers"], "fun":_CleverTransform2}, 
            "standardise":{"params":["transformers"], "fun":_Transform}, 
            "scale":{"params":["inputs_min", "inputs_max"], "fun":_Transform}}

        self.valid_output_transforms = {
            "boxcox":{"params":["transformers"], "fun":_Boxcox},
            "boxcox_all":{"params":["transformers"], "fun":_BoxcoxAll}, 
            "mu-law":{"params":["scale", "mu"], "fun":_MuLaw}, 
            "logv1":{"params":["fp_mean", "fp_var"], "fun":_Transform}, 
            "logv2":{"params":[], "fun":_Transform}, 
            "logv3":{"params":["logged_mean"], "fun":_LogV3},
            "scale":{"params":["output_minmax"], "fun":_Transform}}


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
            self.inputs = torch.tensor(self.inputs, dtype=torch.float)#, device=device)
        if type(self.fp) != torch.Tensor:
            self.fp = torch.tensor(self.fp, dtype=torch.float)


    def inverse_transform(self, predictions):
        for transform in self.output_transforms:
            self.transformed_predictions = self.output_transforms[transform].inverse_transform(predictions)

    def add_prototypes(self, prototypes):
        self.prototypes = prototypes
        for transform in self.output_transforms:
            self.prototypes = self.output_transforms[transform].transform(self.prototypes)
        raise Warning("This funcion is not yet fully implemented!")
    
        # TODO concatenate prototypes to inputs

    def __len__(self):
        return self.inputs.size()[0]

    def __getitem__(self, item):
        return self.inputs[item,:,:], self.fp[item,:,:]  
    
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

    def transform(self):
        for varname in np.unique(self.varnames):
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


        self.parent.transform_parameters["clever_transform_2"] = {}
        self.parent.transform_parameters["clever_transform_2"]["transformers"] = self.transformers      
              
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
            self.logged_mean = self.parent.test_mode["logv3"]["logged mean"]
        
        self.parent.transform_parameters["logv3"] = {}
        self.parent.transform_parameters["logv3"]["logged mean"] = self.logged_mean

    def transform(self, fp):
        logged = np.log10(self.parent.fp+0.0000001)
        fp = logged + abs(self.logged_mean) 
        return fp
    
    def inverse_transform(self, predictions):
        self.parent.predictions = predictions
        transformed_predictions = 10**(predictions - abs(self.logged_mean))-0.0000001      
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
        self.parent.transform_parameters["mu-law"]["mu"] = mu
        self.parent.transform_parameters["mu-law"]["scale"] = scale       
    
    def transform(self, fp):
        fp = np.sign(fp)*np.log(1+self.mu*(np.abs(fp/self.scale)))/(np.log(1+self.mu))
        return fp
    
    def inverse_transform(self, predictions):
        self.parent.predictions = predictions
        transformed_predictions = self.scale * np.sign(predictions) * ((1+self.mu)**(np.abs(predictions))-1)/self.mu 
        return transformed_predictions   


class FootprintsDataset(Dataset):
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

    needs updating!!
    """

    def __init__(self, inputs, fp, standardise=False, scale=False, transform_output=False, scale_output=False, test_mode={}, feature_dim=11, aux_dim=5, clever_transform=False, clever_transform_2=False, input_names=[], standardise_all=False, device="cpu", zeroing=False, binary_fp=None, transform_parameters={}):
        super().__init__()  
        self.inputs = inputs
        self.fp = fp
        self.standardise_inputs=False
        self.transform_output=False
        self.scale_output=False
        self.zeroing=zeroing

        if len(test_mode)>0:
            self.mode="test"
        else:
            self.mode="train"

    

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
                self.clever_transform_2 = True
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
                    if np.shape(fp)[1] == 200*200:
                        # hand crafted case!
                        print("doing special boxcox")
                        points = np.array([(100+x,100+y) for x in list(range(-50,50)) for y in list(range(-50,50))])
                        centre_idxs = np.ravel_multi_index([points[:,0], points[:,1]], (200,200))
                        self.fp = np.zeros_like(fp)
                        self.fp[:,centre_idxs] = test_mode["boxcox"]["centre"].transform(0.0000001+np.squeeze(fp[:,centre_idxs]))

                        rest_idxs = list(set(range(200*200)) - set(centre_idxs))
                        self.fp[:,rest_idxs] = np.reshape(test_mode["boxcox"]["outside"].transform(0.0000001+(fp[:,rest_idxs].flatten().reshape(-1, 1))), np.shape(self.fp[:,rest_idxs]))
                
                    else:
                        self.fp = test_mode["boxcox"].transform(0.0000001+np.squeeze(self.fp))[:,:,None]

                    if zeroing:
                        self.fp[self.fp<0]=0
                        print("zeroing")
                except KeyError:
                    print("no info was passed to do boxcox on the output")
                    pass

            if transform_output=="boxcox_all":
                print("here!")
                self.transform_output="boxcox_all"
                self.boxcox=test_mode["boxcox"]
                self.fp_untransformed = np.copy(fp)
                try:
                    fp = test_mode["boxcox"].transform(0.0000001+np.squeeze(self.fp.flatten()).reshape(-1, 1))[:,:,None]
                    self.fp = np.reshape(fp, np.shape(self.fp_untransformed))

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
            
            elif transform_output=="logv1":
                self.transform_output="logv1"
                self.fp_untransformed = np.copy(fp)
                logged = np.log(self.fp+1)
                self.logged=logged
                self.fp_mean, self.fp_var = test_mode["fp_mean"], test_mode["fp_var"]
                self.fp = (logged-self.fp_mean)/self.fp_var

            elif transform_output=="logv2":
                self.transform_output="logv2"
                self.fp_untransformed = np.copy(fp)
                logged = np.log10(self.fp+0.0000001)
                self.fp = logged

            elif transform_output=="logv3":
                self.transform_output="logv3"
                self.fp_untransformed = np.copy(fp)
                logged = np.log10(self.fp+0.0000001)
                self.logged_mean = test_mode["logged_mean"]
                self.fp = logged + abs(self.logged_mean)
            
                


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
                self.clever_transform_2 = True
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

            elif transform_output=="logv1":
                self.transform_output="logv1"
                self.fp_untransformed = np.copy(fp)
                logged = np.log(self.fp+1)
                self.logged=logged
                self.fp = (logged-np.mean(logged, axis=(0,1)))/np.var(logged, axis=(0,1))
                self.fp_mean = np.mean(logged, axis=(0,1))
                self.fp_var = np.var(logged, axis=(0,1))

            elif transform_output=="logv2":
                self.transform_output="logv2"
                self.fp_untransformed = np.copy(fp)
                logged = np.log10(self.fp+0.0000001)
                self.fp = logged
            elif transform_output=="logv3":
                self.transform_output="logv3"
                self.fp_untransformed = np.copy(fp)
                logged = np.log10(self.fp+0.0000001)
                self.logged_mean = np.mean(np.log10(fp.flatten()[fp.flatten()>0]))
                self.fp = logged + abs(self.logged_mean)


            elif transform_output=="boxcox":
                self.transform_output="boxcox"
                self.fp_untransformed = np.copy(fp)
                if np.shape(fp)[-1] == 200*200:
                    # hand crafted case!
                    print("doing special boxcox")
                    pt = preprocessing.PowerTransformer(method='box-cox', standardize=True)
                    self.boxcox_centre = pt
                    points = np.array([(100+x,100+y) for x in list(range(-50,50)) for y in list(range(-50,50))])
                    centre_idxs = np.ravel_multi_index([points[:,0], points[:,1]], (200,200))
                    self.fp = np.zeros_like(fp)
                    self.fp[:,centre_idxs] = self.boxcox_centre.fit_transform(0.0000001+np.squeeze(fp[:,centre_idxs]))

                    pt = preprocessing.PowerTransformer(method='box-cox', standardize=True)
                    self.boxcox_outside = pt
                    rest_idxs = list(set(range(200*200)) - set(centre_idxs))
                    self.fp[:,rest_idxs] = np.reshape(self.boxcox_outside.fit_transform(0.0000001+(fp[:,rest_idxs].flatten().reshape(-1, 1))), np.shape(self.fp[:,rest_idxs]))

                    self.boxcox = {"centre":self.boxcox_centre, "outside":self.boxcox_outside}

                else:
                    pt = preprocessing.PowerTransformer(method='box-cox', standardize=True)
                    self.boxcox = pt
                    self.fp = self.boxcox.fit_transform(0.0000001+np.squeeze(self.fp))[:,:,None]
                if zeroing:
                    self.fp[self.fp<0]=0
                    print("zeroing")

            elif transform_output=="boxcox_all":
                self.transform_output="boxcox_all"
                self.fp_untransformed = np.copy(fp)
                pt = preprocessing.PowerTransformer(method='box-cox', standardize=True)
                self.boxcox = pt
                fp = self.boxcox.fit_transform(0.0000001+np.squeeze(self.fp.flatten()).reshape(-1, 1))[:,:,None]
                self.fp = np.reshape(fp, np.shape(self.fp_untransformed))

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
        if np.shape(predictions)[0:1] != np.shape(self.fp_untransformed)[0:1]:
            print("careful, the predictions you inputted don't have the same shape as the footprints in this dataset!")
            print(np.shape(predictions), np.shape(self.fp_untransformed))
        if remove_highs==True and self.transform_output=="boxcox":
            predictions[predictions>3]=3
            predictions[predictions<0]=0

        self.predictions=predictions #.detach().numpy()

        #if type(self.predictions) == torch.Tensor:
        #    self.predictions=self.predictions.detach().numpy()

        if remove_negs_before==True and self.transform_output=="boxcox":
            predictions[predictions<0]=0

        if self.transform_output=="boxcox" and self.scale_output:
            #rescaled = (self.predictions-np.squeeze(self.test_mode["output_min"]))*(np.squeeze(self.test_mode["output_max"]) - np.squeeze(self.test_mode["output_min"]))/2
            #rescaled=np.squeeze(self.test_mode["output_min"]) + (self.predictions-np.squeeze(self.test_mode["output_min"]))*(np.squeeze(self.test_mode["output_max"]) - np.squeeze(self.test_mode["output_min"]))/2
            self.transformed_predictions=self.boxcox.inverse_transform(self.output_minmax.inverse_transform(predictions))-0.0000001
        elif self.transform_output=="boxcox" and not self.scale_output:
            if np.shape(self.fp)[1] == 200*200:
                # hand crafted case! 
                print("doing special boxcox")
                points = np.array([(100+x,100+y) for x in list(range(-50,50)) for y in list(range(-50,50))])
                centre_idxs = np.ravel_multi_index([points[:,0], points[:,1]], (200,200))
                
                if type(self.predictions) is torch.Tensor:
                    
                    self.transformed_predictions = torch.zeros(predictions.size()[0:2])
                    print(type(self.predictions[:,centre_idxs, :]))
                    self.transformed_predictions[:,centre_idxs] = torch.tensor(self.boxcox["centre"].inverse_transform(torch.squeeze(self.predictions[:,centre_idxs, :]))-0.0000001, dtype=torch.float)

                    rest_idxs = list(set(range(200*200)) - set(centre_idxs))
                    self.transformed_predictions[:,rest_idxs] = torch.tensor(np.reshape(self.boxcox["outside"].inverse_transform((torch.squeeze(self.predictions[:,rest_idxs,:]).reshape(-1, 1))), self.predictions[:,rest_idxs].size()[0:2])-0.0000001, dtype=torch.float)
                else:
                    self.transformed_predictions = np.zeros_like(self.predictions)
                    self.transformed_predictions[:,centre_idxs] = self.boxcox["centre"].inverse_transform(np.squeeze(self.predictions[:,centre_idxs]))-0.0000001

                    rest_idxs = list(set(range(200*200)) - set(centre_idxs))
                    self.transformed_predictions[:,rest_idxs] = np.reshape(self.boxcox["outside"].inverse_transform((self.predictions[:,rest_idxs].flatten().reshape(-1, 1))), np.shape(self.predictions[:,rest_idxs]))-0.0000001                
            else:       
                self.transformed_predictions=self.boxcox.inverse_transform(self.predictions)-0.0000001

        elif self.transform_output=="boxcox_all":
            print("inverse boxcox all!")
            transformed_predictions=self.boxcox.inverse_transform(self.predictions.flatten().reshape(-1, 1))-0.0000001
            self.transformed_predictions = np.reshape(transformed_predictions, np.shape(self.predictions))

        if self.transform_output=="log":
            self.transformed_predictions=np.exp(((self.predictions)*self.fp_var+self.fp_mean))-1

        if self.transform_output=="mu-law":
            print("inverting")
            normalised = (1/self.transform_parameters["mu"])*(np.exp(predictions*(np.log(1+self.transform_parameters["mu"])))-1) 
            self.transformed_predictions = normalised*self.transform_parameters["max"]

        if self.transform_output=="logv2":
            self.transformed_predictions = 10**(predictions)-0.0000001

        if self.transform_output=="logv3":
            self.transformed_predictions = 10**(predictions - abs(self.logged_mean))-0.0000001

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

            elif transform_output=="logv1":
                self.transform_output="logv1"
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
        #if np.shape(predictions) != np.shape(self.fp_untransformed):
            #print("careful, the predictions you inputted don't have the same shape as the footprints in this dataset!")
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
    
def NMAE(pred, target, mode="all"):
    pred = pred.detach().numpy() if torch.is_tensor(pred) else pred
    target = target.detach().numpy() if torch.is_tensor(target) else target
    if mode=="per cell":
        return np.mean(mean_absolute_error(target, pred, multioutput="raw_values")/(np.mean(target, axis=0)))
    if mode=="all":
        return np.mean(mean_absolute_error(target, pred)/(np.mean(target)))        


