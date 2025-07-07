"""
author: Elena Fillola @elenafillo
"""


import torch
import torch.nn as nn
import numpy as np
#import plenoptic as po

"""
class NLPD_loss(nn.Module):
    ## add here "
    assymetric MSE - penalises underprediction by a factor of alpha before taking square
    ## add here "
    def __init__(self, train_dataset):
        self.min_train = train_dataset.fp.min()
        self.max_train = train_dataset.fp.max()

        self.size = train_dataset.size
        super().__init__()
        

    def forward(self, output, target):
        # scale
        output = (output-self.min_train)/(self.max_train - self.min_train)
        target = (target-self.min_train)/(self.max_train - self.min_train)
        # reshape
        #print(output.max(), output.min(), target.max(), target.min())
        output = torch.reshape(output.squeeze(), ((output.size()[0], self.size, self.size)))
        target = torch.reshape(target.squeeze(), ((target.size()[0], self.size, self.size)))
        #print(output.size(), target.size())
        nlpd = po.metric.nlpd(target[:,None,:,:], output[:,None,:,:]).mean()
        
        return nlpd
"""

def multiply_by(d, w=1000):
    #print("multiplying", w)
    return w*d

def multiply_by_add(d, a=1, w=1000):
    #print("multiplying", w)
    return w*d + a


class MSE_weighted_by_truth(nn.Module):
    def __init__(self, footprint_transform=None, **kwargs):
        super().__init__()
        self.footprint_transform = footprint_transform
        self.transform_kwargs = kwargs
        self.criterion = nn.MSELoss(reduction="none")

    def forward(self, output, target, true_footprint):
        loss = self.criterion(output, target)
        if self.footprint_transform is not None:
            true_footprint = self.footprint_transform(true_footprint, **self.transform_kwargs)
        loss = loss*true_footprint
        return loss.mean()
    
class MSE_weighted_by_truth_nans(nn.Module):
    def __init__(self, footprint_transform=None, **kwargs):
        super().__init__()
        self.footprint_transform = footprint_transform
        self.transform_kwargs = kwargs
        self.criterion = nn.MSELoss(reduction="none")

    def forward(self, output, target, true_footprint):
        nanmask = ~true_footprint.isnan()
        loss = self.criterion(output.nan_to_num(), target.nan_to_num())
        if self.footprint_transform is not None:
            true_footprint = self.footprint_transform(true_footprint.nan_to_num(), **self.transform_kwargs)
        loss = loss*true_footprint*nanmask
        return loss.mean()
    
class MSE_weighted_by_truth_nanmask(nn.Module):
    def __init__(self, footprint_transform=None, **kwargs):
        super().__init__()
        self.footprint_transform = footprint_transform
        self.transform_kwargs = kwargs
        self.criterion = nn.MSELoss(reduction="none")

    def forward(self, output, target, true_footprint, nanmask):
        loss = self.criterion(output, target)
        if self.footprint_transform is not None:
            true_footprint = self.footprint_transform(true_footprint, **self.transform_kwargs)
        loss = loss*true_footprint*nanmask
        return loss.mean()

class MSE_nanmask(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()    
        self.criterion = nn.MSELoss(reduction="none")

    def forward(self, output, target, nanmask):
        loss = self.criterion(output, target)
        loss = loss*nanmask
        return loss.mean()
        
class MSE_nans(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()    
        self.criterion = nn.MSELoss(reduction="none")

    def forward(self, output, target):
        nanmask = ~target.isnan()
        loss = self.criterion(output.nan_to_num(), target.nan_to_num())
        loss = loss*nanmask
        return loss.mean()


class MSE_weighted_by_truth_assymetric(nn.Module):
    def __init__(self, footprint_transform=None, alpha=4, **kwargs):
        super().__init__()
        self.footprint_transform = footprint_transform
        self.transform_kwargs = kwargs
        self.alpha = alpha
        self.criterion = nn.MSELoss(reduction="none")

    def forward(self, output, target, true_footprint):
        loss = self.criterion(output, target)
        positive_mask = (target-output)>0
        
        if self.footprint_transform is not None:
            true_footprint = self.footprint_transform(true_footprint, **self.transform_kwargs)
            
        loss = loss*true_footprint
        loss[positive_mask] = self.alpha*loss[positive_mask]

        return loss.mean()
    
class MSE_assymetric_weighted_by_truth(nn.Module):
    def __init__(self, footprint_transform=None, alpha=4, **kwargs):
        super().__init__()
        self.footprint_transform = footprint_transform
        self.transform_kwargs = kwargs
        self.alpha = alpha
        #self.criterion = nn.MSELoss(reduction="none")

    def forward(self, output, target, true_footprint):
        loss = target-output
        loss[loss>0] = self.alpha*loss[loss>0]
        loss = torch.mean(loss**2)
        
        if self.footprint_transform is not None:
            true_footprint = self.footprint_transform(true_footprint, **self.transform_kwargs)
        loss = loss*true_footprint
        return loss.mean()
    

    



class MSE_weighted(nn.Module):
    """

    """
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        criterion = nn.MSELoss(reduction="none")
        loss = torch.mean(criterion(output, target), dim=1)
        return torch.mean(torch.mul(torch.squeeze(loss),0.1*fp_sum))
    
class MSE_assymetric(nn.Module):
    """
    assymetric MSE - penalises underprediction by a factor of alpha before taking square
    """
    def __init__(self, alpha=4):
        self.alpha=alpha
        super().__init__()
        

    def forward(self, output, target):
        loss=target-output
        loss[loss>0] = self.alpha*loss[loss>0]
        loss = torch.mean(loss**2)
        return loss

def accuracy_positives(preds, fps, threshold=0, return_mean=True):
    # calculates metric intersection over union (IoU) for a footprint, binarised by threshold 
    # if using logv3 , use threshold around -1 or -2
    if type(fps) == torch.Tensor:
        accuracy = torch.sum((fps>threshold)==(preds>threshold))/torch.prod(torch.tensor(fps.size()))
        return torch.mean(accuracy)
    else:
        if len(np.shape(fps))==3:
            fps = np.reshape(np.copy(fps), (len(fps), np.shape(fps)[1]*np.shape(fps)[1]))
            preds = np.reshape(np.copy(preds), (len(preds), np.shape(preds)[1]*np.shape(preds)[1]))

        accuracy = np.sum(np.bitwise_and((fps>threshold),(preds>threshold)), axis=(-1))/((np.shape(preds)[1]))

        if return_mean:
            return np.mean(accuracy)
        if not return_mean:
            return accuracy 


def accuracy(preds, fps, threshold=0, return_mean=True):
    # calculates metric intersection over union (IoU) for a footprint, binarised by threshold 
    # if using logv3 , use threshold around -1 or -2
    if type(fps) == torch.Tensor:
        accuracy = torch.sum((fps>threshold)==(preds>threshold))/torch.prod(torch.tensor(fps.size()))
        return torch.mean(accuracy)
    else:
        if len(np.shape(fps))==3:
            fps = np.reshape(np.copy(fps), (len(fps), np.shape(fps)[1]*np.shape(fps)[1]))
            preds = np.reshape(np.copy(preds), (len(preds), np.shape(preds)[1]*np.shape(preds)[1]))

        accuracy = np.sum((fps>threshold)==(preds>threshold), axis=(-1))/((np.shape(preds)[1]))

        if return_mean:
            return np.mean(accuracy)
        if not return_mean:
            return accuracy

def dice_similarity(fps, preds, threshold=0):
    # calculates metric Dice Similarity for a binary footprint 

    if len(np.shape(fps))==3:
        fps = np.reshape(np.copy(fps), (len(fps), np.shape(fps)[1]*np.shape(fps)[1]))
        preds = np.reshape(np.copy(preds), (len(preds), np.shape(preds)[1]*np.shape(preds)[1]))

    fps_bin = np.copy(fps)>threshold
    preds_bin = np.copy(preds)>threshold


    TP = np.sum(np.logical_and(fps_bin==1, preds_bin==1, where=1), axis=(-1))
    FP = np.sum(np.logical_and(fps_bin==0, preds_bin==1, where=1), axis=(-1))
    FN = np.sum(np.logical_or(fps_bin==1, preds_bin==0, where=1), axis=(-1))
    dice = 2*TP/(2*TP + FP + FN)
    return np.mean(dice)


def intersection_over_union(preds, fps, threshold=0, return_mean=True):
    # calculates metric intersection over union (IoU) for a binary footprint 
    if type(fps) == torch.Tensor:
        intersection = torch.sum(torch.logical_and((fps>threshold)==True, (preds>threshold)==True), dim=tuple(range(1, fps.dim())))
        union = torch.sum(torch.logical_or((fps>threshold)==True, (preds>threshold)==True), dim=tuple(range(1, fps.dim())))
        IoU = torch.divide(intersection,union)
        if return_mean:
            return torch.mean(IoU) 
        else:
            return IoU
    
    else:  
        if len(np.shape(fps))==3:
            fps = np.reshape(np.copy(fps), (len(fps), np.shape(fps)[1]*np.shape(fps)[1]))
            preds = np.reshape(np.copy(preds), (len(preds), np.shape(preds)[1]*np.shape(preds)[1]))
            
        fps_bin = np.copy(fps)>threshold
        preds_bin = np.copy(preds)>threshold
        intersection = np.sum(np.logical_and(fps_bin==1, preds_bin==1, where=1), axis=(-1))
        #print(np.shape(fps))
        #print(intersection)
        union = np.sum(np.logical_or(fps_bin==1, preds_bin==1, where=1), axis=(-1))
        IoU = intersection/union
        if return_mean:
            return np.mean(IoU)
        else:
            return IoU

class CombinedLossNLPD(nn.Module):
    """
    combines assymetric MSE loss with other functions, according to passed weights
    accuracy is calculated as the percentage of correctly predicted pixels (ie true positives and true negatives) when "binarising" the footprint with threshold accuracy_threshold
    """    
    def __init__(self, train_dataset, alpha=4, acc_weighting=1, NLPD_weighting=1, accuracy_threshold=0, metric=intersection_over_union):
        super().__init__()  
        self.alpha=alpha
        self.acc_weighting=acc_weighting
        self.accuracy_threshold=accuracy_threshold
        self.metric = metric
        self.NLPD_weighting = NLPD_weighting
        self.NLPD = NLPD_loss(train_dataset)
              

    def forward(self, output, target):
        loss=target-output
        loss[loss>0] = self.alpha*loss[loss>0]
        loss = torch.mean(loss**2)
        acc = 1-self.metric(output, target, threshold=self.accuracy_threshold)
        nlpd = self.NLPD(output, target)
        return loss + self.acc_weighting*acc + self.NLPD_weighting*nlpd



class CombinedLoss(nn.Module):
    """
    combines assymetric MSE loss (see above) with accuracy, weighted according to acc_weighting factor
    accuracy is calculated as the percentage of correctly predicted pixels (ie true positives and true negatives) when "binarising" the footprint with threshold accuracy_threshold
    """    
    def __init__(self, alpha=4, acc_weighting=1, accuracy_threshold=0, metric=accuracy):
        self.alpha=alpha
        self.acc_weighting=acc_weighting
        self.accuracy_threshold=accuracy_threshold
        self.metric = metric
        super().__init__()        

    def forward(self, output, target):
        loss=target-output
        loss[loss>0] = self.alpha*loss[loss>0]
        loss = torch.mean(loss**2)
        acc = 1-self.metric(output, target, threshold=self.accuracy_threshold)
        return loss + self.acc_weighting*acc
    
      
class MSEassymetricSummed(nn.Module):
    """
    assymetric MSE - penalises underprediction by a factor of alpha before taking square
    """
    def __init__(self, alpha=4, sum_weighting=1):
        self.alpha=alpha
        self.sum_weighting = sum_weighting
        super().__init__()
        

    def forward(self, output, target):
        loss=target-output
        loss[loss>0] = self.alpha*loss[loss>0]
        loss = torch.mean(loss**2)
        
        summed = torch.mean(torch.nn.functional.mse_loss(torch.sum(target, dim=-1), torch.sum(output, dim=-1)))

        loss = loss + self.sum_weighting*summed
        return loss
    

class MSE_ACC_SUM(nn.Module):
    """
    assymetric MSE + summed footprint penalisation + accuracy = loss. penalises underprediction by a factor of alpha before taking square
    """
    def __init__(self, alpha=4, sum_weighting=1, acc_weighting=1, accuracy_threshold=0, metric=accuracy):
        self.alpha=alpha
        self.sum_weighting = sum_weighting
        self.acc_weighting=acc_weighting
        self.accuracy_threshold=accuracy_threshold
        self.metric = metric

        super().__init__()
        
    def forward(self, output, target):
        loss=target-output
        loss[loss>0] = self.alpha*loss[loss>0]
        loss = torch.mean(loss**2)
        
        summed = torch.nn.functional.mse_loss(torch.sum(target, dim=-1), torch.sum(output, dim=-1))

        acc = 1-self.metric(output, target, threshold=self.accuracy_threshold)

        loss = loss + self.sum_weighting*summed + self.acc_weighting*acc
        return loss
    


