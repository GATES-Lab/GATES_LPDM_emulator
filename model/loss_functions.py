import torch
import torch.nn as nn
import numpy as np

class MSE_weighted(nn.Module):
    """

    """
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        criterion = nn.MSELoss(reduction="none")
        loss = torch.mean(criterion(output, target), dim=1)
        fp_sum = torch.abs(torch.sum(target, dim=(1,2)))
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
    
def accuracy(preds, fps, threshold=0):
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

        return np.mean(accuracy)

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


def intersection_over_union(preds, fps, threshold=0):
    # calculates metric intersection over union (IoU) for a binary footprint 
    if len(np.shape(fps))==3:
        fps = np.reshape(np.copy(fps), (len(fps), np.shape(fps)[1]*np.shape(fps)[1]))
        preds = np.reshape(np.copy(preds), (len(preds), np.shape(preds)[1]*np.shape(preds)[1]))
        
    fps_bin = np.copy(fps)>threshold
    preds_bin = np.copy(preds)>threshold
    intersection = np.sum(np.logical_and(fps_bin==1, preds_bin==1, where=1), axis=(-1,-2))
    print(intersection)
    union = np.sum(np.logical_or(fps_bin==1, preds_bin==1, where=1), axis=(-1,-2))
    IoU = intersection/union
    return np.mean(IoU)

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
    
      
