import numpy as np
from scipy import spatial
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
def dice_coeff(im1, im2, empty_score=1.0):
    """Calculates the dice coefficient for the images"""

    im1 = np.asarray(im1).astype(np.bool_)
    im2 = np.asarray(im2).astype(np.bool_)

    if im1.shape != im2.shape:
        print('im1',im1.shape)
        print('im2',im2.shape)
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im1 = im1 > 0.5
    im2 = im2 > 0.5

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    #print(im_sum)

    return 2. * intersection.sum() / im_sum

def dice_coef(output, target):
    smooth = 1e-5

    # we need to use sigmoid because the output of Unet is logit.
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()
    
    dice = (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)
    return dice

def dice_coef_scaler(output, target):
    smooth = 1e-5
    # we need to use sigmoid because the output of Unet is logit.
    with autocast():
        output = torch.sigmoid(output).view(-1).data.cpu().numpy()
        target = target.view(-1).data.cpu().numpy()
        intersection = (output * target).sum()

        dice = (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)
    return dice

def numeric_score(prediction, groundtruth):
    """Computes scores:
    FP = False Positives
    FN = False Negatives
    TP = True Positives
    TN = True Negatives
    return: FP, FN, TP, TN"""

    FP = np.float(np.sum((prediction == 1) & (groundtruth == 0)))
    FN = np.float(np.sum((prediction == 0) & (groundtruth == 1)))
    TP = np.float(np.sum((prediction == 1) & (groundtruth == 1)))
    TN = np.float(np.sum((prediction == 0) & (groundtruth == 0)))

    return FP, FN, TP, TN


def accuracy_score(prediction, groundtruth):
    """Getting the accuracy of the model"""

    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    N = FP + FN + TP + TN
    accuracy = np.divide(TP + TN, N)
    return accuracy * 100.0

def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)