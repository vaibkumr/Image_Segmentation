import torch
import torch.nn as nn
import numpy as np

def dice(img1, img2):
    """
    Not differentiable so can't be used as a loss but only as a metric.
    (This is Kaggle LB metric for satellite cloud in best of my knowledge)
    """
    img1 = np.asarray(img1).astype(np.bool)
    img2 = np.asarray(img2).astype(np.bool)

    intersection = np.logical_and(img1, img2)
    if img1.sum() + img2.sum() == 0:
        # Kaggle LB is not using a smooth dice.
        return 1
    else:
        return 2. * intersection.sum() / (img1.sum() + img2.sum())

def dice_loss(pr, gt, eps=1e-9, threshold=None, activation='sigmoid'):
    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = torch.nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = torch.nn.Softmax2d()
    else:
        raise NotImplementedError(
            "Activation implemented for sigmoid and softmax2d"
        )

    pr = activation_fn(pr)

    if threshold is not None:
        pr = (pr > threshold).float()


    intersection = torch.sum(gt * pr)
    sum1 = torch.sum(pr)
    sum2 = torch.sum(gt)

    score = (2.0 * intersection + eps) / (sum1 + sum2 + eps)
    return score


def dice_kaggle(pr, gt, eps=1e-9, threshold=None, activation='sigmoid'):
    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = torch.nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = torch.nn.Softmax2d()
    else:
        raise NotImplementedError(
            "Activation implemented for sigmoid and softmax2d"
        )

    pr = activation_fn(pr)

    if threshold is not None:
        pr = (pr > threshold).float()

    intersection = torch.sum(gt*pr, axis=[0, 2, 3])
    sum1 = torch.sum(pr, axis=[0, 2, 3])
    sum2 = torch.sum(gt, axis=[0, 2, 3])
    score = (2.0 * intersection + eps) / (sum1 + sum2 + eps)
    return torch.mean(score)

class DiceLoss(nn.Module):
    __name__ = 'dice_loss'

    def __init__(self, eps=1e-9, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1-dice_kaggle(y_pr, y_gt, eps=self.eps,
                    threshold=None, activation=self.activation)


class BCEDiceLoss(DiceLoss):
    __name__ = 'bce_dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__(eps, activation)
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, y_pr, y_gt):
        dice = super().forward(y_pr, y_gt)
        bce = self.bce(y_pr, y_gt)
        return dice + bce
