import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# 支持多分类和二分类
class FocalLoss(nn.Module):
    def __init__(self, num_class, alpha=None, gamma=0,
                size_average=True):
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average

        if self.alpha is None:
            self.alpha = torch.ones(1, self.num_class)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        else:
            raise TypeError('Not support alpha type')

    def forward(self, logit, target):
        batch_size = logit.shape[0]
        logit = logit.view(batch_size, -1)
        target = target.view(batch_size, -1)
        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        pos_pt = logit
        neg_pt = 1 - logit
        gamma = self.gamma
        pos_loss = torch.pow((1 - pos_pt), gamma) * torch.log(pos_pt) * target
        neg_loss = torch.pow((1 - neg_pt), gamma) * torch.log(neg_pt) * (1 - target)
        loss = -1 * (pos_loss + neg_loss)

        if self.size_average:
            loss = loss.sum() / batch_size
        else:
            loss = loss.sum()
        return loss