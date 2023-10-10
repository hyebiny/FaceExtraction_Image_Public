#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class RegressionLoss(nn.Module):
    __name__ = 'Regression'
    def __init__(self, loss_type):
        super(RegressionLoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, logits, labels, weight=None):
        if weight is None:
            if self.loss_type == 'l1':
                return F.l1_loss(logits, labels)
            elif self.loss_type == 'l2':
                return F.mse_loss(logits, labels)
            else:
                raise NotImplementedError("Not Implemented loss type {}".format(self.loss_type))
        else:
            if self.loss_type == 'l1':
                return F.l1_loss(logits*weight, labels*weight, reduction='sum') / (torch.sum(weight) + 1e-8)
            elif self.loss_type == 'l2':
                return F.mse_loss(logits*weight, labels*weight, reduction='sum') / (torch.sum(weight) + 1e-8)
            else:
                raise NotImplementedError("Not Implemented loss type {}".format(self.loss_type))
            

class CompositionLoss(nn.Module):
    __name__ = 'Composition'
    def __init__(self, loss_type):
        super(CompositionLoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, logits, labels, skin, occ_mask, bg, occ, weight=None):
        
        # Composition & Regression
        
        # skin mask is binary mask: white=1, black=0
        tmp = torch.mul(logits, skin)    
        # occ_mask and skin_mask has no intersection
        tmp = torch.add(tmp, occ_mask)
        
        tmp_gt = torch.mul(labels, skin)
        tmp_gt = torch.add(tmp_gt, occ_mask)

        pred_comp = occ * tmp + bg * (1 - tmp)
        gt_comp = occ * tmp_gt + bg * (1 - tmp_gt)

        if weight is None:
            if self.loss_type == 'l1':
                return F.l1_loss(pred_comp, gt_comp)
            elif self.loss_type == 'l2':
                return F.mse_loss(pred_comp, gt_comp)
            else:
                raise NotImplementedError("Not Implemented loss type {}".format(self.loss_type))
        else:
            if self.loss_type == 'l1':
                return F.l1_loss(pred_comp*weight, gt_comp*weight, reduction='sum') / (torch.sum(weight) + 1e-8)
            elif self.loss_type == 'l2':
                return F.mse_loss(pred_comp*weight, gt_comp*weight, reduction='sum') / (torch.sum(weight) + 1e-8)
            else:
                raise NotImplementedError("Not Implemented loss type {}".format(self.loss_type))

class LapLoss(nn.Module):
    __name__ = 'Laplacian'
    def __init__(self, loss_type):
        super(LapLoss, self).__init__()
        '''
        Based on FBA Matting implementation:
        https://gist.github.com/MarcoForte/a07c40a2b721739bb5c5987671aa5270
        '''

        self.loss_type = loss_type
        self.regression = RegressionLoss(loss_type=loss_type)
        
        self.gauss_filter = torch.tensor([[1., 4., 6., 4., 1.],
                                        [4., 16., 24., 16., 4.],
                                        [6., 24., 36., 24., 6.],
                                        [4., 16., 24., 16., 4.],
                                        [1., 4., 6., 4., 1.]]).cuda()
        self.gauss_filter /= 256.
        self.gauss_filter = self.gauss_filter.repeat(1, 1, 1, 1)

    def conv_gauss(self, x, kernel):
        x = F.pad(x, (2,2,2,2), mode='reflect')
        x = F.conv2d(x, kernel, groups=x.shape[1])
        return x
    
    def downsample(self, x):
        return x[:, :, ::2, ::2]
    
    def upsample(self, x, kernel):
        N, C, H, W = x.shape
        cc = torch.cat([x, torch.zeros(N,C,H,W).cuda()], dim = 3)
        cc = cc.view(N, C, H*2, W)
        cc = cc.permute(0,1,3,2)
        cc = torch.cat([cc, torch.zeros(N, C, W, H*2).cuda()], dim = 3)
        cc = cc.view(N, C, W*2, H*2)
        x_up = cc.permute(0,1,3,2)
        return self.conv_gauss(x_up, kernel=4*self.gauss_filter)
    
    def lap_pyramid(self, x, kernel, max_levels=3):
        current = x
        pyr = []
        for level in range(max_levels):
            filtered = self.conv_gauss(current, kernel)
            down = self.downsample(filtered)
            up = self.upsample(down, kernel)
            diff = current - up
            pyr.append(diff)
            current = down
        return pyr
    
    def weight_pyramid(self, x, max_levels=3):
        current = x
        pyr = []
        for level in range(max_levels):
            down = self.downsample(current)
            pyr.append(current)
            current = down
        return pyr
    
    def forward(self, logit, target, weight=None):
        pyr_logit = self.lap_pyramid(x = logit, kernel = self.gauss_filter, max_levels = 5)
        pyr_target = self.lap_pyramid(x = target, kernel = self.gauss_filter, max_levels = 5)
        if weight is not None:
            pyr_weight = self.weight_pyramid(x = weight, max_levels = 5)
            return sum(self.regression(A[0], A[1], weight=A[2]) * (2**i) for i, A in enumerate(zip(pyr_logit, pyr_target, pyr_weight)))
        else:
            return sum(self.regression(A[0], A[1], weight=None) * (2**i) for i, A in enumerate(zip(pyr_logit, pyr_target)))



class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss > self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


class OhemBCELoss(nn.Module):
    def __init__(self, thresh, n_min):
        super(OhemBCELoss, self).__init__()
        self.n_min = n_min
        self.criteria = nn.BCEWithLogitsLoss(reduction='none')
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()

    def forward(self, logits, labels):
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss > self.thresh]

        else:
            loss = loss[: self.n_min]

        return torch.mean(loss)


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, ignore_lb=255, *args, **kwargs):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1. - scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.Sigmoid()

    def forward(self, pr, gt, eps=1e-7):
        pr = self.activation(pr)
        tp = torch.sum(gt * pr)
        fp = torch.sum(pr) - tp
        fn = torch.sum(gt) - tp
        score = (2 * tp + eps) / (2 * tp + fn + fp + eps)
        return 1-score


class IoU(nn.Module):
    __name__ = 'iou_score'

    def __init__(self, threshold=0.0, eps=1e-7):
        super().__init__()
        self.threshold = threshold
        self.eps = 1e-7

    def iou(self, pr, gt):
        pr = (pr > self.threshold).type(pr.dtype)
        intersection = torch.sum(gt * pr) + self.eps
        union = torch.sum(gt) + torch.sum(pr) - intersection + self.eps
        return intersection / union

    def forward(self, pr, gt):
        return self.iou(pr, gt)

class Precision(nn.Module):
    __name__ = 'Precision_score'

    def __init__(self, threshold=0.0, eps=1e-7):
        super().__init__()
        self.threshold = threshold
        self.eps = 1e-7

    def iou(self, pr, gt):
        pr = (pr > self.threshold).type(pr.dtype)
        intersection = torch.sum(gt * pr) + self.eps
        union = torch.sum(pr)+ self.eps
        return intersection / union

    def forward(self, pr, gt):
        return self.iou(pr, gt)

if __name__ == '__main__':
    torch.manual_seed(15)
    criteria1 = OhemCELoss(thresh=0.7, n_min=16 * 20 * 20 // 16).cuda()
    criteria2 = OhemCELoss(thresh=0.7, n_min=16 * 20 * 20 // 16).cuda()
    net1 = nn.Sequential(
        nn.Conv2d(3, 19, kernel_size=3, stride=2, padding=1),
    )
    net1.cuda()
    net1.train()
    net2 = nn.Sequential(
        nn.Conv2d(3, 19, kernel_size=3, stride=2, padding=1),
    )
    net2.cuda()
    net2.train()

    with torch.no_grad():
        inten = torch.randn(16, 3, 20, 20).cuda()
        lbs = torch.randint(0, 19, [16, 20, 20]).cuda()
        lbs[1, :, :] = 255

    logits1 = net1(inten)
    logits1 = F.interpolate(logits1, inten.size()[2:], mode='bilinear')
    logits2 = net2(inten)
    logits2 = F.interpolate(logits2, inten.size()[2:], mode='bilinear')

    loss1 = criteria1(logits1, lbs)
    loss2 = criteria2(logits2, lbs)
    loss = loss1 + loss2
    print(loss.detach().cpu())
    loss.backward()
