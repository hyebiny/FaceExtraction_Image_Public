import torch
from tqdm import tqdm
from meter import AverageValueMeter
from Dataset.utils import tensor2img
import sys
import numpy as np
import pathlib
import os
import cv2
from utils import CONFIG

from torchvision.utils import make_grid
import torch.nn.functional as F
import torch.nn as nn
import math
import scipy
import random


class Epoch:
    def __init__(self, model, loss, metrics, stage_name, device='cpu', sv_pth=None, show_step=10000, verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device
        self.it = 0
        self.show_step = show_step
        if not sv_pth:
            pth = pathlib.Path(__file__).parent.absolute()
            sv_folder = 'res'
            sv_root = os.path.join(pth, sv_folder)
            self.sv_root = sv_root
            if not os.path.exists(sv_root):
                os.mkdir(sv_root)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader, epoch):
        self.on_epoch_start()
        logs = {}
        loss_meter = {loss.__name__: AverageValueMeter() for loss in self.loss} # AverageValueMeter()
        metric_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}
        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not self.verbose) as iterator:
            for sample in iterator:
                self.it += 1
                loss, y_pred, loss_logs = self.batch_update(sample, epoch)

                # loss_value = loss.item()
                # loss_meter.add(loss_value)
                # loss_logs = {self.loss.__name__: loss_meter.mean }
                # logs.update(loss_logs)

                image, mask = sample['image'], sample['mask']
                image, mask = image.to(self.device), mask.to(self.device)

                # loss_logs = {}
                # sum_loss = 0
                # for loss_fn in self.loss:
                #     if 'Composition' == loss_fn.__name__:
                #         if self.stage_name == 'valid':
                #             loss_value = 0
                #         else:
                #             skin, occ_mask = sample['skin'].to(self.device), sample['occ_mask'].to(self.device)
                #             bg, occ = sample['bg'].to(self.device), sample['occ'].to(self.device)
                #             loss_value = loss_fn(y_pred, mask, skin, occ_mask, bg, occ).item()
                #     else:   
                #         loss_value = loss_fn(y_pred, mask).item()
                #     loss_meter[loss_fn.__name__].add(loss_value)
                #     loss_logs.update({loss_fn.__name__: loss_meter[loss_fn.__name__].mean})
                #     sum_loss += loss_meter[loss_fn.__name__].mean
                # loss_logs.update({'total':sum_loss})
                logs.update(loss_logs)

                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, mask).item()
                    metric_meters[metric_fn.__name__].add(metric_value)

                metric_logs = {k: v.mean for k, v in metric_meters.items()}
                logs.update(metric_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

                if self.it % self.show_step == 0:
                    with torch.no_grad():

                        '''pred_mask = (y_pred > 0).type(torch.float32)
                        face_pred = tensor2img(x * pred_mask)
                        face_gt = tensor2img(x * y)
                        img = tensor2img(x)
                        show = np.concatenate((img, face_gt, face_pred), axis=0)
                        sv_name = f'{self.stage_name}_it_{self.it}.png'
                        sv_pth = os.path.join(self.sv_root, sv_name)
                        cv2.imwrite(sv_pth, show[:,:,::-1])'''

                        img = tensor2img(image)
                        # show = np.concatenate((img, face_gt, face_pred), axis=0)

                        gt = make_grid(mask, nrow=image.shape[0], padding=0)
                        gt = gt.detach().permute(1, 2, 0).cpu().numpy() * 255
                        pred = make_grid(y_pred, nrow=image.shape[0], padding=0)
                        pred = pred.detach().permute(1, 2, 0).cpu().numpy() * 255

                        show = np.concatenate((img, gt, pred), axis=0)
                        sv_name = f'{self.stage_name}_it_{self.it}.png'
                        sv_pth = os.path.join(self.sv_root, sv_name)
                        cv2.imwrite(sv_pth, show[:,:,::-1])
                
                # torch.cuda.empty_cache()

        return logs


class GaussianBlurLayer(nn.Module):
    """ Add Gaussian Blur to a 4D tensors
    This layer takes a 4D tensor of {N, C, H, W} as input.
    The Gaussian blur will be performed in given channel number (C) splitly.
    """

    def __init__(self, channels, kernel_size):
        """ 
        Arguments:
            channels (int): Channel for input tensor
            kernel_size (int): Size of the kernel used in blurring
        """

        super(GaussianBlurLayer, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        assert self.kernel_size % 2 != 0

        self.op = nn.Sequential(
            nn.ReflectionPad2d(math.floor(self.kernel_size / 2)), 
            nn.Conv2d(channels, channels, self.kernel_size, 
                      stride=1, padding=0, bias=None, groups=channels)
        )

        self._init_kernel()

    def forward(self, x):
        """
        Arguments:
            x (torch.Tensor): input 4D tensor
        Returns:
            torch.Tensor: Blurred version of the input 
        """

        if not len(list(x.shape)) == 4:
            print('\'GaussianBlurLayer\' requires a 4D tensor as input\n')
            exit()
        elif not x.shape[1] == self.channels:
            print('In \'GaussianBlurLayer\', the required channel ({0}) is'
                  'not the same as input ({1})\n'.format(self.channels, x.shape[1]))
            exit()
            
        return self.op(x)
    
    def _init_kernel(self):
        sigma = 0.3 * ((self.kernel_size - 1) * 0.5 - 1) + 0.8

        n = np.zeros((self.kernel_size, self.kernel_size))
        i = math.floor(self.kernel_size / 2)
        n[i, i] = 1
        kernel = scipy.ndimage.gaussian_filter(n, sigma)

        for name, param in self.named_parameters():
            param.data.copy_(torch.from_numpy(kernel))



class TrainEpoch(Epoch):
    def __init__(self, model, loss, metrics, optimizer, scheduler, device='cpu', verbose=True, sv_pth=None):
        super().__init__(model=model,
                         loss=loss,
                         metrics=metrics,
                         stage_name='train',
                         device=device,
                         verbose=verbose,
                         sv_pth=sv_pth)

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.sv_root = sv_pth
        # for mgmatting
        self.Kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1,30)]

        if CONFIG.model.model == 'modnet':
            self.blurer = GaussianBlurLayer(1, 3).cuda()

    def on_epoch_start(self):
        self.model.train()

    def get_unknown_tensor(self, trimap):
        """
        get 1-channel unknown area tensor from the 3-channel/1-channel trimap tensor
        """
        # if CONFIG.model.trimap_channel == 3:
            # weight = trimap[:, 1:2, :, :].float()
        weight = trimap.eq(1).float()
        return weight

    def get_unknown_tensor_from_pred(self, pred, rand_width=30, train_mode=True):
        ### pred: N, 1 ,H, W 
        N, C, H, W = pred.shape

        pred = pred.data.cpu().numpy()
        uncertain_area = np.ones_like(pred, dtype=np.uint8)
        uncertain_area[pred<1.0/255.0] = 0
        uncertain_area[pred>1-1.0/255.0] = 0

        for n in range(N):
            uncertain_area_ = uncertain_area[n,0,:,:] # H, W
            if train_mode:
                width = np.random.randint(1, rand_width)
            else:
                width = rand_width // 2
            uncertain_area_ = cv2.dilate(uncertain_area_, self.Kernels[width])
            uncertain_area[n,0,:,:] = uncertain_area_

        weight = np.zeros_like(uncertain_area)
        weight[uncertain_area == 1] = 1
        weight = torch.from_numpy(weight).cuda()

        return weight

        
    def batch_update(self, sample, epoch):

        logs = {}
        image, mask = sample['image'], sample['mask']
        image, mask = image.to(self.device), mask.to(self.device)

        self.optimizer.zero_grad()

        prediction = self.model(image)

        loss = 0

        if CONFIG.model.model == 'mgmatting':
            # prediction = prediction['alpha_os8']
            alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = prediction['alpha_os1'], prediction['alpha_os4'], prediction['alpha_os8']

            weight_os8 = self.get_unknown_tensor(trimap)
            weight_os8[...] = 1

            flag = False
            if epoch < 10: # set the warm-up epoch to 10
                flag = True
                weight_os4 = self.get_unknown_tensor(trimap)
                weight_os1 = self.get_unknown_tensor(trimap)
            elif epoch < 10 * 3:
                if random.randint(0,1) == 0:
                    flag = True
                    weight_os4 = self.get_unknown_tensor(trimap)
                    weight_os1 = self.get_unknown_tensor(trimap)
                else:
                    weight_os4 = self.get_unknown_tensor_from_pred(alpha_pred_os8, rand_width=CONFIG.model.self_refine_width1, train_mode=True)
                    alpha_pred_os4[weight_os4==0] = alpha_pred_os8[weight_os4==0]
                    weight_os1 = self.get_unknown_tensor_from_pred(alpha_pred_os4, rand_width=CONFIG.model.self_refine_width2, train_mode=True)
                    alpha_pred_os1[weight_os1==0] = alpha_pred_os4[weight_os1==0]
            else:
                weight_os4 = self.get_unknown_tensor_from_pred(alpha_pred_os8, rand_width=CONFIG.model.self_refine_width1, train_mode=True)
                alpha_pred_os4[weight_os4==0] = alpha_pred_os8[weight_os4==0]
                weight_os1 = self.get_unknown_tensor_from_pred(alpha_pred_os4, rand_width=CONFIG.model.self_refine_width2, train_mode=True)
                alpha_pred_os1[weight_os1==0] = alpha_pred_os4[weight_os1==0]


        if CONFIG.model.model == 'modnet':
            trimap = sample['trimap']
            pred_semantic, pred_detail, prediction = prediction
            
            # get boundary from trimap
            boundaries = (trimap == 0) + (trimap == 2) # makes unknown part to True
            semantic_scale = 10.0
            detail_scale = 10.0

            # calculate the semantic loss
            gt_semantic = F.interpolate(mask, scale_factor=1/16, mode='bilinear')
            gt_semantic = self.blurer(gt_semantic)
            semantic_loss = self.loss[0](pred_semantic, gt_semantic) # regression loss
            semantic_loss = semantic_scale * semantic_loss

            # calculate the detail loss
            pred_boundary_detail = torch.where(boundaries, trimap, pred_detail) # apply trimap to prediction
            gt_detail = torch.where(boundaries, trimap, mask)
            detail_loss = self.loss[0](pred_boundary_detail, gt_detail) # regression loss
            detail_loss = detail_scale * detail_loss

            loss += semantic_loss
            loss += detail_loss
            logs.update({'semantic':semantic_loss})
            logs.update({'detail':detail_loss})


        for loss_function in self.loss:

            if CONFIG.model.model == 'mgmatting':
                if 'Composition' == loss_function.__name__:
                    skin, occ_mask = sample['skin'].to(self.device), sample['occ_mask'].to(self.device)
                    bg, occ = sample['bg'].to(self.device), sample['occ'].to(self.device)
                    tmp = (loss_function(alpha_pred_os1, mask, skin, occ_mask, bg, occ, weight=weight_os1) * 2 + \
                            loss_function(alpha_pred_os4, mask, skin, occ_mask, bg, occ, weight=weight_os4) * 1 + \
                            loss_function(alpha_pred_os8, mask, skin, occ_mask, bg, occ, weight=weight_os8) * 1)/5
                else:   
                    tmp = (loss_function(prediction, mask, weight=weight_os1) * 2 + \
                            loss_function(prediction, mask, weight=weight_os4) * 1 + \
                            loss_function(prediction, mask, weight=weight_os8) * 1)/5

            else:
                if 'Composition' == loss_function.__name__:
                    skin, occ_mask = sample['skin'].to(self.device), sample['occ_mask'].to(self.device)
                    bg, occ = sample['bg'].to(self.device), sample['occ'].to(self.device)
                    tmp = loss_function(prediction, mask, skin, occ_mask, bg, occ)
                else:   
                    tmp = loss_function(prediction, mask)
                
            loss += tmp
            logs.update({loss_function.__name__:tmp})

        logs.update({'total': loss})


        loss.backward()
        self.optimizer.step()
        return loss, prediction, logs


class ValidEpoch(Epoch):
    def __init__(self, model, loss, metrics, device='cpu', verbose=True, sv_pth=None):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='valid',
            device=device,
            verbose=verbose,
            show_step=200,
            sv_pth=None
        )
        self.sv_root = sv_pth

    def on_epoch_start(self):
        self.model.eval()

    # def batch_update(self, x, y):
    def batch_update(self, sample, epoch):

        logs = {}
        x, y = sample['image'], sample['mask']
        x, y = x.to(self.device), y.to(self.device)

        with torch.no_grad():
            prediction = self.model(x)
            # loss = self.loss(prediction, y)
            loss = 0
            if CONFIG.model.model == 'mgmatting':
                prediction = prediction['alpha_os8']

            if CONFIG.model.model == 'modnet':
                _, _, prediction = prediction

            for loss_function in self.loss:
                if 'Composition' == loss_function.__name__:
                    pass
                else:
                    loss += loss_function(prediction, y)

        # pass the logs for this, because I only want to get the metric loss. 

        return loss, prediction, logs


if __name__ == '__main__':
    epoch = Epoch(None, None, None, None)
