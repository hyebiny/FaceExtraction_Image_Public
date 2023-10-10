# from https://gaussian37.github.io/dl-pytorch-lr_scheduler/


import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0 # the size of the first period
        self.T_mult = T_mult # the size of the next period = multiplied T_mult, only integer, increasing period
        self.base_eta_max = eta_max # minimum of the learning rate
        self.eta_max = eta_max # maximum of the learning rate
        self.T_up = T_up # epochs of the warm up 
        self.T_i = T_0 
        self.gamma = gamma # multiply to eta_max
        self.cycle = 0 
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr



if __name__ == '__main__':

    # test the scheduler

    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt

    s_period = 10000
    c_period = 1
    gamma = 0.75
    G_lr = 5e-4
    warmup_step = 1000

    
    model = nn.Linear(2, 1)
    
    G_optimizer = torch.optim.AdamW(model.parameters(), lr=0)
    # G_scheduler = CosineAnnealingWarmUpRestarts(G_optimizer, T_0=s_period, T_mult=c_period, eta_max = G_lr, T_up = warmup_step, gamma = gamma)
    G_scheduler = CosineAnnealingWarmUpRestarts(G_optimizer, T_0=20, T_mult=2, eta_max=0.001, T_up=2, gamma=0.5)


    lrs  = []
    for  i  in  range(1, 60):
        G_scheduler.step()
        lrs.append(G_optimizer.param_groups[0]["lr"])  

    plt.plot(lrs)
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.title("CosineAnnealingWarmRestarts LR Scheduler")
    plt.savefig('./lr_scheduler.png')