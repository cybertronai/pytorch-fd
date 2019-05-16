"""Lamb optimizer."""

import collections
import math
import sys

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

class FDLR(optim.lr_scheduler._LRScheduler):
    """Decays the learning rate of each parameter group by gamma every
    step_size epochs. Notice that such decay can happen simultaneously with
    other changes to the learning rate from outside this scheduler. When
    last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 60
        >>> # lr = 0.0005   if 60 <= epoch < 90
        >>> # ...
        >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, epsilon=.01, gamma=0.1, last_epoch=-1, writer: SummaryWriter = None, use_mom=False, lr=0):
        self.epsilon = epsilon
        self.gamma = gamma
        self.writer = writer
        self.use_mom = use_mom
        self.o = []
        self.lr = lr
        self.factor = 1
        #        self.last_epoch = -1  # have to set here because step is called before parent sets value
        self.ol_sum = 0
        self.or_sum = 0
        self.ols = []
        self.ors = []
        self.count = 0
        self.half_running_ol = 0
        self.half_running_or = 0


        super(FDLR, self).__init__(optimizer, last_epoch)

    def step(self, epoch=None):
        super(FDLR, self).step(epoch)
        ols, ors, ratios = [], [], []
        if self.last_epoch <= 1:  # calling on init, no gradients defined yet
            return
        
        for group in self.optimizer.param_groups:
            beta1, beta2 = group.get('betas', (0,0))
            momentum = group.get('momentum', 0)
            for p in group['params']:
                if self.use_mom:
                    state = self.optimizer.state[p]
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    ol = (p.data * exp_avg).sum() / bias_correction1
                    # subtract first moment from p.data?
                    or_ = (.5 * (1+(beta1 or momentum)) * group['lr'] * (exp_avg_sq / bias_correction2)).sum()
                else:
                    grad = p.grad.data
                    #                    print(p.data)
                    #                    print(p.grad.data)
                    ol = (p.data * grad).sum()

                    #or_ = .5 * (1+(beta1 or momentum)) * group['lr'] * grad.pow(2).sum()
                    #                    or_ = .5 * (1+(beta1 or momentum)) * group['lr'] * grad.pow(2).sum()
                    or_ = 0.5 * self.lr * grad.pow(2).sum()

                ols.append(ol)
                ors.append(or_)
#                self.ratios.append(ratio)
#        print(f"ol: {ol.item():5.4f} or: {or_.item():5.4f} {grad}")
        ratio = torch.tensor(ols).sum()/torch.tensor(ors).sum()
        ol = torch.tensor(ols).sum()
        or_ = torch.tensor(ors).sum()
        
        self.ols.append(ol)
        self.ors.append(or_)
        
        self.half_running_ol = torch.tensor(self.ols[len(self.ols)//2:]).mean()
        self.half_running_or = torch.tensor(self.ors[len(self.ors)//2:]).mean()
        
        self.ol_sum = self.ol_sum+torch.tensor(ols).sum().item()
        self.or_sum = self.or_sum+torch.tensor(ors).sum().item()
        self.count += 1
        if self.writer:
            self.writer.add_scalar('fd/ol', torch.tensor(ols).sum(), self.last_epoch)
            self.writer.add_scalar('fd/or', torch.tensor(ors).sum(), self.last_epoch)
            self.writer.add_scalar('fd/ratio', ratio, self.last_epoch)
            self.writer.add_scalar('fd/ol_half', self.half_running_ol, self.last_epoch)
            self.writer.add_scalar('fd/or_half', self.half_running_or, self.last_epoch)
            self.writer.add_scalar('fd/ratio_avg', self.ol_sum/self.or_sum, self.last_epoch)
            self.writer.add_scalar('fd/ratio_half_avg', self.half_running_ol/self.half_running_or, self.last_epoch)


    def get_lr(self):
        if self.last_epoch <= 1:
            return self.base_lrs

        return [group['lr'] * self.factor
                for group in self.optimizer.param_groups]
    
        
