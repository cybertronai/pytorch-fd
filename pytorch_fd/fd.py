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

    def __init__(self, optimizer, epsilon=.01, gamma=0.1, last_epoch=-1, writer: SummaryWriter = None, use_mom=False):
        self.epsilon = epsilon
        self.gamma = gamma
        self.writer = writer
        self.use_mom = use_mom
        self.o = []
        self.lr = 0
        self.factor = 1
        #        self.last_epoch = -1  # have to set here because step is called before parent sets value
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
                    lr = 0.5
                    or_ = 0.5 * lr * grad.pow(2).sum()

                ratio = ol / or_ - 1
                ols.append(ol)
                ors.append(or_)
#                self.ratios.append(ratio)
        print(f"ol: {ol.item():5.4f} or: {or_.item():5.4f} {grad}")
        ratio = torch.tensor(ratios).mean()   
        self.o.append(ratio)
        half_running = torch.tensor(self.o[len(self.o)//2:]).mean()
        if self.writer:
            for data, label in ((ols, 'ol'), (ors, 'or'), (ratios, 'ratio')):
                tensor = torch.tensor(data)
                #self.writer.add_histogram(f'fd/{label}', tensor, self.last_epoch)
                self.writer.add_scalar(f'fd/{label}', tensor.mean(), self.last_epoch)
            self.writer.add_scalar('fd/half_running', half_running, self.last_epoch)
        if half_running.abs() < self.epsilon:
            self.factor = self.gamma
        else:
            self.factor = 1


    def get_lr(self):
        if self.last_epoch <= 1:
            return self.base_lrs

        return [group['lr'] * self.factor
                for group in self.optimizer.param_groups]
    
        
