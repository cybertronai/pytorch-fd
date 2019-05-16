"""MNIST example.

Based on https://github.com/pytorch/examples/blob/master/mnist/main.py
"""

from __future__ import print_function

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.distributions import MultivariateNormal
from torch.distributions import Normal

from pytorch_fd import FDLR


class Net(nn.Module):
    def __init__(self, dims=2):
        super(Net, self).__init__()
        w0 = to_pytorch([1] * dims)
        self.w = nn.Parameter(w0)

    def forward(self, x):
        return dot(self.w, x)


def to_pytorch(t):
    x = np.array(t).astype(np.float32)
    return torch.from_numpy(x)


def dot(a, b):
    return (a * b).sum()


def norm_squared(a):
    a = a.reshape(-1)
    return (a * a).sum()


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--scheduler', type=str, default='fdlr', choices=['cosine', 'fd', 'step'],
                        help='which scheduler to use')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['adam', 'sgd'],
                        help='which scheduler to use')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=60, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--save-checkpoint', action='store_true')
    parser.add_argument('--load-checkpoint', action='store_true')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 0.0025)')
    parser.add_argument('--momentum', type=float, default=0, metavar='LR',
                        help='learning rate (default: 0.0025)')
    parser.add_argument('--seed', type=int, default=2, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--use-least-squares', type=int, default=1,
                        help='use least squares instead of cross entropy')
    parser.add_argument('--run', type=str, default='fd',
                        help='name of logging run')

    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    writer = SummaryWriter()

    model = Net()
    assert args.optimizer == 'sgd'
    assert args.momentum == 0
    assert args.scheduler == 'fdlr'

    optimizer = optim.SGD(model.parameters(), lr=0.0025, momentum=0)
    scheduler = FDLR(optimizer, epsilon=.01, gamma=0.1, writer=writer,
                     use_mom=False, lr=args.lr)

    model.train()

    dims = 2
    v2 = 0.1
    offdiag = torch.ones((dims, dims)) - torch.eye(dims)
    offset = 0.1  # 0 means singular, 1 is standard normal
    covmat = torch.eye(dims) + (1 - offset) * offdiag
    mean = torch.zeros(dims)
    mean = mean.to(device)
    covmat = covmat.to(device)
    Xs = MultivariateNormal(mean, covmat)
    Ye = Normal(to_pytorch(0).to(device),
                to_pytorch(v2).to(device))
    wt = to_pytorch([1, 1]).to(device)
    batch_idx = 0
    time0 = time.time()
    iter = 0
    while True:
        x = Xs.sample()
        yerror = Ye.sample()
        y = dot(x, wt) + yerror
        optimizer.zero_grad()
        loss = 0.5 * norm_squared(y - model(x))
        if time.time()-time0>2:
            half = scheduler.half_running_ol/scheduler.half_running_or
            full = scheduler.ol_sum/scheduler.or_sum
            print(f"iter {iter:05d} loss {loss.item():.2f} half_running {half.item():.2f} full_avg {full:.2f}")
            time0 = time.time()


        loss.backward()
        scheduler.step()
        optimizer.step()
        iter+=1



if __name__ == '__main__':
    main()
