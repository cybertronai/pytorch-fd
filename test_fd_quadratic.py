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


def train(args, model, device, optimizer, epoch, event_writer, scheduler):
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
    while True:
        x = Xs.sample()
        yerror = Ye.sample()
        y = dot(x, wt) + yerror
        optimizer.zero_grad()
        loss = 0.5 * norm_squared(y - model(x))
        if time.time()-time0>10:
            print(loss.item())
            time0 = time.time()
            
        loss.backward()
        scheduler.step()
        optimizer.step()
        batch_idx += 1
        if batch_idx % args.log_interval == 0:
            step = (batch_idx + 1)
            event_writer.add_scalar('loss', loss, scheduler.last_epoch)
            event_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], scheduler.last_epoch)


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

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    writer = SummaryWriter()

    if args.load_checkpoint:
        model = torch.load('model.pt')
    else:
        model = Net()
    model = model.to(device)
    assert args.optimizer == 'sgd'
    assert args.momentum == 0
    assert args.scheduler == 'fdlr'

    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(.9, .99))
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, int(1e6))
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
    else:
        scheduler = FDLR(optimizer, epsilon=.01, gamma=0.1, writer=writer, use_mom=args.optimizer == 'adam', lr=args.lr)

    try:
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, optimizer, epoch, writer, scheduler)
    except KeyboardInterrupt:
        print('Exit early')

    if args.save_checkpoint:
        with open('model.pt', 'wb') as f:
            torch.save(model, f)


if __name__ == '__main__':
    main()
