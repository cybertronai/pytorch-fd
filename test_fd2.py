"""MNIST example.

Based on https://github.com/pytorch/examples/blob/master/mnist/main.py
"""

from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from pytorch_fd import FDLR


def toscalar(t):  # use on python scalars/pytorch scalars
    """Converts Python scalar or PyTorch tensor to Python scalar"""
    if isinstance(t, (float, int)): return t
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]


def to_pytorch(t):
    x = np.array(t).astype(np.float32)
    return torch.from_numpy(x)


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        W0 = [[0, 0]]
        self.W = nn.Parameter(to_pytorch(W0))

    def forward(self, x):
        Y = to_pytorch([[1, 1]])
        err = self.W @ x.reshape((2, 1)) - Y
        dsize = 2
        loss = err @ err.transpose(0, 1) / dsize / 2
        loss = loss.reshape(-1)[0]
        return loss


def train(model, optimizer, scheduler):
    model.train()
    indices = [0, 1, 0, 1, 0, 1, 0, 1]
    for i in range(len(indices)):
        data = X[:, indices[i]]
        loss = model(data)
        print("loss", loss.item())
        print("params", model.W.data)
        optimizer.zero_grad()
        loss.backward()
        print("gradient1", model.W.grad.data)
        scheduler.step()
        optimizer.step()
        print()



X = to_pytorch([[1, 0], [0, 1]])


def main():
    model = SimpleNet()
    writer = SummaryWriter()

    optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0)
    scheduler = FDLR(optimizer, epsilon=0, gamma=0, writer=writer, use_mom=False)
    train(model, optimizer, scheduler)


if __name__ == '__main__':
    main()
