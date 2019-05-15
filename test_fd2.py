"""MNIST example.

Based on https://github.com/pytorch/examples/blob/master/mnist/main.py
"""

from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import sys

from pytorch_fd import FDLR

# Expected results

# {
#  {0.5, 0, 0.25},
#  {0.5, 0., 0.25},
#  {0.125, -0.25, 0.0625},
#  {0.125, -0.25, 0.0625},
#  {0.03125, -0.1875, 0.015625},
#  {0.03125, -0.1875, 0.015625},
#  {0.0078125, -0.109375, 0.00390625},
#  {0.0078125, -0.109375, 0.00390625}
# }

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
        return self.W @ x.reshape((2, 1))


def train(model, optimizer, scheduler):
    X = to_pytorch([[1, 0], [0, 1]])
    Y = to_pytorch([[1, 1]])

    model.train()
    indices = [0, 1, 0, 1, 0, 1, 0, 1]

    for i in range(len(indices)):
        data = X[:, indices[i]]
        Yt = Y[:,indices[i]]
        Yp = model(data)
        err = Yp - Yt
        dsize = 2
        loss = err @ err.transpose(0, 1) / 2
        loss = loss.reshape(-1)[0]
        sys.stdout.write(f"loss: {loss.item():5.4f} ")
        optimizer.zero_grad()
        loss.backward()
        scheduler.step()
        optimizer.step()




def main():
    model = SimpleNet()
    writer = SummaryWriter()

    print('model.W', model.W)
    print('model.parameters', list(model.parameters()))
    optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0)
    print('optimizer.param_groups', optimizer.param_groups)
    scheduler = FDLR(optimizer, epsilon=0, gamma=0, writer=writer, use_mom=False)
    print('optimizer.param_groups2', optimizer.param_groups)
    train(model, optimizer, scheduler)


if __name__ == '__main__':
    main()
