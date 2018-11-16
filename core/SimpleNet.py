import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

if torch.cuda.is_available():
    dtypeFloat = torch.cuda.FloatTensor
    dtypeLong = torch.cuda.LongTensor
else:
    dtypeFloat = torch.FloatTensor
    dtypeLong = torch.LongTensor


class SimpleNet(nn.Module):
    """
    Implements a simple feed-forward net
    """

    def __init__(self, net_parameters):
        super(SimpleNet, self).__init__()

        self.name = 'simple_net'

        n_units_1 = 300
        n_units_2 = 50
        input_size = net_parameters['D']
        n_components = net_parameters['n_components']

        self.fc1 = nn.Linear(input_size, n_units_1)
        self.fc2 = nn.Linear(n_units_1, n_units_2)
        self.fc3 = nn.Linear(n_units_2, n_components)


    def forward(self, G):
        # Data matrix
        x = G.data

        # Unroll into single vector
        x = x.view(x.shape[0], -1)

        # Pass raw data matrix X directly as input
        x = Variable(torch.FloatTensor(x).type(dtypeFloat), requires_grad=False)

        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

    def loss(self, y, y_target):
        # L2 loss
        loss = nn.MSELoss()(y, y_target)

        return loss

    def pairwise_loss(self, y, y_target, W):
        distances_1 = y_target[W.row, :] - y_target[W.col, :]
        distances_2 = y[W.row, :] - y[W.col, :]
        loss = torch.mean(torch.pow(distances_1.norm(dim=1) - distances_2.norm(dim=1), 2))

        return loss

    def update(self, lr):
        update = torch.optim.Adam(self.parameters(), lr=lr)

        return update

    def update_learning_rate(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return optimizer

    def nb_param(self):
        return self.nb_param
