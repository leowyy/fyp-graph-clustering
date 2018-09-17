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


class SimpleConvNet(nn.Module):
    """
    Implements a simple conv net with the following architecture
    2 conv layers with ReLU, batch norm and max pooling
    """
    def __init__(self, net_parameters):
        super(SimpleConvNet, self).__init__()

        n_channels = net_parameters['n_channels']
        n_units_1 = net_parameters['n_units_1']
        n_units_2 = net_parameters['n_units_2']
        n_components = net_parameters['n_components']

        self.conv1 = nn.Conv2d(n_channels, n_units_1, 5, 1, 2)  # n_channels, n_output, ks, stride, padding
        self.conv2 = nn.Conv2d(n_units_1, n_units_2, 5, 1, 2)

        self.norm1 = nn.BatchNorm2d(n_units_1)
        self.norm2 = nn.BatchNorm2d(n_units_2)

        self.fc = nn.Linear(n_units_2 * 7 * 7, n_components)  # 28 -> 14 -> 7

        # init
        self.init_weights_Graph_OurConvNet(n_units_2 * 7 * 7, n_components, 1)

    def forward(self, G):
        # Data matrix
        x = G.data

        # Pass raw data matrix X directly as input
        x = Variable(torch.FloatTensor(x).type(dtypeFloat), requires_grad=False)

        out = F.relu(self.conv1(x), inplace=True)
        out = self.norm1(out)
        out = F.max_pool2d(out, 3, 2, 1)  # ks, stride, padding

        out = F.relu(self.conv2(out), inplace=True)
        out = self.norm2(out)
        out = F.max_pool2d(out, 3, 2, 1)

        # Unroll output into a single vector
        out = out.view(out.size(0), -1)

        out = self.fc(out)
        return out

    def init_weights_Graph_OurConvNet(self, Fin_fc, Fout_fc, gain):

        scale = gain * np.sqrt(2.0 / Fin_fc)
        self.fc.weight.data.uniform_(-scale, scale)
        self.fc.bias.data.fill_(0)

    def loss(self, y, y_target):
        # L2 loss
        loss = nn.MSELoss()(y, y_target)

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
