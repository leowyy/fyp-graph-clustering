import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np

from core.GraphConvNetCell import GraphConvNetCell


if torch.cuda.is_available():
    dtypeFloat = torch.cuda.FloatTensor
    dtypeLong = torch.cuda.LongTensor
else:
    dtypeFloat = torch.FloatTensor
    dtypeLong = torch.LongTensor


class GraphConvNet(nn.Module):

    def __init__(self, net_parameters, task_parameters):

        super(GraphConvNet, self).__init__()

        # parameters
        flag_task = task_parameters['flag_task']
        Voc = net_parameters['Voc']
        D = net_parameters['D']
        nb_clusters_target = net_parameters['nb_clusters_target']
        H = net_parameters['H']
        L = net_parameters['L']

        # vector of hidden dimensions
        net_layers = []
        for layer in range(L):
            net_layers.append(H)

        # embedding
        self.encoder = nn.Embedding(Voc, D)

        # CL cells
        # NOTE: Each graph convnet cell uses *TWO* convolutional operations
        net_layers_extended = [D] + net_layers  # include embedding dim
        L = len(net_layers)
        list_of_gnn_cells = []  # list of NN cells
        for layer in range(L // 2):
            Hin, Hout = net_layers_extended[2 * layer], net_layers_extended[2 * layer + 2]
            list_of_gnn_cells.append(GraphConvNetCell(Hin, Hout))

        # register the cells for pytorch
        self.gnn_cells = nn.ModuleList(list_of_gnn_cells)

        # fc
        Hfinal = net_layers_extended[-1]
        self.fc = nn.Linear(Hfinal, nb_clusters_target)

        # init
        self.init_weights_Graph_OurConvNet(Voc, D, Hfinal, nb_clusters_target, 1)

        # print
        print('\nnb of hidden layers=', L)
        print('dim of layers (w/ embed dim)=', net_layers_extended)
        print('\n')

        # class variables
        self.L = L
        self.net_layers_extended = net_layers_extended
        self.flag_task = flag_task
        self.tracker = []

    def init_weights_Graph_OurConvNet(self, Fin_enc, Fout_enc, Fin_fc, Fout_fc, gain):

        scale = gain * np.sqrt(2.0 / Fin_enc)
        self.encoder.weight.data.uniform_(-scale, scale)
        scale = gain * np.sqrt(2.0 / Fin_fc)
        self.fc.weight.data.uniform_(-scale, scale)
        self.fc.bias.data.fill_(0)

    def forward(self, G):

        # signal
        x = G.signal  # V-dim
        x = Variable(torch.LongTensor(x).type(dtypeLong), requires_grad=False)

        # encoder
        x_emb = self.encoder(x)  # V x D

        # Extract first embedding layer
        self.tracker.append(x_emb)

        # graph operators
        # Edge = start vertex to end vertex
        # E_start = E x V mapping matrix from edge index to corresponding start vertex
        # E_end = E x V mapping matrix from edge index to corresponding end vertex
        E_start = G.edge_to_starting_vertex
        E_end = G.edge_to_ending_vertex
        E_start = torch.from_numpy(E_start.toarray()).type(dtypeFloat)
        E_end = torch.from_numpy(E_end.toarray()).type(dtypeFloat)
        E_start = Variable(E_start, requires_grad=False)
        E_end = Variable(E_end, requires_grad=False)

        # convnet cells
        x = x_emb
        for layer in range(self.L // 2):
            gnn_layer = self.gnn_cells[layer]
            x = gnn_layer(x, E_start, E_end)  # V x Hfinal
            # Extract embedding layer
            self.tracker.append(x)

        # FC
        x = self.fc(x)

        return x

    def loss(self, y, y_target, weight):

        loss = nn.CrossEntropyLoss(weight=weight.type(dtypeFloat))(y, y_target)

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

    def add_tracker(self, tracker):
        self.tracker = tracker

    def get_tracker(self):
        return self.tracker
