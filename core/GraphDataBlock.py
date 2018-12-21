import numpy as np
import scipy.sparse as sp
from sklearn import manifold
import torch


class GraphDataBlock(object):
    def __init__(self, X, labels, W=None):
        # Convert to torch if numpy array
        if type(X) is np.ndarray:
            X = torch.from_numpy(X).type(torch.FloatTensor)

        # Unroll each feature into a single vector
        X_unrolled = X.view(X.shape[0], -1).numpy()

        # Get affinity matrix
        if W is None:
            embedder = manifold.SpectralEmbedding(n_components=2)
            W = embedder._get_affinity_matrix(X_unrolled)

        W = sp.coo_matrix(W)  # sparse matrix

        # Get edge information
        nb_edges = W.nnz
        nb_vertices = W.shape[0]
        self.edge_to_starting_vertex = sp.coo_matrix((np.ones(nb_edges), (np.arange(nb_edges), W.row)),
                                                shape=(nb_edges, nb_vertices))
        self.edge_to_ending_vertex = sp.coo_matrix((np.ones(nb_edges), (np.arange(nb_edges), W.col)),
                                              shape=(nb_edges, nb_vertices))

        # Save as attributes
        self.data = X           # data matrix
        self.target = []        # embedding matrix
        self.labels = labels    # labels
        self.adj_matrix = W     # affinity matrix
