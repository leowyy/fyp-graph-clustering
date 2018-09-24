import numpy as np
import scipy.sparse as sp
from sklearn import manifold
from timeit import default_timer as timer

from core.DimReduction import DimReduction


class DataEmbeddingGraph(object):
    def __init__(self, X, labels, method='spectral'):
        # Unrolled image vectors
        X_unrolled = X.view(X.shape[0], -1).numpy()

        # Get affinity matrix
        embedder = manifold.SpectralEmbedding(n_components=2, random_state=0,
                                              eigen_solver="arpack")
        W = embedder._get_affinity_matrix(X_unrolled)
        W = sp.coo_matrix(W)  # sparse matrix

        # Reduction method
        dim_red = DimReduction(n_components=2)
        int_labels = [int(l) for l in labels]

        # Get embedding matrix
        start = timer()
        y = dim_red.fit_transform(X_unrolled, method, int_labels)
        end = timer()

        # Get edge information
        nb_edges = W.nnz
        nb_vertices = W.shape[0]
        edge_to_starting_vertex = sp.coo_matrix((np.ones(nb_edges), (np.arange(nb_edges), W.row)),
                                                shape=(nb_edges, nb_vertices))
        edge_to_ending_vertex = sp.coo_matrix((np.ones(nb_edges), (np.arange(nb_edges), W.col)),
                                              shape=(nb_edges, nb_vertices))

        # Save as attributes
        self.data = X  # data matrix
        self.target = y  # embedding matrix
        self.labels = labels  # labels
        self.adj_matrix = W  # affinity matrix
        self.edge_to_starting_vertex = edge_to_starting_vertex
        self.edge_to_ending_vertex = edge_to_ending_vertex
        self.time_to_compute = end - start
