import numpy as np
import scipy.sparse as sp
from sklearn import manifold
from timeit import default_timer as timer


class DataEmbeddingGraph(object):
    def __init__(self, X, labels, reduction_method='spectral'):
        # Unrolled image vectors
        X_unrolled = X.view(X.shape[0], -1).numpy()

        # Get affinity matrix
        embedder = manifold.SpectralEmbedding(n_components=2, random_state=0,
                                              eigen_solver="arpack")
        W = embedder._get_affinity_matrix(X_unrolled)
        W = sp.coo_matrix(W)  # sparse matrix

        if reduction_method == 'spectral':
            pass
        elif reduction_method == 'tsne':
            embedder = manifold.TSNE(n_components=2, init='pca', random_state=0)
        else:
            raise ValueError('Solver type was not recognised.')

        # Get embedding matrix
        start = timer()
        y = embedder.fit_transform(X_unrolled)
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
