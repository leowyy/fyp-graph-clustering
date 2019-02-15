import networkx as nx
import numpy as np
import time


MAX_DISTANCE = 1e6


def get_shortest_path_matrix(adj, verbose=0):
    n = adj.shape[0]
    if verbose:
        print("Computing all pairs shortest path lengths for {} nodes...".format(n))
    t_start = time.time()
    G = nx.from_numpy_matrix(adj)
    path_lengths = dict(nx.all_pairs_shortest_path_length(G))
    path_lengths_matrix = np.array([[path_lengths[i].get(k, MAX_DISTANCE) for k in range(n)] for i in range(n)])

    t_elapsed = time.time() - t_start
    if verbose:
        print("Time to compute shortest paths (s) = {:.4f}".format(t_elapsed))
    return path_lengths_matrix


def neighbor_sampling(adj, minibatch_indices, D_layers):
    """
    Args:
        adj: adjacency matrix of the COMPLETE graph in csr format
        minibatch_indices: indices subset of all vertices
        D_layers: sampling strategy per layer, default D_layers=[5, 10],
                    if D_layers[l] = -1, sample all neighbors at layer l
    """
    selected_indices = list(minibatch_indices)
    for i in minibatch_indices:
        one_hop_neighbors = adj[i].nonzero()[1]
        if D_layers[0] != -1 and len(one_hop_neighbors) > D_layers[0]:
            one_hop_neighbors = np.random.choice(one_hop_neighbors, size=D_layers[0], replace=False)

        selected_indices += list(one_hop_neighbors)

        for j in one_hop_neighbors:
            two_hop_neighbors = adj[j].nonzero()[1]
            if  D_layers[1] != -1 and len(two_hop_neighbors) > D_layers[1]:
                two_hop_neighbors = np.random.choice(two_hop_neighbors, size=D_layers[1], replace=False)

            selected_indices += list(two_hop_neighbors)
    return np.unique(np.array(selected_indices))