import networkx as nx
import numpy as np
import time


def get_shortest_path_matrix(adj):
    print("Computing all pairs shortest path lengths...")
    t_start = time.time()
    G = nx.from_numpy_matrix(adj.toarray())
    path_lengths = dict(nx.all_pairs_shortest_path_length(G))
    n = len(path_lengths)
    path_lengths_matrix = np.array([[path_lengths[i][k] for k in range(n)] for i in range(n)])

    t_elapsed = time.time() - t_start
    print("Time to compute shortest paths (s) = {:.4f}".format(t_elapsed))
    return path_lengths_matrix