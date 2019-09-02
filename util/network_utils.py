import torch
import numpy as np
from util.graph_utils import neighbor_sampling
from core.GraphDataBlock import GraphDataBlock
from tqdm import tqdm


def save_checkpoint(state, filename):
    torch.save(state, filename)


def get_net_projection(net, dataset, n_batches=1, n_components=2):
    """
    Get visualization of dataset using a projection net
    Args:
        net (GraphConvNet): projection net
        dataset (EmbeddingDataSet): dataset to project
        n_batches (int): number of batches to split dataset
        n_components (int): dimensional of output
    Returns:
        y_pred (np.array): low dimensional map of data points, matrix of size n x n_components
    """
    net.eval()

    dataset.create_all_data(n_batches=n_batches, shuffle=False)
    if n_batches == 1:
        return _get_net_projection(net, dataset.all_data[0])

    y_pred = np.zeros((len(dataset.labels), n_components))
    for G in tqdm(dataset.all_data):
        y_pred_original = _get_net_projection(net, G, sampling=False, dataset=dataset)
        y_pred[G.original_indices] = y_pred_original  # Place results into full matrix
    return y_pred


def _get_net_projection(net, G, sampling=False, dataset=None):
    """
    Helper function for get_net_projection
    Args:
        net (GraphConvNet): projection net
        G (GraphDataBlock): graph block to project
        sampling (Boolean): whether to expand the graph block via neighbor sampling
        dataset (EmbeddingDataSet): provided as input to perform neighbor sampling
    Returns:
        y_pred_original (np.array): low dimensional map of data points, matrix of size n x n_components
    """
    # if torch.cuda.is_available():
    #     y_pred_neighborhood = net.forward(G).cpu().detach().numpy()
    if not sampling:
        return net.forward(G).detach().numpy()

    is_sorted = lambda a: np.all(a[:-1] <= a[1:])
    assert is_sorted(G.original_indices)
    assert dataset is not None

    original_idx = G.original_indices
    neighborhood_idx = neighbor_sampling(dataset.adj_matrix, original_idx, [-1, -1])

    # Package into GraphDataBlock
    inputs_subset = dataset.inputs[neighborhood_idx]
    labels_subset = dataset.labels[neighborhood_idx]
    adj_subset = dataset.adj_matrix[neighborhood_idx, :][:, neighborhood_idx]
    G = GraphDataBlock(inputs_subset, labels=labels_subset, W=adj_subset)

    # Get projection of the expanded GraphDataBlock, without sampling this time
    y_pred_neighborhood = _get_net_projection(net, G, sampling=False)

    # Get mask of indices of original within neighborhood
    ix = np.isin(neighborhood_idx, original_idx)

    # Retrieve predictions for original indices only
    y_pred_original = y_pred_neighborhood[ix]

    return y_pred_original


def get_net_embeddings(net, dataset, net_type, H=50):
    # Use the model object to select the desired layer
    if net_type == 'graph':
        layer = net._modules['gnn_cells'][0]
    elif net_type == 'simple':
        layer = net._modules['relu']

    # Get the total number of data points
    n = len(dataset.labels)

    # Define a function that will copy the output of a layer
    my_embedding = torch.zeros([n, H])

    def copy_data(m, i, o):
        my_embedding.copy_(o.data)

    # Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)

    # Perform projection to capture embeddings
    get_net_projection(net, dataset)

    return my_embedding
