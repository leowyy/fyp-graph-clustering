import numpy as np
from timeit import default_timer as timer
import torch

from sklearn import neighbors
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from core.DimReduction import DimReduction


def trustworthiness(X, X_embedded, n_neighbors=5, metric='euclidean', precomputed=False):
    """Expresses to what extent the local structure is retained.

    The trustworthiness is within [0, 1]. It is defined as

    .. math::

        T(k) = 1 - \frac{2}{nk (2n - 3k - 1)} \sum^n_{i=1}
            \sum_{j \in U^{(k)}_i} (r(i, j) - k)

    where :math:`r(i, j)` is the rank of the embedded datapoint j
    according to the pairwise distances between the embedded datapoints,
    :math:`U^{(k)}_i` is the set of points that are in the k nearest
    neighbors in the embedded space but not in the original space.

    * "Neighborhood Preservation in Nonlinear Projection Methods: An
      Experimental Study"
      J. Venna, S. Kaski
    * "Learning a Parametric Embedding by Preserving Local Structure"
      L.J.P. van der Maaten

    Parameters
    ----------
    X : array, shape (n_samples, n_features) or (n_samples, n_samples)
        If the metric is 'precomputed' X must be a square distance
        matrix. Otherwise it contains a sample per row.

    X_embedded : array, shape (n_samples, n_components)
        Embedding of the training data in low-dimensional space.

    n_neighbors : int, optional (default: 5)
        Number of neighbors k that will be considered.

    precomputed : bool, optional (default: False)
        Set this flag if X is a precomputed square distance matrix.

    Returns
    -------
    trustworthiness : float
        Trustworthiness of the low-dimensional embedding.
    """
    if precomputed:
        dist_X = X
    elif metric == 'cosine':
        dist_X = pairwise_distances(X, metric='cosine')
    else:
        dist_X = pairwise_distances(X, squared=True)
    dist_X_embedded = pairwise_distances(X_embedded, squared=True)
    ind_X = np.argsort(dist_X, axis=1)
    ind_X_embedded = np.argsort(dist_X_embedded, axis=1)[:, 1:n_neighbors + 1]

    n_samples = X.shape[0]
    t = 0.0
    ranks = np.zeros(n_neighbors)
    for i in range(n_samples):
        for j in range(n_neighbors):
            ranks[j] = np.where(ind_X[i] == ind_X_embedded[i, j])[0][0]
        ranks -= n_neighbors
        t += np.sum(ranks[ranks > 0])
    t = 1.0 - t * (2.0 / (n_samples * n_neighbors *
                          (2.0 * n_samples - 3.0 * n_neighbors - 1.0)))
    return t


def nearest_neighbours_generalisation_accuracy(X, y, n_neighbors=1):
    """
    Obtains the average 10-fold validation accuracy of a NN classifier trained on the given embeddings
    """
    kf = KFold(n_splits=10)
    kf.get_n_splits(X)
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = neighbors.KNeighborsClassifier(n_neighbors)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        scores.append(accuracy_score(y_test, y_pred))
    return np.average(scores)


def evaluate_net_metrics(all_test_data, net, distance_metric='euclidean'):
    """
    Given a embedding net,
    Obtains the average trustworthiness, NN accuracy and time taken to compute
    all_test_data should be a list of DataEmbeddingGraph objects
    """
    net.eval()
    n_test = len(all_test_data)
    trust_tracker = np.zeros((n_test,))
    one_nn_tracker = np.zeros((n_test,))
    five_nn_tracker = np.zeros((n_test,))
    time_tracker = np.zeros((n_test,))

    for i in range(n_test):
        G = all_test_data[i]
        time_start = timer()
        if torch.cuda.is_available():   
            y_pred = net.forward(G).cpu().detach().numpy()
        else:    
            y_pred = net.forward(G).detach().numpy()
        time_end = timer()
        time_tracker[i] = time_end - time_start

        X = G.data.view(G.data.shape[0], -1).numpy()
        trust_tracker[i] = trustworthiness(X, y_pred, n_neighbors=5, metric=distance_metric)
        one_nn_tracker[i] = nearest_neighbours_generalisation_accuracy(y_pred, G.labels.numpy(), 1)
        five_nn_tracker[i] = nearest_neighbours_generalisation_accuracy(y_pred, G.labels.numpy(), 5)
    return np.average(trust_tracker), np.average(one_nn_tracker), np.average(five_nn_tracker), np.average(time_tracker)


def evaluate_embedding_metrics(all_test_data, embedder, distance_metric='euclidean'):
    """
    Given an embedder,
    Obtains the average trustworthiness, NN accuracy and time taken to compute
    all_test_data should be a list of DataEmbeddingGraph objects
    """
    #dim_red = DimReduction(n_components=2)
    n_test = len(all_test_data)
    trust_tracker = np.zeros((n_test,))
    one_nn_tracker = np.zeros((n_test,))
    five_nn_tracker = np.zeros((n_test,))
    time_tracker = np.zeros((n_test,))

    for i in range(n_test):
        G = all_test_data[i]
        X = G.data.view(G.data.shape[0], -1).numpy()  # unroll into a single vector
        time_start = timer()
        #X_emb = dim_red.fit_transform(X, method)
        X_emb = embedder.fit_transform(X)
        time_end = timer()
        time_tracker[i] = time_end - time_start

        trust_tracker[i] = trustworthiness(X, X_emb, n_neighbors=5, metric=distance_metric)
        one_nn_tracker[i] = nearest_neighbours_generalisation_accuracy(X_emb, G.labels.numpy(), 1)
        five_nn_tracker[i] = nearest_neighbours_generalisation_accuracy(X_emb, G.labels.numpy(), 5)
    return np.average(trust_tracker), np.average(one_nn_tracker), np.average(five_nn_tracker), np.average(time_tracker)
