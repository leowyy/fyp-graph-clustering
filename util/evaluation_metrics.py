import numpy as np
from sklearn import neighbors
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


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


def evaluate_viz_metrics(y_emb, dataset, distance_metric='euclidean'):
    """
    Given a low-dimensional embedding y_emb,
    Obtains the average trustworthiness, NN accuracy, custom losses
    dataset can be either a EmbeddingDataSet or GraphDataBlock
    """
    results = {}

    # results["trust"] = trustworthiness(dataset.inputs, y_emb, n_neighbors=5, metric=distance_metric)
    results["one_nn"] = nearest_neighbours_generalisation_accuracy(y_emb, dataset.labels, 1)
    results["five_nn"] = nearest_neighbours_generalisation_accuracy(y_emb, dataset.labels, 5)
    results["graph_dist"], results["feature_dist"] = combined_dist_metric(y_emb, dataset.inputs, dataset.adj_matrix, k=5)
    results['total_dist'] = results["graph_dist"] + results["feature_dist"]

    for k, v in results.items():
        print("Metric {} = {:.4f}".format(k, v))
    return results


def graph_trustworthiness(y_emb, path_matrix, n_neighbors=5):
    dist_X_emb = pairwise_distances(y_emb, squared=True)
    ind_X_emb = np.argsort(dist_X_emb, axis=1)[:, 1:n_neighbors + 1]

    n_samples = y_emb.shape[0]
    t = 0.0
    min_sum = 0.0
    max_sum = 0.0
    for i in range(n_samples):
        ranks = path_matrix[i][ind_X_emb[i, :]]
        t += np.sum(ranks)
        lengths_from_i = sorted(path_matrix[i])
        min_sum += sum(lengths_from_i[1:n_neighbors + 1])
        max_sum += sum(lengths_from_i[-n_neighbors:])
    t = 1.0 - (t - min_sum) / (max_sum - min_sum)
    return t


def neighborhood_preservation(y_emb, path_matrix, max_graph_dist=2):
    dist_X_emb = pairwise_distances(y_emb, squared=True)
    ind_X_emb = np.argsort(dist_X_emb, axis=1)[:, 1:]

    n_samples = y_emb.shape[0]
    t = 0.0
    for i in range(n_samples):
        graph_n = {k for k, v in enumerate(path_matrix[i]) if 0 < v <= max_graph_dist}
        if len(graph_n) == 0:
            t += 1
            continue
        layout_n = set(ind_X_emb[i][:len(graph_n)])
        intersection_size = len(graph_n.intersection(layout_n))
        t += intersection_size / (2*len(graph_n) - intersection_size)
    return t/n_samples


def combined_dist_metric(y_emb, feature_matrix, W, k=5):
    from sklearn.neighbors import kneighbors_graph
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics.pairwise import pairwise_distances

    scaler = StandardScaler()
    z_emb = scaler.fit_transform(y_emb)
    z_dist_matrix = pairwise_distances(z_emb, squared=True)

    # feature_dist_matrix = pairwise_distances(feature_matrix, metric='cosine')
    knn_graph = kneighbors_graph(feature_matrix, n_neighbors=k, mode='connectivity', metric='cosine',
                                 include_self=False)

    # Average edge length in the original graph
    graph_dist = np.sum(W.toarray() * z_dist_matrix) / W.getnnz()

    # Average edge length in the kNN graph
    feature_dist = np.sum(knn_graph.toarray() * z_dist_matrix) / knn_graph.getnnz()

    return graph_dist, feature_dist


def run_regression(train_embeds, train_labels, test_embeds, test_labels):
    np.random.seed(1)
    from sklearn.linear_model import SGDClassifier
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import f1_score
    dummy = DummyClassifier()
    dummy.fit(train_embeds, train_labels)
    log = SGDClassifier(loss="log", n_jobs=10, tol=1e-3)
    log.fit(train_embeds, train_labels)
    print("F1 score:", f1_score(test_labels, log.predict(test_embeds), average="micro"))
    print("Random baseline f1 score:", f1_score(test_labels, dummy.predict(test_embeds), average="micro"))


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
