import numpy as np
from timeit import default_timer as timer

from sklearn import neighbors
from sklearn.manifold.t_sne import trustworthiness
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from core.DimReduction import DimReduction


def one_nearest_neighbours_generalisation_accuracy(X, y):
    """
    Obtains the average 10-fold validation accuracy of a 1-NN classifier trained on the given embeddings
    """
    kf = KFold(n_splits=10)
    kf.get_n_splits(X)
    errors = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = neighbors.KNeighborsClassifier(n_neighbors=1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        errors.append(accuracy_score(y_test, y_pred))
    return np.average(errors)


def evaluate_net_metrics(all_test_data, net):
    """
    Given a projection net,
    Obtains the average trustworthiness, 1-NN accuracy and time taken to compute
    all_test_data should be a list of DataEmbeddingGraph objects
    """
    n_test = len(all_test_data)
    trust_tracker = np.zeros((n_test,))
    one_nn_tracker = np.zeros((n_test,))
    time_tracker = np.zeros((n_test,))

    for i in range(n_test):
        G = all_test_data[i]
        time_start = timer()
        y_pred = net.forward(G).detach().numpy()
        time_end = timer()
        time_tracker[i] = time_end - time_start

        X = G.data.view(G.data.shape[0], -1).numpy()
        trust_tracker[i] = trustworthiness(X, y_pred, n_neighbors=5)
        one_nn_tracker[i] = one_nearest_neighbours_generalisation_accuracy(y_pred, G.labels.numpy())
    return np.average(trust_tracker), np.average(one_nn_tracker), np.average(time_tracker)


def evaluate_embedding_metrics(all_test_data, method):
    """
    Given an embedding method,
    Obtains the average trustworthiness, 1-NN accuracy and time taken to compute
    all_test_data should be a list of DataEmbeddingGraph objects
    """
    dim_red = DimReduction(n_components=2)
    n_test = len(all_test_data)
    trust_tracker = np.zeros((n_test,))
    one_nn_tracker = np.zeros((n_test,))
    time_tracker = np.zeros((n_test,))

    for i in range(n_test):
        G = all_test_data[i]
        X = G.data.view(G.data.shape[0], -1).numpy()
        time_start = timer()
        X_emb = dim_red.fit_transform(X, method)
        time_end = timer()
        time_tracker[i] = time_end - time_start

        trust_tracker[i] = trustworthiness(X, X_emb, n_neighbors=5)
        one_nn_tracker[i] = one_nearest_neighbours_generalisation_accuracy(X_emb, G.labels.numpy())
    return np.average(trust_tracker), np.average(one_nn_tracker), np.average(time_tracker)