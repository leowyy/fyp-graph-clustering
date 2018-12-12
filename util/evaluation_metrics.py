import numpy as np
from timeit import default_timer as timer
import torch

from sklearn import neighbors
from sklearn.manifold.t_sne import trustworthiness
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from core.DimReduction import DimReduction


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


def evaluate_net_metrics(all_test_data, net):
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
        trust_tracker[i] = trustworthiness(X, y_pred, n_neighbors=5)
        one_nn_tracker[i] = nearest_neighbours_generalisation_accuracy(y_pred, G.labels.numpy(), 1)
        five_nn_tracker[i] = nearest_neighbours_generalisation_accuracy(y_pred, G.labels.numpy(), 5)
    return np.average(trust_tracker), np.average(one_nn_tracker), np.average(five_nn_tracker), np.average(time_tracker)


def evaluate_embedding_metrics(all_test_data, embedder):
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

        trust_tracker[i] = trustworthiness(X, X_emb, n_neighbors=5)
        one_nn_tracker[i] = nearest_neighbours_generalisation_accuracy(X_emb, G.labels.numpy(), 1)
        five_nn_tracker[i] = nearest_neighbours_generalisation_accuracy(X_emb, G.labels.numpy(), 5)
    return np.average(trust_tracker), np.average(one_nn_tracker), np.average(five_nn_tracker), np.average(time_tracker)
