from sklearn.metrics.pairwise import pairwise_distances
import torch
import numpy as np
import time
from util.graph_utils import get_shortest_path_matrix
from util.training_utils import get_torch_dtype


dtypeFloat, dtypeLong = get_torch_dtype()


def Hbeta(D, beta):
    eps = 10e-15
    P = np.exp(-D * beta)
    sumP = np.maximum(np.sum(P), eps)
    H = np.log(sumP) + beta * np.sum(np.multiply(D, P)) / sumP
    P = P / sumP
    return H, P


def x2p(D, u=15, tol=1e-4, print_iter=500, max_tries=50, verbose=0):
    # Initialize some variables
    n = D.shape[0]                     # number of instances
    P = np.zeros((n, n))               # empty probability matrix
    beta = np.ones(n)                  # empty precision vector
    logU = np.log(u)                   # log of perplexity (= entropy)

    # Run over all datapoints
    if verbose > 0: print('Computing P-values...')
    for i in range(n):

        if verbose > 1 and print_iter and i % print_iter == 0:
            print('Computed P-values {} of {} datapoints...'.format(i, n))

        # Set minimum and maximum values for precision
        betamin = float('-inf')
        betamax = float('+inf')

        # Compute the Gaussian kernel and entropy for the current precision
        indices = np.concatenate((np.arange(0, i), np.arange(i + 1, n)))
        Di = D[i, indices]
        H, thisP = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while abs(Hdiff) > tol and tries < max_tries:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i]
                if np.isinf(betamax):
                    beta[i] *= 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax = beta[i]
                if np.isinf(betamin):
                    beta[i] /= 2
                else:
                    beta[i] = (beta[i] + betamin) / 2

            # Recompute the values
            H, thisP = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, indices] = thisP

    if verbose > 0:
        print('Mean value of sigma: {}'.format(np.mean(np.sqrt(1 / beta))))
        print('Minimum value of sigma: {}'.format(np.min(np.sqrt(1 / beta))))
        print('Maximum value of sigma: {}'.format(np.max(np.sqrt(1 / beta))))

    return P, beta


def compute_joint_probabilities(samples, batch_size=10000, d=2, perplexity=30, metric='euclidean', adj=None, alpha=0, tol=1e-5, verbose=0):
    # Initialize some variables
    n = samples.shape[0]
    batch_size = min(batch_size, n)

    # Precompute joint probabilities for all batches
    if verbose > 0: print('Precomputing P-values...')
    batch_count = int(n / batch_size)
    P = np.zeros((batch_count, batch_size, batch_size))
    for i, start in enumerate(range(0, n - batch_size + 1, batch_size)):
        curX = samples[start:start+batch_size]                    # select batch

        # Compute pairwise distances
        if verbose > 0: print('Computing pairwise distances...')
        if metric == 'euclidean':
            D = pairwise_distances(curX, metric=metric, squared=True)
        elif metric == 'cosine':
            D = pairwise_distances(curX, metric=metric)
        elif metric == 'shortest_path':
            assert adj is not None and alpha == 0
            D = get_shortest_path_matrix(adj.toarray(), verbose)
            alpha = 0

        # Augment distances with adjacency matrix
        if adj is not None and alpha != 0:
            W = adj.toarray()
            affinity = alpha * W * D
            D = D - affinity

        t_start = time.time()
        P[i], beta = x2p(D, perplexity, tol, verbose=verbose)      # compute affinities using fixed perplexity
        #print("Time to compute X2P = {}".format(time.time() - t_start))

        P[i][np.isnan(P[i])] = 0                                   # make sure we don't have NaN's
        P[i] = (P[i] + P[i].T)  # / 2                              # make symmetric
        P[i] = P[i] / P[i].sum()                                   # obtain estimation of joint probabilities
        P[i] = np.maximum(P[i], np.finfo(P[i].dtype).eps)

    # Reshape to 2D torch tensor
    P = P.reshape((n, n))
    P = torch.from_numpy(P).type(dtypeFloat)

    return P


def tsne_torch_loss(P, X_emb):
    d = 2
    n = P.shape[1]
    v = d - 1.  # degrees of freedom
    eps = 10e-15  # needs to be at least 10e-8 to get anything after Q /= K.sum(Q)

    # Euclidean pairwise distances in the low-dimensional map
    sum_act = torch.sum(X_emb.pow(2), dim=1)
    Q = sum_act + torch.reshape(sum_act, [-1, 1]) + -2 * torch.mm(X_emb, torch.t(X_emb))

    Q = Q / v
    Q = torch.pow(1 + Q, -(v + 1) / 2)
    Q *= 1 - torch.eye(n).type(dtypeFloat)
    Q /= torch.sum(Q)
    Q = torch.clamp(Q, min=eps)
    C = torch.log((P + eps) / (Q + eps))
    C = torch.sum(P * C)
    return C
