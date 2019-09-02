import torch
import numpy as np
from scipy.sparse.csgraph import laplacian


if torch.cuda.is_available():
    dtypeFloat = torch.cuda.FloatTensor
    dtypeLong = torch.cuda.LongTensor
    dtypeDouble = torch.cuda.DoubleTensor
else:
    dtypeFloat = torch.FloatTensor
    dtypeLong = torch.LongTensor
    dtypeDouble = torch.DoubleTensor


def graph_cut_torch_loss(adj, X_emb):
    L = laplacian(adj, normed=False, return_diag=False)
    L = torch.from_numpy(L.toarray()).type(dtypeFloat)

    cut = torch.trace(torch.mm(torch.mm(torch.t(X_emb), L), X_emb))

    return cut


def covariance_constraint(adj, X_emb):
    D = np.sum(adj.toarray(), axis=1)
    D = np.diag(D)
    D = torch.from_numpy(D).type(dtypeFloat)
    diff = torch.mm(torch.mm(torch.t(X_emb), D), X_emb) - torch.eye(X_emb.shape[1])
    loss = torch.mean(torch.pow(diff.norm(dim=1), 2))

    return loss
