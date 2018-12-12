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
