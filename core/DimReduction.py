from sklearn import manifold, decomposition
from experiments.oracle_embedding import oracle_embedding


class DimReduction(object):
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, X, method, labels=None):
        if method == 'oracle':
            return oracle_embedding(labels, shuffle=False)
        elif method == 'oracle_shuffle':
            return oracle_embedding(labels, shuffle=True)

        if method == 'spectral':
            embedder = manifold.SpectralEmbedding(n_components=self.n_components, random_state=0,
                                                  eigen_solver="arpack")
        elif method == 'tsne':
            embedder = manifold.TSNE(n_components=self.n_components, init='pca', random_state=0)
        
        elif method == 'isomap':
            embedder = manifold.Isomap(n_neighbors=30, n_components=self.n_components)
        
        elif method == 'pca':
            embedder = decomposition.TruncatedSVD(n_components=self.n_components)

        elif method == 'lle':
            embedder = manifold.LocallyLinearEmbedding(n_neighbors=30, n_components=self.n_components,
                                                       method='standard')
        elif method == 'mds':
            embedder = manifold.MDS(n_components=self.n_components, n_init=1, max_iter=100)

        else:
            raise ValueError('Solver type was not recognised.')

        return embedder.fit_transform(X)
