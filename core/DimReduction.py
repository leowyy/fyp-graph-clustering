from sklearn import manifold


class DimReduction(object):
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, X, solver_type, n_neighbors=None):
        if solver_type.lower() == "spectral":
            embedder = manifold.SpectralEmbedding(n_components=self.n_components, random_state=0,
                                                  eigen_solver="arpack")
        elif solver_type.lower() == "tsne":
            embedder = manifold.TSNE(n_components=self.n_components, init='pca', random_state=0)

        elif solver_type.lower() == "isomap":
            embedder = manifold.Isomap(n_neighbors, n_components=self.n_components)

        else:
            raise ValueError('Solver type was not recognised.')

        return embedder.fit_transform(X)
