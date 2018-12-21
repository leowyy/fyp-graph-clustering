import numpy as np


def oracle_embedding(labels, shuffle=False):
    means = [(0, 0), (1, 0), (2, 0), (3, 0),
             (0, 1), (1, 1), (2, 1),
             (0, 2), (1, 2), (2, 2), ]
    means = [np.array(m) for m in means]

    if shuffle:
        np.random.shuffle(means)

    def get_num_coordinates(label):
        mean = means[label]
        cov = [[0.05, 0], [0, 0.05]]
        coord = np.random.multivariate_normal(mean, cov, 1)
        return coord

    X_emb = np.zeros((len(labels), 2))
    for i, l in enumerate(labels):
        X_emb[i, :] = get_num_coordinates(l)

    return X_emb
