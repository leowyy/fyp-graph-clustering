import matplotlib.pyplot as plt
import numpy as np


# Functions for plotting data with labels 0-9
def plot_embedding(X, labels=None, title=None):
    plt.figure()
    ax = plt.subplot(111)
    plot_embedding_subplot(ax, X, labels, title)


def plot_embedding_subplot(ax, X, labels=None, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    if labels is not None:
        for i in range(X.shape[0]):
            ax.text(X[i, 0], X[i, 1], str(labels[i]),
                    color=plt.cm.Set1(labels[i] / 10.),
                    fontdict={'weight': 'bold', 'size': 9})

    ax.set_xticks([]), ax.set_yticks([])
    if title is not None:
        ax.set_title(title)
