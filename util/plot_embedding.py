import numpy as np
from bokeh.palettes import Category20_20, Category20b_20, Accent8
import matplotlib.pyplot as plt


def plot_embedding(y_emb, labels, s=7, ax=None, title=""):
    labels = np.array([int(l) for l in labels])

    colormap = np.array(Category20_20 + Category20b_20 + Accent8)

    if ax is None:
        f, ax = plt.subplots(1, sharex='col', figsize=(10, 8), dpi=300)
    else:
        ax.set_axis_off()

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title(title)

    ax.scatter(y_emb[:, 0], y_emb[:, 1], s=s, c=colormap[labels])

    ax.margins(0.05, 0.05)
    plt.autoscale(tight=True)
    plt.tight_layout()
