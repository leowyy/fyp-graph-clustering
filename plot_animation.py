import os
import numpy as np
from core.EmbeddingDataSet import EmbeddingDataSet
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import scipy.sparse as sp
from bokeh.palettes import Category20_20, Category20b_20, Accent8
from matplotlib import collections  as mc
plt.rcParams['animation.ffmpeg_path'] = '/Users/signapoop/anaconda3/envs/py36/bin/ffmpeg'

colormap = np.array(Category20_20 + Category20b_20 + Accent8)

def plot_graph_embedding(y_emb, labels, adj, ax, edge_colormask, line_alpha=0.2, s=4, title=""):

    ax.set_title(title)

    # Plot edges
    p0 = y_emb[adj.row, :]
    p1 = y_emb[adj.col, :]
    p_0 = [tuple(row) for row in p0]
    p_1 = [tuple(row) for row in p1]
    lines = list(zip(p_0, p_1))
    lc = mc.LineCollection(lines, linewidths=0.5, colors=colormap[edge_colormask])
    lc.set_alpha(line_alpha)
    ax.add_collection(lc)

    ax.scatter(y_emb[:, 0], y_emb[:, 1], s=s, c=colormap[labels])

    # plt.tight_layout()


def main():
    dataset_name = 'cora_test'
    input_dir = '/Users/signapoop/Desktop/data'
    root = 'results/cora_third_test_13'

    dataset = EmbeddingDataSet(dataset_name, input_dir, train=True)
    dataset.create_all_data(n_batches=1, shuffle=False)

    f, ax = plt.subplots(1, sharex='col', figsize=(5, 4), dpi=400)
    ax.set_axis_off()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    labels = np.array([int(l) for l in dataset.labels])
    adj = sp.coo_matrix(dataset.adj_matrix)
    classA = labels[adj.row]
    classB = labels[adj.col]
    mask = classA == classB
    edge_colormask = mask * (classA + 1) - 1

    max_i = 250
    all_y_emb = []
    for i in range(max_i):
        fname = os.path.join(root, 'proj_' + str(i+1) + '.npy')
        all_y_emb.append(np.load(fname))

    convergence_i = list(range(170, 170, 10))
    convergence_i = list(np.repeat(convergence_i, 5))
    list_of_i = list(range(1, max_i+1)) + convergence_i

    for i in convergence_i:
        fname = os.path.join(root, 'proj_' + str(i) + '.npy')
        all_y_emb.append(np.load(fname))

    print(len(list_of_i))
    print(len(all_y_emb))
    max_i = len(list_of_i)

    def animate(i):
        print(i)
        if i > 0:
            for artist in ax.lines + ax.collections:
                artist.remove()
        title = "iteration = {}".format(list_of_i[i])
        plot_graph_embedding(all_y_emb[i], labels, adj, ax=ax, line_alpha=0.1, title=title, edge_colormask=edge_colormask)

    Writer = animation.writers['ffmpeg']
    #writer = Writer(fps=8, metadata=dict(artist='Me'), bitrate=1800, codec="libx264")
    writer = Writer(fps=8, metadata=dict(artist='Me'), bitrate=-1)
    ani = animation.FuncAnimation(f, animate, frames=max_i, repeat=False)
    ani.save('test.mp4', writer=writer)

if __name__ == "__main__":
    main()
