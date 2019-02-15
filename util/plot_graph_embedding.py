import numpy as np
import scipy.sparse as sp
import bokeh.plotting as bp
from bokeh.plotting import show
from bokeh.models.glyphs import Segment
from bokeh.palettes import Category20_20, Category20b_20, Accent8


def plot_graph_embedding(y_emb, labels, adj, line_alpha=0.2):
    labels = np.array([int(l) for l in labels])
    adj = sp.coo_matrix(adj)

    # colormap = np.array([
    #     "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
    #     "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
    #     "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
    #     "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"
    # ])
    colormap = np.array(Category20_20 + Category20b_20 + Accent8)

    plot_fig = bp.figure(plot_width=900, plot_height=650,
                         tools="pan, wheel_zoom, box_zoom, reset, previewsave",
                         x_axis_type=None, y_axis_type=None, min_border=1)

    data_dict = {'x': y_emb[:, 0],
                 'y': y_emb[:, 1],
                 'color': colormap[labels]}

    mySource = bp.ColumnDataSource(data_dict)

    classA = labels[adj.row]
    classB = labels[adj.col]
    mask = classA == classB
    edge_colormask = mask * (classA + 1) - 1

    # edge_colormask = edge_colormask[mask]
    # p0 = X_emb[adj.row[mask],:]
    # p1 = X_emb[adj.col[mask],:]

    p0 = y_emb[adj.row, :]
    p1 = y_emb[adj.col, :]

    source = bp.ColumnDataSource(dict(
            x0=p0[:,0],
            y0=p0[:,1],
            x1=p1[:,0],
            y1=p1[:,1],
            color=colormap[edge_colormask]
        )
    )

    glyph = Segment(x0="x0", y0="y0", x1="x1", y1="y1", line_color="color", line_width=1, line_alpha=line_alpha)
    plot_fig.add_glyph(source, glyph)

    plot_fig.circle(x='x', y='y', color='color', source=mySource)

    show(plot_fig)
