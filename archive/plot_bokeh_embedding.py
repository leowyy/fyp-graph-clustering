import numpy as np
import bokeh.plotting as bp
from bokeh.plotting import show


def plot_embedding(X_emb, labels, named_labels):
    # 20 colors
    colormap = np.array([
        "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
        "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
        "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
        "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"
    ])

    plot_fig = bp.figure(plot_width=700, plot_height=500,
                         tools="pan, wheel_zoom, box_zoom, reset, hover, previewsave",
                         x_axis_type=None, y_axis_type=None, min_border=1)

    data_dict = {'x': X_emb[:, 0],
                 'y': X_emb[:, 1],
                 'color': colormap[labels],
                 'label': named_labels}

    mySource = bp.ColumnDataSource(data_dict)

    plot_fig.circle(x='x', y='y', color='color', legend='label', source=mySource)
    plot_fig.legend.location = (0, 70)
    new_legend = plot_fig.legend[0]
    plot_fig.legend[0].plot = None
    plot_fig.add_layout(new_legend, 'right')
    plot_fig.legend.label_text_font_size = '7pt'

    show(plot_fig)
