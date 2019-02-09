import torch
import numpy as np

def get_net_projection(all_data, net):
    all_y_pred = []
    net.eval()

    for G in all_data:
        # if torch.cuda.is_available():
        #     y_pred = net.forward(G).cpu().detach().numpy()
        # else:
        #     y_pred = net.forward(G).detach().numpy()
        y_pred = net.forward(G).detach().numpy()
        all_y_pred.append(y_pred)

    return np.concatenate(all_y_pred, axis=0)


def get_net_embeddings(all_data, net, net_type, H=50):
    # Use the model object to select the desired layer
    if net_type =='graph':
        layer = net._modules['gnn_cells'][0]
    elif net_type == 'simple':
        layer = net._modules['relu']

    # Get the total number of data points
    n = sum([len(G.labels) for G in all_data])

    # Define a function that will copy the output of a layer
    my_embedding = torch.zeros([n, H])

    def copy_data(m, i, o):
        my_embedding.copy_(o.data)

    # Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)

    # Perform projection to capture embeddings
    get_net_projection(all_data, net)

    return my_embedding
