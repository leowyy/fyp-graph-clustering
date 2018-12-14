import os
import time
import torch
from torch.autograd import Variable
from core.tsne_torch_loss import compute_joint_probabilities


if torch.cuda.is_available():
    print('cuda available')
    dtypeFloat = torch.cuda.FloatTensor
    dtypeLong = torch.cuda.LongTensor
else:
    print('cuda not available')
    dtypeFloat = torch.FloatTensor
    dtypeLong = torch.LongTensor


def save_checkpoint(state, filename):
    torch.save(state, filename)


def train(net, embedding_dataset, opt_parameters, loss_function, checkpoint_dir):
    # Optimization parameters
    split_batches = opt_parameters['split_batches']
    metric = opt_parameters['distance_metric']
    lr = opt_parameters['learning_rate']
    max_iters = opt_parameters['max_iters']
    batch_iters = opt_parameters['batch_iters']
    decay_rate = opt_parameters['decay_rate']
    start_epoch = opt_parameters['start_epoch']
    checkpoint_interval = max_iters / 5

    # Optimizer
    optimizer = net.update(lr)

    # Statistics
    t_start = time.time()
    t_start_total = time.time()
    average_loss_old = 1e10
    running_tsne_loss = 0.0
    running_graph_loss = 0.0
    running_loss = 0.0
    running_total = 0
    tab_results = []

    all_P_initialised = False

    # Hyperparameters
    alpha = 1  # Weight of graph edges to calculation of P
    beta = 0.1  # Weight of graph cut loss

    for iteration in range(start_epoch+1, start_epoch+max_iters+1):
        # Set the net to training mode
        net.train()

        # Create a new set of data blocks
        if loss_function in ['tsne_loss', 'tsne_graph_loss']:
            if split_batches or not all_P_initialised:
                embedding_dataset.create_all_train_data(split_batches=split_batches, shuffle=True)
                all_P = []
                for G in embedding_dataset.all_train_data:
                    X = G.data.view(G.data.shape[0], -1).numpy()
                    P = compute_joint_probabilities(X, perplexity=30, metric=metric, adj=G.adj_matrix, alpha=alpha, verbose=0)
                    P = P.reshape((X.shape[0], X.shape[0]))
                    P = torch.from_numpy(P).type(dtypeFloat)
                    all_P.append(P)
                all_P_initialised = True

        # Forward pass through all training data
        for i, G in enumerate(embedding_dataset.all_train_data):
            # Forward pass
            y_pred = net.forward(G)

            # Target embedding matrix
            y_true = G.target
            y_true = Variable(torch.FloatTensor(y_true).type(dtypeFloat), requires_grad=False)

            # Compute overall loss
            if loss_function == 'pairwise_loss':
                loss = net.pairwise_loss(y_pred, y_true, G.adj_matrix)
            elif loss_function == 'composite_loss':
                loss1 = net.loss(y_pred, y_true)
                loss2 = net.pairwise_loss(y_pred, y_true, G.adj_matrix)
                loss = 0.5 * loss1 + 0.5 * loss2
            elif loss_function == 'tsne_loss':
                loss = net.tsne_loss(all_P[i], y_pred, metric=metric)
            elif loss_function =='tsne_graph_loss':
                loss1 = net.tsne_loss(all_P[i], y_pred, metric=metric)
                loss2 = net.graph_cut_loss(G.adj_matrix, y_pred)
                loss = (1-beta) * loss1 + beta * loss2
                running_tsne_loss += loss1.item()
                running_graph_loss += loss2.item()

            loss_train = loss.item()
            running_loss += loss_train
            running_total += 1

            # Backpropagate
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # learning rate, print results
        if not iteration % batch_iters:

            # time
            t_stop = time.time() - t_start
            t_start = time.time()

            # update learning rate
            average_loss = running_loss / running_total
            if average_loss > 0.99 * average_loss_old:
                lr /= decay_rate
            average_loss_old = average_loss
            optimizer = net.update_learning_rate(optimizer, lr)

            # print results
            if loss_function == 'tsne_graph_loss':
                average_tsne_loss = running_tsne_loss / running_total
                average_graph_loss = running_graph_loss / running_total
                running_tsne_loss = 0.0
                running_graph_loss = 0.0
                print('iteration= %d, loss(%diter)= %.8f, tsne_loss= %.8f, graph_loss= %.8f, '
                      'lr= %.8f, time(%diter)= %.2f' %
                      (iteration, batch_iters, average_loss, average_tsne_loss, average_graph_loss, lr, batch_iters, t_stop))
                tab_results.append([iteration, average_loss, time.time() - t_start_total])
            else:
                print('iteration= %d, loss(%diter)= %.8f, lr= %.8f, time(%diter)= %.2f' %
                      (iteration, batch_iters, average_loss, lr, batch_iters, t_stop))
                tab_results.append([iteration, average_loss, time.time() - t_start_total])

            running_loss = 0.0
            running_total = 0

        if iteration % checkpoint_interval == 0:
            print('Saving checkpoint at iteration = {}\n'.format(iteration))
            filename = os.path.join(checkpoint_dir, net.name + '_' + str(iteration) + '.pkl')
            save_checkpoint({
                'state_dict': net.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, filename)

    return tab_results