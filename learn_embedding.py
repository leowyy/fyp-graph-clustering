import os
import time
import torch
from core.tsne_torch_loss import compute_joint_probabilities, tsne_torch_loss
from core.graph_cut_torch_loss import covariance_constraint, graph_cut_torch_loss
from util.evaluation_metrics import evaluate_net_metrics


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


def train(net, train_set, opt_parameters, checkpoint_dir, val_set=None):
    # Optimization parameters
    n_batches = opt_parameters['n_batches']
    shuffle_flag = opt_parameters['shuffle_flag']
    metric = opt_parameters['distance_metric']
    distance_reduction = opt_parameters['distance_reduction']
    graph_weight = opt_parameters['graph_weight']
    loss_function = opt_parameters['loss_function']

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

    all_features_P_initialised = False

    for iteration in range(start_epoch+1, start_epoch+max_iters+1):
        net.train()  # Set the net to training mode

        # Create a new set of data blocks
        if loss_function in ['tsne_loss', 'tsne_graph_loss']:
            if n_batches > 1 or not all_features_P_initialised:
                train_set.create_all_data(n_batches=n_batches, shuffle=shuffle_flag)
                all_features_P = []
                all_graph_P = []
                for G in train_set.all_data:
                    X = G.data.view(G.data.shape[0], -1).numpy()
                    P = compute_joint_probabilities(X, perplexity=30, metric=metric, adj=G.adj_matrix, alpha=distance_reduction)
                    all_features_P.append(P)

                    if loss_function =='tsne_graph_loss':
                        P = compute_joint_probabilities(X, perplexity=30, metric='shortest_path', adj=G.adj_matrix)
                        all_graph_P.append(P)
                all_features_P_initialised = True

        # Forward pass through all training data
        for i, G in enumerate(train_set.all_data):
            y_pred = net.forward(G)

            if loss_function == 'tsne_loss':
                loss = tsne_torch_loss(all_features_P[i], y_pred)
            elif loss_function =='tsne_graph_loss':
                feature_loss = tsne_torch_loss(all_features_P[i], y_pred)
                graph_loss = tsne_torch_loss(all_graph_P[i], y_pred)

                loss = (1-graph_weight) * feature_loss + graph_weight * graph_loss
                running_tsne_loss += feature_loss.item()
                running_graph_loss += graph_loss.item()

            running_loss += loss.item()
            running_total += 1

            # Backprop
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # update learning rate, print results, perform validation
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
                running_covariance_loss = 0.0
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

            if val_set is not None:
                validate(net, val_set)

        # save checkpoint
        if iteration % checkpoint_interval == 0:
            print('Saving checkpoint at iteration = {}\n'.format(iteration))
            filename = os.path.join(checkpoint_dir, net.name + '_' + str(iteration) + '.pkl')
            save_checkpoint({
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename)

    return tab_results


def validate(net, val_set):
    net.eval()
    trust_score, one_nn_score, five_nn_score, time_elapsed = evaluate_net_metrics(val_set.all_data, net)
    print("Trustworthy score = {:.4f}".format(trust_score))
    print("1-NN generalisation accuracy = {:.4f}".format(one_nn_score))
    print("5-NN generalisation accuracy = {:.4f}".format(five_nn_score))
    print("Average time to compute (s) = {:.4f}\n".format(time_elapsed))
