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


def train(net, all_train_data, opt_parameters, loss_function, checkpoint_dir):
    # Optimization parameters
    lr = opt_parameters['learning_rate']
    max_iters = opt_parameters['max_iters']
    batch_iters = opt_parameters['batch_iters']
    decay_rate = opt_parameters['decay_rate']
    checkpoint_interval = max_iters / 5

    # Optimizer
    optimizer = net.update(lr)

    # Statistics
    t_start = time.time()
    t_start_total = time.time()
    average_loss_old = 1e10
    running_loss = 0.0
    running_total = 0
    tab_results = []

    if loss_function == 'tsne_loss':
        all_P = []
        for G in all_train_data:
            X = G.data.view(G.data.shape[0], -1).numpy()
            P = compute_joint_probabilities(X, verbose=0, perplexity=30)
            P = P.reshape((X.shape[0], X.shape[0]))
            P = torch.from_numpy(P).type(dtypeFloat)
            all_P.append(P)

    for iteration in range(1, max_iters+1):
        net.train()
        for i, G in enumerate(all_train_data):
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
                loss = net.tsne_loss(all_P[i], y_pred)

            loss_train = loss.data[0]
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
            running_loss = 0.0
            running_total = 0

            # print results
            print('iteration= %d, loss(%diter)= %.8f, lr= %.8f, time(%diter)= %.2f' %
                  (iteration, batch_iters, average_loss, lr, batch_iters, t_stop))
            tab_results.append([iteration, average_loss, time.time() - t_start_total])

        if iteration % checkpoint_interval == 0:
            print('Saving checkpoint at iteration = {}\n'.format(iteration))
            filename = os.path.join(checkpoint_dir, net.name + '_' + str(iteration) + '.pkl')
            save_checkpoint({
                'state_dict': net.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, filename)

    return tab_results