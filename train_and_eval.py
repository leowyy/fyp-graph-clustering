import os
import pickle
import argparse
import pathlib
import torch

from learn_embedding import train
from core.EmbeddingDataSet import EmbeddingDataSet
from core.GraphConvNet2 import GraphConvNet2
from core.OldGraphConvNet2 import OldGraphConvNet2
from core.SimpleNet import SimpleNet


def save_metadata(checkpoint_dir, task_parameters, net_parameters, opt_parameters, epoch_num):
    metadata_filename = os.path.join(checkpoint_dir, "experiment_metadata_{}.txt".format(epoch_num))
    with open(metadata_filename, 'w') as f:
        f.write('-----------------------\n')
        f.write('\n'.join(["%s = %s" % (k, v) for k, v in task_parameters.items()]))

        f.write('\n-----------------------\n')
        f.write('\n'.join(["%s = %s" % (k, v) for k, v in net_parameters.items()]))

        f.write('\n-----------------------\n')
        f.write('\n'.join(["%s = %s" % (k, v) for k, v in opt_parameters.items()]))


def save_train_log(checkpoint_dir, tab_results, epoch_num):
    logs_filename = os.path.join(checkpoint_dir, "experiment_results_{}.pkl".format(epoch_num))
    with open(logs_filename, 'wb') as f:
        pickle.dump(tab_results, f)


def get_oldest_net(output_dir):
    filenames = os.listdir(output_dir)
    max_iteration = 0
    target_file = ''
    for fname in filenames:
        if '_net_' in fname:
            iteration_num = int(fname.split('_')[-1][:-4])
            if iteration_num > max_iteration:
                max_iteration = iteration_num
                target_file = fname
    return os.path.join(output_dir, target_file), max_iteration


def main(input_dir, output_dir, dataset_name, net_type, resume_folder):
    # optimization parameters
    opt_parameters = {}
    opt_parameters['learning_rate'] = 0.00075  # ADAM
    opt_parameters['max_iters'] = 500
    opt_parameters['batch_iters'] = 50
    opt_parameters['save_flag'] = True
    opt_parameters['decay_rate'] = 1.25
    opt_parameters['start_epoch'] = 0
    opt_parameters['distance_metric'] = 'cosine'
    opt_parameters['split_batches'] = False  # Set to true if training on subgraphs

    dataset = EmbeddingDataSet(dataset_name, input_dir)
    dataset.create_all_train_data(split_batches=opt_parameters['split_batches'], shuffle=True)
    dataset.summarise()

    task_parameters = {}
    task_parameters['net_type'] = net_type
    task_parameters['loss_function'] = 'tsne_loss'
    task_parameters['n_components'] = 2
    task_parameters['val_flag'] = False

    net_parameters = {}
    net_parameters['n_components'] = task_parameters['n_components']
    net_parameters['D'] = dataset.input_dim  # input dimension
    net_parameters['H'] = 50  # number of hidden units
    net_parameters['L'] = 10  # number of hidden layers

    # Initialise network
    if net_type == 'graph':
        net = GraphConvNet2(net_parameters)
    elif net_type == 'old_graph':
        net = OldGraphConvNet2(net_parameters)
    elif net_type == 'simple':
        net = SimpleNet(net_parameters)
        opt_parameters['max_iters'] = 1000
        opt_parameters['batch_iters'] = 50

    device = 'cpu'
    if torch.cuda.is_available():
        net.cuda()
        device = 'cuda'

    if resume_folder != 'NA':
        checkpoint_dir = os.path.join(output_dir, resume_folder)
        net_filename, start_epoch = get_oldest_net(checkpoint_dir)
        checkpoint = torch.load(net_filename, map_location=device)
        print("Resuming training from: {}".format(net_filename))
        net.load_state_dict(checkpoint['state_dict'])
        opt_parameters['start_epoch'] = start_epoch
    else:
        # Create checkpoint dir
        subdirs = [x[0] for x in os.walk(output_dir) if dataset_name in x[0]]
        run_number = str(len(subdirs) + 1)
        checkpoint_dir = os.path.join(output_dir, dataset_name + '_' + run_number)
        pathlib.Path(checkpoint_dir).mkdir(exist_ok=True)  # create the directory if it doesn't exist

    print("Number of network parameters = {}".format(net.nb_param))
    print('Saving results into: {}'.format(checkpoint_dir))

    if 2 == 1:  # fast debugging
        opt_parameters['max_iters'] = 10
        opt_parameters['batch_iters'] = 1

    tab_results = train(net, dataset, opt_parameters, task_parameters['loss_function'], checkpoint_dir)

    end_epoch = opt_parameters['start_epoch'] + opt_parameters['max_iters']

    if opt_parameters['save_flag']:
        save_metadata(checkpoint_dir, task_parameters, net_parameters, opt_parameters, end_epoch)
        save_train_log(checkpoint_dir, tab_results, end_epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Embedding net training')
    parser.add_argument('-i', '--input_dir', type=str, help='input dir')
    parser.add_argument('-o', '--output_dir', type=str, help='output dir')
    parser.add_argument('-d', '--dataset_name', type=str, help='name of dataset')
    parser.add_argument('-net', '--net_type', type=str, help='type of network')
    parser.add_argument('-r', '--resume_folder', type=str, default='NA', help='folder to resume from')
    args = parser.parse_args()

    print("Input directory: {}".format(args.input_dir))
    print("Output directory: {}".format(args.output_dir))
    print("Dataset name: {}".format(args.dataset_name))
    print("Network type: {}".format(args.net_type))
    print("Resume from folder: {}".format(args.resume_folder))

    main(args.input_dir, args.output_dir, args.dataset_name, args.net_type, args.resume_folder)
