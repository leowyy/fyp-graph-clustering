import argparse
import os
import pathlib
from core.EmbeddingDataSet import EmbeddingDataSet
from core.GraphConvNet2 import GraphConvNet2
import torch
from learn_embedding import train


def main(input_dir, output_dir, dataset_name):
    dataset = EmbeddingDataSet(dataset_name, input_dir)
    dataset.prepare_train_data()
    dataset.summarise()

    task_parameters = {}
    task_parameters['net_type'] = 'graph_net'
    task_parameters['loss_function'] = 'composite_loss'
    task_parameters['n_components'] = 2
    task_parameters['val_flag'] = False

    net_parameters = {}
    net_parameters['n_components'] = task_parameters['n_components']
    net_parameters['D'] = dataset.input_dim  # input dimension
    net_parameters['H'] = 50  # number of hidden units
    net_parameters['L'] = 10  # number of hidden layers

    # optimization parameters
    opt_parameters = {}
    opt_parameters['learning_rate'] = 0.00075  # ADAM
    opt_parameters['max_iters'] = 200
    opt_parameters['batch_iters'] = 10
    opt_parameters['save_flag'] = True
    opt_parameters['decay_rate'] = 1.25

    if 1 == 1:  # fast debugging
        opt_parameters['max_iters'] = 5
        opt_parameters['batch_iters'] = 1

    subdirs = [x[0] for x in os.walk(output_dir) if dataset_name in x[0]]
    run_number = str(len(subdirs) + 1)
    checkpoint_dir = os.path.join(output_dir, dataset_name + '_'  + run_number)
    pathlib.Path(checkpoint_dir).mkdir(exist_ok=True)  # create the directory if it doesn't exist
    print('Saving results into: {}'.format(checkpoint_dir))

    net = GraphConvNet2(net_parameters)
    if torch.cuda.is_available():
        net.cuda()

    train(net, dataset.all_train_data, opt_parameters, task_parameters['loss_function'], checkpoint_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Embedding net training')
    parser.add_argument('-i', '--input_dir', type=str, help='input dir')
    parser.add_argument('-o', '--output_dir', type=str, help='output dir')
    parser.add_argument('-d', '--dataset_name', type=str, help='name of dataset')
    args = parser.parse_args()

    print("Input directory: {}".format(args.input_dir))
    print("Output directory: {}".format(args.output_dir))
    print("Dataset name: {}".format(args.dataset_name))
    main(args.input_dir, args.output_dir, args.dataset_name)
