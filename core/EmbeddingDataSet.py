import os
import pickle
import numpy as np
import torch
from core.DataEmbeddingGraph import DataEmbeddingGraph


class EmbeddingDataSet():
    train_dir = {'mnist': 'mnist_train.pkl',
                 'usps': 'usps_train_tsne.pkl',
                 '20news': '20news_train_tsne.pkl',
                 'yux': 'yux_train_tsne_shuffle.pkl',
                 'fasttext': 'fasttext_train_tsne.pkl',
                 'mnist_embeddings': 'mnist_embeddings_train.pkl',
                 'imagenet': 'imagenet_train.pkl',
                 'cora': 'cora_subset.pkl'}

    test_dir = {'mnist': 'mnist_test.pkl',
                'usps': 'usps_test_tsne.pkl',
                '20news': None,
                'yux': None,
                'fasttext': None,
                'mnist_embeddings': None,
                'imagenet': None,
                'cora': 'cora_subset.pkl'}

    def __init__(self, name, data_dir, train=True):
        self.name = name
        self.data_dir = data_dir
        self.train_dir = EmbeddingDataSet.train_dir[name]
        self.test_dir = EmbeddingDataSet.test_dir[name]
        self.max_train_size = None
        self.input_dim = None
        self.is_labelled = False

        self.all_data = []

        # Extract data from file contents
        data_root = os.path.join(self.data_dir, self.name)
        if train:
            fname = os.path.join(data_root, self.train_dir)
        else:
            assert self.test_dir is not None
            fname = os.path.join(data_root, self.test_dir)
        with open(fname, 'rb') as f:
            file_contents = pickle.load(f)

        self.inputs = file_contents[0]
        self.labels = file_contents[1]
        self.X_emb = file_contents[2]
        if len(file_contents) > 3:
            self.adj_matrix = file_contents[3]
        else:
            self.adj_matrix = None

        self.is_labelled = len(self.labels) != 0
        self.is_graph = self.adj_matrix is not None
        self.input_dim = np.prod(self.inputs.shape[1:])

        if type(self.labels) is np.ndarray:
            self.labels = torch.from_numpy(self.labels).type(torch.FloatTensor)

    def create_all_data(self, max_train_size=None, split_batches=True, shuffle=False):
        # If not split batches, train on full graph
        if not split_batches:
            G_all = DataEmbeddingGraph(self.inputs, self.labels, method=None, W=self.adj_matrix)
            # G_all.target = self.X_emb
            self.all_data = [G_all]
            return

        if max_train_size is None:
            max_train_size = self.inputs.shape[0]
        self.max_train_size = max_train_size

        # Initialise all_train_data: list of DataEmbeddingGraph blocks
        i = 0
        labels_subset = []
        adj_subset = None
        self.all_data = []

        all_indices = np.arange(0, self.inputs.shape[0])
        if shuffle:
            np.random.shuffle(all_indices)

        while i < self.max_train_size:
            # Draw a random training batch of variable size
            num_samples = np.random.randint(300, 600)
            mask = all_indices[i: min(i + num_samples, self.max_train_size)]
            inputs_subset = self.inputs[mask]
            # X_emb_subset = self.X_emb[mask]
            if self.is_labelled:
                labels_subset = self.labels[mask]
            if self.is_graph:
                adj_subset = self.adj_matrix[mask, :][:, mask]

            # Package data into graph block
            G = DataEmbeddingGraph(inputs_subset, labels_subset, method=None, W=adj_subset)
            # G.target = X_emb_subset  # replace target with pre-computed embeddings

            self.all_data.append(G)
            i += num_samples

        if self.all_data[-1].data.shape[0] < 300:
            self.all_data = self.all_data[:-1]

    def summarise(self):
        print("Name of dataset = {}".format(self.name))
        print("Input dimension = {}".format(self.input_dim))
        print("Number of training samples = {}".format(self.max_train_size))
        print("Training labels = {}".format(self.is_labelled))
        print("Graph information = {}".format(self.is_graph))


if __name__ == "__main__":
    name = 'cora'
    data_dir = '/Users/signapoop/desktop/data'
    dataset = EmbeddingDataSet(name, data_dir)
    dataset.create_all_data()
    dataset.summarise()
