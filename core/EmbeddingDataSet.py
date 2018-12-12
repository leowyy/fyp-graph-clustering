import os
import pickle
import numpy as np
from core.DataEmbeddingGraph import DataEmbeddingGraph


class EmbeddingDataSet():
    train_dir = {'mnist': 'mnist_train_tsne.pkl',
                 'usps': 'usps_train_tsne.pkl',
                 '20news': '20news_train_tsne.pkl',
                 'yux': 'yux_train_tsne_shuffle.pkl',
                 'fasttext': 'fasttext_train_tsne.pkl',
                 'mnist_embeddings': 'mnist_embeddings_train.pkl',
                 'imagenet': 'imagenet_train.pkl',
                 'cora': 'cora_train.pkl'}

    test_dir = {'mnist': 'mnist_test_tsne.pkl',
                'usps': 'usps_test_tsne.pkl',
                '20news': None,
                'yux': None,
                'fasttext': None,
                'mnist_embeddings': None,
                'imagenet': None,
                'cora': None}

    def __init__(self, name, data_dir):
        self.name = name
        self.data_dir = data_dir
        self.train_dir = EmbeddingDataSet.train_dir[name]
        self.test_dir = EmbeddingDataSet.test_dir[name]
        self.max_train_size = None
        self.input_dim = None
        self.is_labelled = False

        self.all_train_data = []

    def prepare_train_data(self, max_train_size=None):
        data_root = os.path.join(self.data_dir, self.name)
        train_file = os.path.join(data_root, self.train_dir)
        with open(train_file, 'rb') as f:
            file_contents = pickle.load(f)

        inputs = file_contents[0]
        labels = file_contents[1]
        X_emb = file_contents[2]
        if len(file_contents) > 3:
            adj_matrix = file_contents[3]
        else:
            adj_matrix = None

        if max_train_size is None:
            max_train_size = inputs.shape[0]

        self.is_labelled = len(labels) != 0
        self.is_graph = adj_matrix is not None

        self.max_train_size = max_train_size
        self.input_dim = np.prod(inputs.shape[1:])

        # Initialise all_train_data: list of DataEmbeddingGraph blocks
        i = 0
        labels_subset = []
        adj_subset = None
        self.all_train_data = []
        while i <= self.max_train_size:
            # Draw a random training batch of variable size
            num_samples = np.random.randint(200, 500)
            mask = list(range(i, min(i + num_samples, self.max_train_size)))
            inputs_subset = inputs[mask]
            X_emb_subset = X_emb[mask]
            if self.is_labelled:
                labels_subset = labels[mask]
            if self.is_graph:
                adj_subset = adj_matrix[mask, :][:, mask]

            # Package into graph block
            G = DataEmbeddingGraph(inputs_subset, labels_subset, method=None, W=adj_subset)
            G.target = X_emb_subset  # replace target with pre-computed embeddings

            self.all_train_data.append(G)
            i += num_samples

        if self.all_train_data[-1].data.shape[0] < 200:
            print('Removing the last block...')
            self.all_train_data = self.all_train_data[:-1]

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
    dataset.prepare_train_data()
    dataset.summarise()
