import pickle
import os
import numpy as np
from core.DataEmbeddingGraph import DataEmbeddingGraph


class EmbeddingDataSet():
    train_dir = {'mnist': 'set_20000_mnist_tsne.pkl',
                 'usps': 'set_7291_usps_tsne.pkl',
                 '20news': '20news_train_tsne.pkl',
                 'yux': 'yux_train_tsne_shuffle.pkl',
                 'fasttext': 'fasttext_train_tsne.pkl',
                 'mnist_embeddings': 'mnist_embeddings_train.pkl',
                 'imagenet': 'imagenet_train.pkl'}

    test_dir = {'mnist': 'set_100_mnist_spectral_size_200_500.pkl',
                'usps': 'set_100_usps_spectral_size_200_500.pkl',
                '20news': None,
                'yux': None,
                'fasttext': None,
                'mnist_embeddings': None,
                'imagenet': None}

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
            [inputs, labels, X_emb] = pickle.load(f)

        if max_train_size is None:
            max_train_size = inputs.shape[0]

        self.is_labelled = len(labels) != 0

        self.max_train_size = max_train_size
        self.input_dim = np.prod(inputs.shape[1:])

        # Initialises all_train_data: list of DataEmbeddingGraph blocks

        self.all_train_data = []
        num_train_samples = 0
        labels_subset = []
        while num_train_samples <= self.max_train_size:
            # Draw a random training batch of variable size
            num_samples = np.random.randint(200, 500)
            inputs_subset = inputs[num_train_samples:num_train_samples + num_samples]
            X_emb_subset = X_emb[num_train_samples:num_train_samples + num_samples]
            if self.is_labelled:
                labels_subset = labels[num_train_samples:num_train_samples + num_samples]

            # Package into graph block
            G = DataEmbeddingGraph(inputs_subset, labels_subset, method=None)
            G.target = X_emb_subset  # replace target with pre-computed embeddings

            self.all_train_data.append(G)
            num_train_samples += num_samples

        if self.all_train_data[-1].data.shape[0] < 200:
            print('Removing the last block...')
            self.all_train_data = self.all_train_data[:-1]

    def summarise(self):
        print("Name of dataset = {}".format(self.name))
        print("Input dimension = {}".format(self.input_dim))
        print("Number of training samples = {}".format(self.max_train_size))
        print("Training labels = {}".format(self.is_labelled))


if __name__ == "__main__":
    name = 'fasttext'
    data_dir = '/Users/signapoop/desktop/data'
    dataset = EmbeddingDataSet(name, data_dir)
    dataset.prepare_train_data()
    dataset.summarise()
