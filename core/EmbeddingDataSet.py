import os
import pickle
import numpy as np
from core.GraphDataBlock import GraphDataBlock


class EmbeddingDataSet():
    train_dir = {'mnist': 'mnist_train.pkl',
                 'usps': 'usps_train_tsne.pkl',
                 '20news': '20news_train_tsne.pkl',
                 'fasttext': 'fasttext_train_tsne.pkl',
                 'imagenet': 'imagenet_train.pkl',
                 'cora': 'cora_subset.pkl'}

    test_dir = {'mnist': 'mnist_test.pkl',
                'usps': 'usps_test_tsne.pkl',
                '20news': None,
                'fasttext': None,
                'imagenet': None,
                'cora': 'cora_subset.pkl'}

    def __init__(self, name, data_dir, train=True):
        self.name = name
        self.data_dir = data_dir
        self.train_dir = EmbeddingDataSet.train_dir[name]
        self.test_dir = EmbeddingDataSet.test_dir[name]
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
        self.adj_matrix = None
        if len(file_contents) > 3:
            self.adj_matrix = file_contents[3]

        self.is_labelled = len(self.labels) != 0
        self.is_graph = self.adj_matrix is not None
        self.input_dim = np.prod(self.inputs.shape[1:])

        self.all_indices = np.arange(0, self.inputs.shape[0])

    def create_all_data(self, n_batches=1, shuffle=False):
        # Initialise all_train_data: list of DataEmbeddingGraph blocks
        i = 0
        labels_subset = []
        adj_subset = None
        self.all_data = []

        if shuffle:
            np.random.shuffle(self.all_indices)

        # Split equally
        # TODO: Another option to split randomly
        chunk_sizes = self.get_k_equal_chunks(self.inputs.shape[0], k=n_batches)

        for num_samples in chunk_sizes:
            mask = self.all_indices[i: i + num_samples]
            inputs_subset = self.inputs[mask]
            # X_emb_subset = self.X_emb[mask]
            if self.is_labelled:
                labels_subset = self.labels[mask]
            if self.is_graph:
                adj_subset = self.adj_matrix[mask, :][:, mask]

            # Package data into graph block
            G = GraphDataBlock(inputs_subset, labels=labels_subset, W=adj_subset)
            # G.target = X_emb_subset  # replace target with pre-computed embeddings

            self.all_data.append(G)
            i += num_samples

    def summarise(self):
        print("Name of dataset = {}".format(self.name))
        print("Input dimension = {}".format(self.input_dim))
        print("Number of training samples = {}".format(self.inputs.shape[0]))
        print("Training labels = {}".format(self.is_labelled))
        print("Graph information = {}".format(self.is_graph))

    def get_k_equal_chunks(self, n, k):
        # returns n % k sub-arrays of size n//k + 1 and the rest of size n//k
        p, r = divmod(n, k)
        return [p + 1 for _ in range(r)] + [p for _ in range(k - r)]

    def get_current_inputs(self):
        inputs = self.inputs[self.all_indices]
        labels = self.labels[self.all_indices]
        adj = self.adj_matrix[self.all_indices, :][:, self.all_indices]
        return inputs, labels, adj


if __name__ == "__main__":
    name = 'cora'
    data_dir = '/Users/signapoop/desktop/data'
    dataset = EmbeddingDataSet(name, data_dir)
    dataset.create_all_data()
    dataset.summarise()
