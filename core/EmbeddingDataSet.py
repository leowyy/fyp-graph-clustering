import os
import pickle
import numpy as np
import scipy.sparse as sp
import time
from core.GraphDataBlock import GraphDataBlock
from util.graph_utils import neighbor_sampling


class EmbeddingDataSet():
    train_dir = {'cora': 'cora_train.pkl',
                 'cora_full': 'cora_full.pkl',
                 'pubmed': 'pubmed.pkl',
                 'pubmed_full': 'pubmed_full.pkl',
                 'citeseer_full': 'citeseer_full.pkl',
                 'reddit_full': 'reddit_full.pkl'}

    test_dir = {'cora': 'cora_test.pkl',
                'cora_full': 'cora_full.pkl',
                'pubmed': 'pubmed.pkl',
                'pubmed_full': 'pubmed_full.pkl',
                'citeseer_full': 'citeseer_full.pkl',
                'reddit_full': 'reddit_full.pkl'}

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
        self.adj_matrix = file_contents[2]

        self.is_labelled = len(self.labels) != 0
        self.input_dim = self.inputs.shape[1]

        self.all_indices = np.arange(0, self.inputs.shape[0])

        # Convert adj to csr matrix
        self.inputs = sp.csr_matrix(self.inputs)
        self.adj_matrix = sp.csr_matrix(self.adj_matrix)

    def create_all_data(self, n_batches=1, shuffle=False, sampling=False):
        # Initialise all_train_data: list of DataEmbeddingGraph blocks
        i = 0
        labels_subset = []
        self.all_data = []

        if shuffle:
            np.random.shuffle(self.all_indices)
        else:
            self.all_indices = np.arange(0, self.inputs.shape[0])

        # Split equally
        # TODO: Another option to split randomly
        chunk_sizes = self.get_k_equal_chunks(self.inputs.shape[0], k=n_batches)

        t_start = time.time()

        for num_samples in chunk_sizes:
            mask = self.all_indices[i: i + num_samples]

            # Perform sampling to obtain local neighborhood of mini-batch
            if sampling:
                D_layers = [9, 14]  # max samples per layer
                mask = neighbor_sampling(self.adj_matrix, mask, D_layers)

            inputs_subset = self.inputs[mask]
            adj_subset = self.adj_matrix[mask, :][:, mask]

            if self.is_labelled:
                labels_subset = self.labels[mask]

            # Package data into graph block
            G = GraphDataBlock(inputs_subset, labels=labels_subset, W=adj_subset)

            self.all_data.append(G)
            i += num_samples

        t_elapsed = time.time() - t_start
        print('Data blocks of length: ', [len(G.labels) for G in self.all_data])
        print("Time to create all data (s) = {:.4f}".format(t_elapsed))
        #print([G.edge_to_starting_vertex.getnnz() for G in self.all_data])

    def summarise(self):
        print("Name of dataset = {}".format(self.name))
        print("Input dimension = {}".format(self.input_dim))
        print("Number of training samples = {}".format(self.inputs.shape[0]))
        print("Training labels = {}".format(self.is_labelled))

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
