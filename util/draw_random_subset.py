import torch
from torch.utils.data import Subset
import numpy as np


def draw_random_subset(dataset, num_samples):
    assert num_samples <= len(dataset)
    random_idx = np.random.choice(list(range(len(dataset))), num_samples, replace=False)
    subset = Subset(dataset, random_idx)
    new_size = [num_samples] + list(subset[0][0].shape)

    inputs = torch.zeros(new_size)
    labels = torch.zeros(num_samples, dtype=torch.int)

    for idx, (x, y) in enumerate(subset):
        inputs[idx] = x
        labels[idx] = y

    return inputs, labels
