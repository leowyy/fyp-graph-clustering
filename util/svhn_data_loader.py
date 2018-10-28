import numpy as np

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


def get_train_val_set(data_dir,
                      valid_size=0.2,
                      shuffle=True):
    """
    Params
    ------
    - data_dir: path directory to the dataset.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.

    Returns
    -------
    - train_data: training set
    - val_data: validation set
    - train_sampler
    - val_sampler
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    normalize = transforms.Normalize(
        mean=[0.4377, 0.4438, 0.4728],
        std=[0.1201, 0.1231, 0.1052],
    )

    # define transforms
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # load the dataset
    train_data = datasets.SVHN(
        root=data_dir, split='train',
        download=True, transform=train_transform,
    )

    val_data = datasets.SVHN(
        root=data_dir, split='train',
        download=True, transform=valid_transform,
    )

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.shuffle(indices)

    train_idx, val_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    return train_data, val_data, train_sampler, val_sampler


def get_test_set(data_dir):
    """
    Params
    ------
    - data_dir: path directory to the dataset

    Returns
    -------
    - test_data
    """
    normalize = transforms.Normalize(
        mean=[0.4377, 0.4438, 0.4728],
        std=[0.1201, 0.1231, 0.1052],
    )

    # define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    test_data = datasets.SVHN(
        root=data_dir, split='test',
        download=True, transform=transform,
    )

    return test_data
