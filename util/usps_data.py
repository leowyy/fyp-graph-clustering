import h5py
import numpy as np
import torch
from torchvision import transforms


# The dataset has 7291 train and 2007 test images. The images are 16x16 grayscale pixels.

def draw_random_usps_samples(data_dir, n_samples=None, train=True):
    with h5py.File(data_dir, 'r') as hf:

        if train:
            train = hf.get('train')
            X = train.get('data')[:]
            labels = train.get('target')[:]

        else:
            test = hf.get('test')
            X = test.get('data')[:]
            labels = test.get('target')[:]
    
    if n_samples is not None:
        random_idx = np.random.choice(list(range(X.shape[0])), n_samples, replace=False)
        X = X[random_idx, :]
        labels = labels[random_idx]
    else:
        n_samples = len(labels)

    X = X.reshape(-1, 1, 16, 16).transpose((0, 2, 3, 1))  # convert to HWC

    train_transform = transforms.Compose([
        transforms.ToPILImage(mode='F'),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.25448,), (0.3846,)),
        # transforms.Normalize((0.1307,), (0.3081,)),
    ])

    # Resize input images to 28x28
    inputs = torch.zeros((n_samples, 1, 28, 28))
    for i in range(X.shape[0]):
        inputs[i] = train_transform(X[i, ::])

    return inputs, torch.from_numpy(labels)
