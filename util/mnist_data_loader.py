from torchvision import transforms
import torchvision.datasets as datasets


def get_train_set(data_dir):
    # define transform
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_data = datasets.MNIST(
        root=data_dir, train=True,
        download=True, transform=train_transform,
    )
    return train_data


def get_test_set(data_dir):
    """
    Params
    ------
    - data_dir: path directory to the dataset
    Returns
    -------
    - test_data
    """

    # define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    test_data = datasets.MNIST(
        root=data_dir, train=False,
        download=True, transform=transform,
    )
    return test_data
