from torchvision import transforms
from util.usps import USPS


def get_train_set(data_dir, augment):
    # define transform
    if augment:
        train_transform = transforms.Compose([
            transforms.ToPILImage(mode='F'),
            transforms.Resize((32, 32)),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            #transforms.Normalize((0.1307,), (0.3081,)),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToPILImage(mode='F'),
            transforms.ToTensor(),
            #transforms.Normalize((0.1307,), (0.3081,)),
        ])

    train_data = USPS(
        root=data_dir, train=True,
        download=True, transform=train_transform,
    )
    return train_data


def get_test_set(data_dir, augment):
    """
    Params
    ------
    - data_dir: path directory to the dataset
    Returns
    -------
    - test_data
    """

    # define transform
    if augment:
        transform = transforms.Compose([
            transforms.ToPILImage(mode='F'),
            transforms.Resize((32, 32)),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            #transforms.Normalize((0.1307,), (0.3081,)),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToPILImage(mode='F'),
            transforms.ToTensor(),
            #transforms.Normalize((0.1307,), (0.3081,)),
        ])

    test_data = USPS(
        root=data_dir, train=False,
        download=True, transform=transform,
    )
    return test_data
