import os
import h5py
import numpy as np
import torch as th
from torch.utils.data import Dataset, DataLoader


def normalize(normalized_velocities, max, min):
    return 2 * (normalized_velocities - min) / (max - min) -1

def denormalize(velocities, max, min):
    return (velocities + 1) * (max - min) / 2 + min


def load_data(dataset_repertory, dataset_name, normalized=False):
    """
    Load the turbulence time series with shape [B x C x T]
    """
    file_path = os.path.join(dataset_repertory, dataset_name)
    with h5py.File(file_path , 'r') as h5f:
        rx0 = np.array(h5f.get('min'))
        rx1 = np.array(h5f.get('max'))
        velocities = np.array(h5f.get('train')).swapaxes(1, 2)
    if not normalized :    
        velocities = denormalize(velocities, rx1, rx0)
    return velocities


def training_test_split(dataset, training_size=0.8, validation_size=0.1, normalization=True):
    """
    Split the dataset into training, validation, and test subsets.
    """
    N = dataset.shape[0]
    n_train = int(N * training_size)
    n_val = int(N * validation_size)
    train = dataset[:n_train]
    val = dataset[n_train:n_train + n_val] if n_val > 0 else None
    test = dataset[n_train + n_val:]
    if normalization:
        train_min = th.amin(train, dim=(0, 2), keepdim=True)
        train_max = th.amax(train, dim=(0, 2), keepdim=True)
        train = normalize(train, train_max, train_min)
        if val is not None:
            val = normalize(val, train_max, train_min)
        test = normalize(test, train_max, train_min)
    return (train, val, test) if val is not None else (train, test)


class TurbDataset(Dataset):
    """
    Initialize the turbulence dataset.
    """
    def __init__(self, 
                 data):
        super().__init__()
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx]
        return x
    

def loader(dataset, batch_size, shuffle=True, infinite=True):
    """
    Create an infinite DataLoader iterator for training model.
    """
    torch_dataset = TurbDataset(dataset)
    loader = DataLoader(
        torch_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=1,
        drop_last=True
    )
    if infinite:
        while True:
            for batch in loader:
                yield batch
    else:
        for batch in loader:
            yield batch