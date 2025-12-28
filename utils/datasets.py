import os
import h5py
import numpy as np


def load_data(dataset_repertory, dataset_name, normalized=False):
    file_path = os.path.join(dataset_repertory, dataset_name)
    with h5py.File(file_path , 'r') as h5f:
        rx0 = np.array(h5f.get('min'))
        rx1 = np.array(h5f.get('max'))
        velocities = np.array(h5f.get('train')).swapaxes(1, 2)
    if not normalized :    
        velocities = ((velocities+1)*(rx1-rx0)/2 + rx0)
    return velocities