import os
import os.path as osp
import pickle
import subprocess
from typing import Any

import h5py
import numpy as np
import torch

num_channels, num_frames, height, width = None, None, None, None


def create_dir(dir_name: str):
    """Create a directory if it does not exist yet."""
    if not osp.exists(dir_name):
        os.makedirs(dir_name)


def move_files(source_path: str, destpath: str):
    """Move files from `source_path` to `dest_path`."""
    subprocess.call(["mv", source_path, destpath])


def load_pickle(pickle_path: str) -> Any:
    """Load a pickle file."""
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    return data


def load_hdf5(hdf5_path: str) -> Any:
    with h5py.File(hdf5_path, "r") as h5file:
        data = {key: np.array(value) for key, value in h5file.items()}
    return data


def save_hdf5(data: Any, hdf5_path: str):
    with h5py.File(hdf5_path, "w") as h5file:
        for key, value in data.items():
            h5file.create_dataset(key, data=value)


def save_pickle(data: Any, pickle_path: str):
    """Save data in a pickle file."""
    with open(pickle_path, "wb") as f:
        pickle.dump(data, f, protocol=4)


def load_txt(txt_path: str):
    """Load a txt file."""
    with open(txt_path, "r") as f:
        data = f.read()
    return data


def save_txt(data: str, txt_path: str):
    """Save data in a txt file."""
    with open(txt_path, "w") as f:
        f.write(data)


def load_pth(pth_path: str) -> Any:
    """Load a pth (PyTorch) file."""
    data = torch.load(pth_path)
    return data


def save_pth(data: Any, pth_path: str):
    """Save data in a pth (PyTorch) file."""
    torch.save(data, pth_path)
