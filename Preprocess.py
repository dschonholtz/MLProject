"""
This is a file to support fetching the data for a given patient, and shuffling it
It maintains the index for each sample so data can be shuffled randomly but then back to back samples can be re-aligned
later for time voting.
"""
import os

import torch
import h5py
from torch.utils.data import Dataset
from sklearn.preprocessing import normalize
import numpy as np


# Define a function to get the label from the filename
def get_label(filename):
    if filename.split(os.sep)[-1].startswith("pre"):
        return 0, 1
    else:
        return 1, 0


# Define a custom dataset class that inherits from torch.utils.data.Dataset
class H5Dataset(Dataset):
    def __init__(self, h5_dir):
        # Get the list of h5 files in the directory
        self.h5_files = []
        self.pre_list = []
        self.non_list = []
        self.x = []
        self.y = []
        self.file_channels = []
        sample_num = 0
        for f in os.listdir(h5_dir):
            if f.endswith(".h5"):
                sample_num += 1
                if f.startswith('pre'):
                    self.pre_list.append(os.path.join(h5_dir, f))
                else:
                    self.non_list.append(os.path.join(h5_dir, f))
        self.pre_list.sort(key=self.filename_sort_key)
        self.non_list.sort(key=self.filename_sort_key)
        self.non_idx = len(self.pre_list)
        self.h5_files = self.pre_list + self.non_list
        print('Got all Filenames. Loading data...')
        h5_idx = 0
        for f in self.h5_files:
            normalized = self.read_and_normalize_h5(f)
            # separate out each channel into its own 1d array and then add them to h5_data
            all_channels = self.break_up_channels(normalized)
            self.x += all_channels
            label = get_label(f)
            self.y += [label] * len(all_channels)
            self.file_channels.append(len(all_channels))
            h5_idx += 1

        print('All data is loaded. About to begin training.')

    def __len__(self):
        # Return the number of h5 files
        return len(self.x)

    def __getitem__(self, index):
        # Get the h5 file and data at the given index
        # Convert the data and label to torch tensors
        data = torch.from_numpy(self.x[index][...])
        label = torch.tensor(self.y[index])
        # Return a tuple of data and label
        return data, label

    @staticmethod
    def read_and_normalize_h5(h5_file):
        with h5py.File(h5_file, 'r') as f:
            x = f['data'][()]
            x = np.transpose(x)
            x = normalize(x, norm='l2', axis=1, copy=True, return_norm=False)
            x = np.expand_dims(x, -1)
            # print(x.shape)
        return x

    @staticmethod
    def break_up_channels(normalized_data):
        all_channels = []
        for i in range(normalized_data.shape[0]):
            all_channels.append(normalized_data[i, :, :].T)
        return all_channels

    @staticmethod
    def filename_sort_key(f_name):
        return int(f_name.split('sample')[-1].split('.')[0])

    # def get_prev_samples(self, filename, label, num_samples=3):
    #     if label:
    #         file_list = self.pre_list
    #     else:
    #         file_list = self.non_list

