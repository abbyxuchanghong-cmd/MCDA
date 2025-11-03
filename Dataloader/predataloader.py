# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 10:43:53 2024

@author: XCH
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import numpy as np

class Load_Dataset(Dataset):
    def __init__(self, dataset, dataset_configs):
        super().__init__()
        self.num_channels = dataset_configs.input_channels
        # Load samples
        x_data = dataset["samples"]
        print(np.array(x_data).shape)
        # Convert to torch tensor
        if isinstance(x_data, np.ndarray):
            x_data = torch.from_numpy(x_data)
        # Check samples dimensions.
        # The dimension of the data is expected to be (N, C, L)
        # where N is the number of samples, C is the number of channels, and L is the sequence length
        if len(x_data.shape) == 2:
            x_data = x_data.unsqueeze(1)
        elif len(x_data.shape) == 3 and x_data.shape[1] != self.num_channels:
            x_data = x_data.transpose(1, 2)
        # Normalize data
        if dataset_configs.normalize:
            data_mean = torch.mean(x_data, dim=(0, 2))
            data_std = torch.std(x_data, dim=(0, 2))
            self.transform = transforms.Normalize(mean=data_mean, std=data_std)
        else:
            self.transform = None
        self.x_data = x_data.float()
        self.len = x_data.shape[0]

    def __getitem__(self, index):
        x = self.x_data[index]
        if self.transform:
            x = self.transform(self.x_data[index].reshape(self.num_channels, -1, 1)).reshape(self.x_data[index].shape)
        return x

    def __len__(self):
        return self.len

def predata_generator(data_path, domain_id, dataset_configs, hparams, dtype):
    # loading dataset file from path
    dataset_file = torch.load(os.path.join(data_path, f"{dtype}_{domain_id}.pt"), weights_only=False)
    print(os.path.join(data_path, f"{dtype}_{domain_id}.pt"))
    # Loading datasets
    dataset = Load_Dataset(dataset_file, dataset_configs)
    if dtype == "test" or dtype == "predict":  # you don't need to shuffle or drop last batch while testing
        shuffle  = False
        drop_last = False
    else:
        shuffle = dataset_configs.shuffle
        drop_last = dataset_configs.drop_last
    # Dataloaders
    data_loader = torch.utils.data.DataLoader(dataset=dataset, 
                                              batch_size=hparams["batch_size"],
                                              shuffle=shuffle, 
                                              drop_last=drop_last, 
                                              num_workers=0)
    return data_loader