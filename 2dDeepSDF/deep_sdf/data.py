#!/usr/bin/env python3

import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data
import math
# import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib

import deep_sdf.workspace as ws


def get_instance_filenames(data_source):
    return glob.glob(os.path.join(data_source, "*.npy"))


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]


def read_sdf_samples_into_ram(filename):
    npy = np.load(filename)
    tensor = torch.from_numpy(npy)

    return tensor


# for evaluation time
def create_single_sample_position(label_path, nb_pixel=None):
    label = np.load(label_path)
    if nb_pixel is None:
        nb_pixel = label.shape[-1]
    max_d = math.sqrt(2*nb_pixel**2)
    label = label.reshape((-1, 1)) / max_d
    

    coords = np.array([[i, j] for i in range(nb_pixel)
                       for j in range(nb_pixel)])
    coords = (coords - (nb_pixel-1)//2) / ((nb_pixel)//2)

    data = np.concatenate((coords, label), axis=-1)
    data = torch.from_numpy(data)
    return data


# for training time
def create_sample_position(labels_path, nb_pixel):
    labels = np.load(labels_path)

    labels = labels.reshape((labels.shape[0], -1))
    max_d = math.sqrt(2*nb_pixel**2)
    labels = labels[:, :, np.newaxis] / max_d

    coords = np.array([[i, j] for i in range(nb_pixel)
                       for j in range(nb_pixel)])
    coords = (coords - (nb_pixel-1)//2) / ((nb_pixel)//2)

    coords_all = np.broadcast_to(
        coords, (labels.shape[0], coords.shape[0], coords.shape[1]))
    dataset = np.concatenate((coords_all, labels), axis=-1)
    dataset = torch.from_numpy(dataset)
    return dataset


class SDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        labels_path,
        nb_pixel,
        print_filename=False,
        num_files=1000,
    ):
        self.dataset = create_sample_position(labels_path, nb_pixel)
        # self.data_source = data_source
        # self.npyfiles = get_instance_filenames(data_source, split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], idx
