#!/usr/bin/env python3
import json
import os
import torch
import numpy as np
import random

specifications_filename = "specs.json"
logs_filename = "Logs.pth"
model_params_dir = "ModelParameters"
real_dataset_folder = "real"
fake_dataset_folder = "fake"


def split_dataset(dataset, train_split, val_split, shuffle=True):
	if shuffle == True:
		random.shuffle(dataset)
	train_set = dataset[:int(train_split*len(dataset))]
	val_set = dataset[int(train_split*len(dataset)):int((train_split+val_split)*len(dataset))]
	if train_split + val_split == 1:
		return train_set, val_set
	else:
		test_set = dataset[int((train_split+val_split)*len(dataset)):]
		return train_set, val_set, test_set


def load_experiment_specifications(experiment_directory):

	filename = os.path.join(experiment_directory, specifications_filename)

	if not os.path.isfile(filename):
		raise Exception(
			"The experiment directory ({}) does not include specifications file "
			+ '"specs.json"'.format(experiment_directory)
		)

	return json.load(open(filename))


def load_logs(logs_dir):

	full_filename = os.path.join(logs_dir, logs_filename)

	if not os.path.isfile(full_filename):
		raise Exception('log file "{}" does not exist'.format(full_filename))

	data = torch.load(full_filename)

	return (
		data["loss"],
		data["val_loss"],
		data["epoch"],
	)


def clip_logs(loss_log, val_loss_log, epoch):

	loss_log = loss_log[: epoch]
	val_loss_log = val_loss_log[: epoch]
	return (loss_log, val_loss_log)


# normalize x and y coordinates to 0 mean 1 std
def normalize(data):
	means = np.mean(data, axis=(0, 1))
	stds = np.std(data, axis=(0, 1))
	x_mean = means[1]
	y_mean = means[2]
	x_std = stds[1]
	y_std = stds[2]
	data[:, :, 1] = (data[:, :, 1] - x_mean) / x_std
	data[:, :, 2] = (data[:, :, 2] - y_mean) / y_std
	print("mean", np.mean(data[:, :, 1]), np.std(data[:, :, 1]))
	stop

	# for element in data:
	#     element[:,1] = (element[:,1] - x_mean) / x_std
	#     element[:,2] = (element[:,2] - y_mean) / y_std
