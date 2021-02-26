# Import statements
import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import os
from glob import glob
from tqdm import tqdm

import utils.workspace as ws
from utils.workspace import normalize
from generator.generator import Generator
from autoencoder.autoencoder.autoencoder import AutoEncoder
from discriminator.discriminator.discriminator import Discriminator
import time


# # Determining which device to use
# use_cuda = torch.cuda.is_available()
# if use_cuda:
# 	device = torch.device("cuda")
# 	print("Training on GPU")
# else:
# 	device = torch.device("cpu")
# 	print("Training on CPU")


# Loss function
def loss_function(y_pred, y_true, size_average=False):
	criterion = nn.BCELoss(reduction='sum' if not size_average else 'mean')
	return criterion(y_pred, y_true)


# Main script
def main(experiment_directory, continue_from):

	# Determining which device to use
	use_cuda = torch.cuda.is_available()
	if use_cuda:
		device = torch.device("cuda")
		print("Training on GPU")
	else:
		device = torch.device("cpu")
		print("Training on CPU")

	# Loading the training specifications/parameters
	specs = ws.load_experiment_specifications(experiment_directory)

	# -------------------
	# Loading autoencoder model
	# -------------------
	# experiment_directory_ae = '../autoencoder/autoencoder'
	experiment_directory_ae = specs["Dependencies"]["ExpDirAE"]
	specs_ae = ws.load_experiment_specifications(experiment_directory_ae)
	checkpoint_ae = str(specs_ae["eval"]["Checkpoint"])

	input_dim = specs_ae["InputDim"]
	latent_dim = specs_ae["LatentDim"]
	dropout = specs_ae["Dropout"]
	autoencoder = AutoEncoder(input_dim, latent_dim, dropout).to(device).float()

	path_to_model_dir = os.path.join(experiment_directory_ae, ws.model_params_dir)
	print("Loading checkpoint {} model from: {}".format(
		checkpoint_ae, os.path.abspath(path_to_model_dir)))
	autoencoder.load_model_parameters(path_to_model_dir, checkpoint_ae)
	for param in autoencoder.parameters():
		param.requires_grad = False
	autoencoder.eval()
	# ------------
	# End Loading autoencoder model
	# ------------

	# -------------------
	# Loading discriminator model
	# -------------------
	# Instantiating the model
	# experiment_directory_d = '../discriminator/discriminator'
	experiment_directory_d = specs["Dependencies"]["ExpDirDiscriminator"]
	specs_d = ws.load_experiment_specifications(experiment_directory_d)
	checkpoint_d = str(specs_d["eval"]["Checkpoint"])

	input_dim = specs_d["InputDim"]
	discriminator = Discriminator(input_dim).to(device)

	path_to_model_dir = os.path.join(
		experiment_directory_d, ws.model_params_dir)
	print("Loading checkpoint {} model from: {}".format(
		checkpoint_d, os.path.abspath(path_to_model_dir)))
	discriminator.load_model_parameters(path_to_model_dir, checkpoint_d)
	for param in discriminator.parameters():
		param.requires_grad = False

	discriminator.eval()
	# -------------------
	# End Loading discriminator model
	# -------------------

	# Loading the dataset
	
	data_dir = specs["DataDir"]
	filenames = sorted(glob(os.path.join(data_dir, '*.npy')))
	dataset = np.concatenate([np.load(filename) for filename in filenames], axis=0)
	print("Shape of the full dataset {}".format(dataset.shape))

	# Splitting the dataset
	train_split = specs["TrainSplit"]
	val_split = specs["ValSplit"]
	train_set, val_set = ws.split_dataset(dataset, train_split, val_split)

	# Converting the dataset from numpy to pytorch
	train_set = torch.from_numpy(train_set)
	val_set = torch.from_numpy(val_set)

	print("Training set: {}\nValidation set: {}".format(
		len(train_set), len(val_set)))

	# Instantiating the model
	input_dim = specs["InputDim"]
	output_dim = specs["OutputDim"]
	hid_dim = specs["HiddenDim"]
	n_layers = specs["NLayers"]
	# dropout = specs["Dropout"]
	model = Generator(input_dim, output_dim, hid_dim,
					  n_layers)
	model = model.to(device).float()

	# optimization function
	lr = specs["LearningRate"]
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	epochs = specs["NumEpochs"]
	start_epoch = 1
	batch_size = specs["BatchSize"]
	clip = specs["GradClip"]
	log_frequency = specs["LogFrequency"]
	logs_dir = specs["LogsDir"]
	model_params_dir = specs["ModelDir"]
	checkpoints = list(
		range(
			specs["CheckpointFrequency"],
			specs["NumEpochs"] + 1,
			specs["CheckpointFrequency"],
		)
	)

	loss_log = []
	val_loss_log = []

	if continue_from is not None:

		print('Continuing from "{}"'.format(continue_from))

		model_epoch = model.load_model_parameters(
			model_params_dir, continue_from)
		loss_log, val_loss_log, log_epoch = ws.load_logs(logs_dir)

		if not log_epoch == model_epoch:
			loss_log, val_loss_log = ws.clip_logs(
				loss_log, val_loss_log, model_epoch
			)

		start_epoch = model_epoch + 1

		print("Model loaded from epoch: {}".format(model_epoch))

	print("Starting training")
	train(model, train_set, val_set, start_epoch, epochs, batch_size, optimizer,
		  clip, log_frequency, logs_dir, model_params_dir, checkpoints, loss_log, val_loss_log, device, autoencoder, discriminator, specs)


def train(model, train_set, val_set, start_epoch, epochs, batch_size, optimizer, clip, log_frequency,
		  logs_dir, model_params_dir, checkpoints, loss_log, val_loss_log, device, autoencoder, discriminator, specs):
	discriminator = discriminator.eval()
	autoencoder = autoencoder.eval()

	for epoch in tqdm(range(start_epoch, epochs+1)):
		print('Doing epoch {} of {}'.format(epoch, epochs))
		e_start_time = time.time()
		epoch_loss = 0
		epoch_val_loss = 0
		h = model.init_hidden(batch_size)
		h = tuple([h[0].to(device).float(), h[1].to(device).float()])

		# Looping through the whole dataset
		for inputs in model.dataloader(train_set, batch_size):
			inputs = inputs.to(device).float()
			# print(inputs.shape)
			# print(inputs)
			# sto
			labels = torch.ones([batch_size]).to(device)
			h = tuple([each.data for each in h])

			# -----------------
			#  Train Generator
			# -----------------

			# zero accumulated gradients
			model.zero_grad()
			outputs, h = model(inputs, h)
			# print(outputs.shape)
			# print(outputs)
			# sto
			# with torch.no_grad():
			# Encode trajectories with autoencoder
			outputs = outputs.view(outputs.size(0), -1)
			lat_traj=autoencoder.encode(outputs)

			# Discriminator step
			env_dim = specs["EnvDim"]
			env_all = torch.stack([inputs[k][0][:env_dim] for k in range(len(inputs))], dim=0)
			env_lat_traj=torch.cat([env_all, lat_traj], dim=1)
			
			outputs=discriminator(env_lat_traj)
			

			# calculate the loss and perform backprop
			loss=loss_function(outputs, labels)
			epoch_loss += loss.item()
			loss.backward()
			# `clip_grad_norm` helps prevent the exploding gradient problem.
			nn.utils.clip_grad_norm_(model.parameters(), clip)
			optimizer.step()

		e_end_time=time.time()
		print('Total epoch {} finished in {} seconds.'
			  .format(epoch, e_end_time - e_start_time))
		loss_log.append(epoch_loss/len(train_set))
		print("epoch loss", 1000000* epoch_loss/len(train_set))
		

		# Get validation loss
		model.eval()
		with torch.no_grad():
			val_h = model.init_hidden(batch_size)
			val_h = tuple([val_h[0].to(device).float(), val_h[1].to(device).float()])
			for val_inputs in model.dataloader(val_set, batch_size):
				val_inputs=val_inputs.to(device).float()
				val_labels=torch.ones([batch_size]).to(device)
				val_h=tuple([each.data for each in val_h])

				val_outputs, val_h=model(val_inputs, val_h)

				# Encode trajectories with autoencoder
				val_outputs = val_outputs.view(val_outputs.size(0), -1)
				val_lat_traj=autoencoder.encode(val_outputs)

				# Discriminator step
				env_all = torch.stack([val_inputs[k][0][:env_dim] for k in range(len(val_inputs))], dim=0)
				env_lat_traj=torch.cat([env_all, val_lat_traj], dim=1)
				val_outputs=discriminator(env_lat_traj)

				# calculate the loss and perform backprop
				val_loss=loss_function(val_outputs, val_labels)
				epoch_val_loss += val_loss.item()

		val_loss_log.append(epoch_val_loss/len(val_set))
		print("epoch val loss", 1000000*epoch_val_loss/len(val_set))

		model.train()

		if epoch in checkpoints:
			model.save_model(model_params_dir, str(epoch)+".pth", epoch)

		if epoch % log_frequency == 0:
			model.save_model(model_params_dir, "latest.pth", epoch)
			model.save_logs(
				logs_dir,
				loss_log,
				val_loss_log,
				epoch,
			)


if __name__ == "__main__":

	import argparse

	arg_parser=argparse.ArgumentParser(
		description="Train an Unconditional Generator")
	arg_parser.add_argument(
		"--experiment",
		"-e",
		dest="experiment_directory",
		required=True,
		help="The experiment directory. This directory should include "
		+ "experiment specifications in 'specs.json"
	)
	arg_parser.add_argument(
		"--continue",
		"-c",
		dest="continue_from",
		help="A snapshot to continue from. This can be 'latest' to continue"
		+ "from the latest running snapshot, or an integer corresponding to "
		+ "an epochal snapshot.",
	)

	args=arg_parser.parse_args()

	main(args.experiment_directory, args.continue_from)
