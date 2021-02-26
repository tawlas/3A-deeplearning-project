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
from models.generator.generator import Generator
from autoencoder.autoencoder.autoencoder import AutoEncoder
from models.discriminator.discriminator import Discriminator
import time



# Loss function
def loss_function_dis(y_pred, y_true, size_average=True):
	criterion = nn.BCELoss(reduction='sum' if not size_average else 'mean')
	return criterion(y_pred, y_true)

def loss_function_traj(y_pred, y_true, size_average=True):
	criterion = nn.MSELoss(reduction='sum' if not size_average else 'mean')
	return criterion(y_pred, y_true)

# Main script
def main(experiment_directory, continue_from):
	# seeding for random shuffling of the dataset
	np.random.seed(1)

	# Determining which device to use
	use_cuda = torch.cuda.is_available()
	if use_cuda:
		device = torch.device("cuda")
		print("Training on GPU")
	else:
		device = torch.device("cpu")
		print("Training on CPU")

	# experiment_directory_d = experiment_directory + "discriminator"
	# experiment_directory_g = experiment_directory + "generator"
	# Loading the training specifications/parameters
	# specs_d = ws.load_experiment_specifications(experiment_directory_d)
	# specs_g = ws.load_experiment_specifications(experiment_directory_g)
	specs = ws.load_experiment_specifications(experiment_directory)

	# -------------------
	# Loading autoencoder model
	# -------------------
	# experiment_directory_ae = '../autoencoder/autoencoder'
	experiment_directory_ae = specs["ExpDirAE"]
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


	# Loading the dataset
	real_data_path = specs["RealDataDir"]
	fake_data_path = specs["FakeDataDir"]
	filenames_real = sorted(glob(os.path.join(real_data_path, "*.npy")))
	filenames_fake = sorted(glob(os.path.join(fake_data_path, "*.npy")))
	
	real_dataset = np.concatenate([np.load(f) for f in filenames_real[:]], axis=0)
	np.random.shuffle(real_dataset)
	fake_dataset = np.concatenate([np.load(f) for f in filenames_fake], axis=0)
	np.random.shuffle(fake_dataset)
	# Handling uneven length dataset for when loading a batch in training. TODO: think of a better way by considering uneven dataset 
	min_len = min(len(real_dataset), len(fake_dataset))
	real_dataset = real_dataset[:min_len]
	fake_dataset = fake_dataset[:min_len]
	print("Shape of real dataset", real_dataset.shape)
	print("Shape of fake dataset", fake_dataset.shape)
	# print(real_dataset[:100, -1])
	# sto
	
	# # concat real and fake data and shuffle
	# dataset = np.concatenate([real_dataset, fake_dataset], axis=0)
	# np.random.shuffle(dataset)
	# print("Shape of the dataset", dataset.shape)


	# # Splitting the dataset
	# train_split = specs["TrainSplit"]
	# val_split = specs["ValSplit"]
	# train_set, val_set = ws.split_dataset(dataset, train_split, val_split)

	# Converting the dataset from numpy to pytorch
	real_dataset = torch.from_numpy(real_dataset)
	fake_dataset = torch.from_numpy(fake_dataset)
	# train_set = torch.from_numpy(train_set)
	# val_set = torch.from_numpy(val_set)

	# print("Training set: {}\nValidation set: {}".format(
	# 	len(train_set), len(val_set)))

	# Instantiating the generator
	specs_g = specs["generator"]
	input_dim = specs_g["InputDim"]
	n_points = specs_g["NPoints"]
	generator = Generator(input_dim, n_points).to(device)
	# optimization function
	lr_g = specs_g["LearningRate"]
	optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr_g)
	clip_g = specs_g["GradClip"]
	logs_dir_g = specs_g["LogsDir"]
	model_params_dir_g = specs_g["ModelDir"]
	

	# Instantiating the discriminator
	specs_d = specs["discriminator"]
	# input_dim = train_set.size(-1) - 1 # for the label appended at the end
	input_dim = specs_d["InputDim"] # for the label appended at the end
	discriminator = Discriminator(input_dim).to(device)
	# optimization function
	lr_d = specs_d["LearningRate"]
	optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d)
	clip_d = specs_d["GradClip"]
	logs_dir_d = specs_d["LogsDir"]
	model_params_dir_d = specs_d["ModelDir"]


	epochs = specs["NumEpochs"]
	start_epoch = 1
	batch_size = specs["BatchSize"]
	log_frequency = specs["LogFrequency"]
	checkpoints = list(
		range(
			specs["CheckpointFrequency"],
			specs["NumEpochs"] + 1,
			specs["CheckpointFrequency"],
		)
	)

	loss_log_g = []
	loss_log_d = []
	# val_loss_log = []

	# TODO: adjust this if code below 
	if continue_from is not None:

		print('Continuing from "{}"'.format(continue_from))

		model_epoch = generator.load_model_parameters(
			model_params_dir_g, continue_from)
		loss_log, val_loss_log, log_epoch = ws.load_logs(logs_dir)
		

		if not log_epoch == model_epoch:
			loss_log, val_loss_log = ws.clip_logs(
				loss_log, val_loss_log, model_epoch
			)

		start_epoch = model_epoch + 1

		print("Model loaded from epoch: {}".format(model_epoch))

	print("Starting training")
	train(generator, discriminator, real_dataset, fake_dataset, start_epoch, epochs, batch_size, optimizer_g, optimizer_d,
		  clip_g, clip_d, log_frequency, logs_dir_g,logs_dir_d , model_params_dir_g, model_params_dir_d, checkpoints, loss_log_g, loss_log_d, device, autoencoder)


def train(generator, discriminator, real_dataset, fake_dataset, start_epoch, epochs, batch_size,optimizer_g, optimizer_d, clip_g, clip_d, log_frequency,
		  logs_dir_g, logs_dir_d, model_params_dir_g, model_params_dir_d, checkpoints, loss_log_g, loss_log_d, device, autoencoder):
	n_points = generator.n_points

	for epoch in tqdm(range(start_epoch, epochs+1)):
		print('Doing epoch {} of {}'.format(epoch, epochs))
		e_start_time = time.time()
		epoch_loss_g = 0
		epoch_loss_d = 0
		# epoch_val_loss = 0

		# Looping through the whole dataset
		for data in zip(generator.dataloader(real_dataset, batch_size), discriminator.dataloader(fake_dataset, batch_size)):
			real_data = data[0].to(device).float()
			fake_data = data[1].to(device).float()
			
			# retrieve and concat env, start and goal
			env = real_data[:, :-2*n_points]
			start = real_data[:,-2*n_points:-2*n_points+2]
			goal = real_data[:,-2:]
			trajectory_gt = real_data[:,-2*n_points:]
			real_inputs = torch.cat([env, start , goal], dim=1)
			# labels for mse distance between the generated trajectory and the ground_truth traj
			labels_traj = real_data[:,-2*n_points+2:-2]
			# labels for discriminator
			valid = torch.ones([batch_size,1]).to(device).float()
			fake = torch.zeros([batch_size,1]).to(device).float()

			# -----------------
			#  Train Generator
			# -----------------

			# zero accumulated gradients
			optimizer_g.zero_grad()

			gen_outputs = generator(real_inputs)

			# L2 Loss of trajectories
			loss_trajectory = loss_function_traj(gen_outputs, labels_traj)

			# Discriminator Loss
			gen_traj = torch.cat([start, gen_outputs, goal], dim=1)
			gen_traj = gen_traj.view(-1, 2*n_points)
			# # Encode trajectories with autoencoder
			lat_traj = autoencoder.encode(gen_traj)
			# print("Lat traj requires grad", lat_traj.requires_grad)
			# # concat env and lat traj
			env_lat_traj = torch.cat([env, lat_traj], dim=1)
			# print("Env requires grad", env.requires_grad)
			# # Discriminator step
			validity_gen = discriminator(env_lat_traj)
			# print("validtygen shape", validity_gen.shape)
			# # compute the loss and perform backprop
			g_loss_d = loss_function_dis(validity_gen, valid)

			# total loss
			g_loss = g_loss_d + loss_trajectory / 2

			epoch_loss_g += g_loss.item()
			g_loss.backward()
			# `clip_grad_norm` helps prevent the exploding gradient problem.
			nn.utils.clip_grad_norm_(generator.parameters(), clip_g)
			optimizer_g.step()

			# ---------------------
			#  Train Discriminator
			# ---------------------

			# zero accumulated gradients
			optimizer_d.zero_grad()

			# Loss for real images
			lat_traj_gt = autoencoder.encode(trajectory_gt)
			env_lat_traj_gt = torch.cat([env, lat_traj_gt], dim=1)
			
			validity_real = discriminator(env_lat_traj_gt)
			d_loss_real = loss_function_dis(validity_real, valid)

			# Loss for fake images
			validity_fake = discriminator(fake_data)
			d_loss_fake = loss_function_dis(validity_fake, fake)

			# Total discriminator loss
			d_loss = (d_loss_real + d_loss_fake) / 2

			epoch_loss_d += d_loss.item()
			d_loss.backward()
			# `clip_grad_norm` helps prevent the exploding gradient problem.
			nn.utils.clip_grad_norm_(discriminator.parameters(), clip_d)
			optimizer_d.step()



		e_end_time = time.time()
		
		print('Total epoch {} finished in {} seconds.'
			  .format(epoch, e_end_time - e_start_time))
		loss_log_g.append(epoch_loss_g/len(real_dataset))
		loss_log_d.append(epoch_loss_d/len(fake_dataset))
		print("epoch loss G", 1000000* epoch_loss_g/len(real_dataset))
		print("epoch loss D", 1000000* epoch_loss_d/len(fake_dataset))
		

		# # Get validation loss
		# model.eval()
		# for val_data in model.dataloader(val_set, batch_size):
		# 	val_data = val_data.to(device).float()
		# 	# retrieve env, start and goal
		# 	val_inputs = torch.cat([val_data[:,:-2*n_points], val_data[:,-2*n_points:-2*n_points+2], val_data[:,-2:]], dim=1)
		# 	labels_traj = val_data[:,-2*n_points+2:-2]
		# 	labels_dis = torch.ones([batch_size]).to(device)

		# 	# zero accumulated gradients
		# 	model.zero_grad()
		# 	val_outputs = model(val_inputs)

		# 	# L2 Loss of trajectories
		# 	val_loss_trajectory = loss_function_traj(val_outputs, labels_traj)

		# 	# Discriminator Loss
		# 	val_outputs = torch.cat([val_data[:,-2*n_points:-2*n_points+2],val_outputs, val_data[:,-2:]], dim=1)
		# 	val_outputs = val_outputs.view(-1, n_points, 2)
		# 	# concat x and y dim: all the x and the all the y for each trajectory since autoencoder was train that manner
		# 	val_outputs = torch.cat([val_outputs[:, :, 0], val_outputs[:, :, 1]], dim=-1)
			
		# 	  # Encode trajectories with autoencoder
		# 	lat_traj = autoencoder.encode(val_outputs)
		# 	  # Discriminator step
		# 	env_lat_traj = torch.cat([val_data[:,:-2*n_points], lat_traj], dim=1)
		# 	val_outputs = discriminator(env_lat_traj)
		# 	  # calculate the loss and perform backprop
		# 	val_loss_discriminator = loss_function_dis(val_outputs, labels_dis)

		# 	# total loss
		# 	val_loss = val_loss_discriminator + val_loss_trajectory

		# 	epoch_val_loss += val_loss.item()

		# val_loss_log.append(epoch_val_loss/len(val_set))
		# print("epoch val loss", 1000000*epoch_val_loss/len(val_set))

		# model.train()

		if epoch in checkpoints:
			generator.save_model(model_params_dir_g, str(epoch)+".pth", epoch)
			discriminator.save_model(model_params_dir_d, str(epoch)+".pth", epoch)

		if epoch % log_frequency == 0:
			generator.save_model(model_params_dir_g, "latest.pth", epoch)
			discriminator.save_model(model_params_dir_d, "latest.pth", epoch)
			generator.save_logs(
				logs_dir_g,
				loss_log_g,
				# val_loss_log,
				loss_log_d,
				epoch,
			)
			discriminator.save_logs(
				logs_dir_d,
				loss_log_d,
				# val_loss_log,
				[],
				epoch,
			)


if __name__ == "__main__":

	import argparse

	arg_parser = argparse.ArgumentParser(
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

	args = arg_parser.parse_args()

	main(args.experiment_directory, args.continue_from)
