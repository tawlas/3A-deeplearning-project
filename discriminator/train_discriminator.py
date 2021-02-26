# Import statements
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import utils.workspace as ws
from utils.workspace import normalize
from discriminator.discriminator import Discriminator
import time
from glob import glob

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report as cr


# Loss function
def loss_function(y_pred, y_true, size_average=True):
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

	# Loading the dataset
	real_data_path = specs["RealDataDir"]
	fake_data_path = specs["FakeDataDir"]
	filenames_real = sorted(glob(os.path.join(real_data_path, "*.npy")))[:-300]
	filenames_fake = sorted(glob(os.path.join(fake_data_path, "*.npy")))[:-100]
	real_dataset = np.concatenate([np.load(f) for f in filenames_real[:]], axis=0)
	np.random.shuffle(real_dataset)
	fake_dataset = np.concatenate([np.load(f) for f in filenames_fake], axis=0)
	# print(real_dataset[:100, -1])
	# sto
	np.random.shuffle(fake_dataset)

	# concat real and fake data and shuffle
	dataset = np.concatenate([real_dataset, fake_dataset], axis=0)
	np.random.shuffle(dataset)
	print("Shape of the dataset", dataset.shape)
	

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
	dropout = specs["Dropout"]
	# input_dim = specs["InputDim"]
	input_dim = train_set.size(-1) - 1 # for the label appended at the end
	
	model = Discriminator(input_dim).to(device)

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
	precision_log = []
	recall_log = []
	accuracy_log = []
	roc_auc_score_log = []
	val_loss_log = []
	val_precision_log = []
	val_recall_log = []
	val_accuracy_log = []
	val_roc_auc_score_log = []

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
		  clip, log_frequency, logs_dir, model_params_dir, checkpoints, loss_log, val_loss_log,precision_log,
			val_precision_log,
			recall_log,
			val_recall_log,
			accuracy_log,
			val_accuracy_log,
			roc_auc_score_log,
			val_roc_auc_score_log, device)


def train(model, train_set, val_set, start_epoch, epochs, batch_size, optimizer, clip, log_frequency,
		  logs_dir, model_params_dir, checkpoints, loss_log, val_loss_log,precision_log,
			val_precision_log,
			recall_log,
			val_recall_log,
			accuracy_log,
			val_accuracy_log,
			roc_auc_score_log,
			val_roc_auc_score_log,
			device):

	for epoch in tqdm(range(start_epoch, epochs+1)):
		print('Doing epoch {} of {}'.format(epoch, epochs))
		e_start_time = time.time()
		epoch_loss = 0
		epoch_val_loss = 0

		y_true = []
		y_pred = []
		# Looping through the whole dataset
		for data in model.dataloader(train_set, batch_size):
			data = data.float().to(device)
			# retrieve inputs and labels
			inputs = data[:, :-1]
			labels = data[:, -1]

			# inputs: concat of env and lat traj
			# labels: 0 or 1 for fake or real (collide or avoid )

			# zero accumulated gradients
			model.zero_grad()
			outputs = model(inputs)
			print("outputs", outputs)
			print("labels", labels)
			

			# collecting y_pred and y_true for metric measurements
			y_pred.extend([1 if p.item() >= 0.5 else 0 for p in outputs])
			y_true.extend([int(e.item()) for e in labels])

			# calculate the loss and perform backprop
			loss = loss_function(outputs, labels)
			epoch_loss += loss.item()
			loss.backward()
			# `clip_grad_norm` helps prevent the exploding gradient problem.
			nn.utils.clip_grad_norm_(model.parameters(), clip)
			optimizer.step()

		e_end_time = time.time()
		print('Total epoch {} finished in {} seconds.'
			  .format(epoch, e_end_time - e_start_time))
		n_iter_train = len(train_set) / batch_size
		loss_log.append(epoch_loss/len(train_set))
		print("epoch loss:", epoch_loss/len(train_set))
		precision_log.append(precision_score(y_true, y_pred))
		recall_log.append(recall_score(y_true, y_pred))
		accuracy_log.append(accuracy_score(y_true, y_pred))
		# roc_auc_score_log.append(roc_auc_score(y_true, y_pred))
		roc_auc_score_log.append(0)

		# Get validation loss
		model.eval()
		val_y_true = []
		val_y_pred = []
		for val_data in model.dataloader(val_set, batch_size):
			val_data = val_data.float().to(device)
			val_inputs = val_data[:, :-1]
			val_labels = val_data[:, -1]

			val_outputs = model(val_inputs)
			print("Sum of val outputs", val_outputs.sum())
			# collecting y_pred and y_true for metric measurements
			val_y_pred.extend([ 1 if p.item() >= 0.5 else 0 for p in val_outputs])
			val_y_true.extend([int(e.item()) for e in val_labels])

			val_loss = loss_function(val_outputs.squeeze(), val_labels)
			epoch_val_loss += val_loss.item()
			
		n_iter_val = len(val_set) / batch_size
		val_loss_log.append(epoch_val_loss/len(val_set))
		print("epoch val loss:", epoch_val_loss/len(val_set))
		val_precision_log.append(precision_score(val_y_true, val_y_pred))
		val_recall_log.append(recall_score(val_y_true, val_y_pred))
		val_accuracy_log.append(accuracy_score(val_y_true, val_y_pred))
		# val_roc_auc_score_log.append(roc_auc_score(val_y_true, val_y_pred))
		val_roc_auc_score_log.append(0)

		model.train()

		if epoch in checkpoints:
			model.save_model(model_params_dir, str(epoch)+".pth", epoch)

		if epoch % log_frequency == 0:
			model.save_model(model_params_dir, "latest.pth", epoch)
			model.save_logs(
				logs_dir,
				loss_log,
				val_loss_log,
				precision_log,
				val_precision_log,
				recall_log,
				val_recall_log,
				accuracy_log,
				val_accuracy_log,
				roc_auc_score_log,
				val_roc_auc_score_log,
				epoch
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
