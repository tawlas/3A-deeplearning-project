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
from autoencoder.autoencoder import AutoEncoder
import time

# Loss function


def loss_function(y_pred, y_true, size_average=False):
    criterion = nn.MSELoss(reduction='sum' if not size_average else 'mean')
    return criterion(y_pred, y_true)


def main(experiment_directory, continue_from):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
        print("Training on GPU")
    else:
        device = torch.device("cpu")
        print("Training on CPU")

    specs = ws.load_experiment_specifications(experiment_directory)

    # Loading the dataset
    data_path = specs["DataPath"]
    dataset = np.load(os.path.join(data_path))

    # Stacking all the points  x,y, x, y etc..
    dataset = dataset.reshape(dataset.shape[0], -1)
    input_dim = dataset.shape[-1]

    # Splitting the dataset
    train_split = specs["TrainSplit"]
    val_split = specs["ValSplit"]
    train_set, val_set = ws.split_dataset(dataset, train_split, val_split)

    train_set = torch.from_numpy(train_set)
    val_set = torch.from_numpy(val_set)

    print("Training set: {}\nValidation set: {}".format(
        len(train_set), len(val_set)))

    # normalize x and y coordinates to 0 mean and 1 std
    # normalize(train_set)
    # normalize(val_set)

    # Instantiating the model

    latent_dim = specs["LatentDim"]
    dropout = specs["Dropout"]
    model = AutoEncoder(input_dim, latent_dim, dropout).to(device).float()

    # optimization function
    lr = specs["LearningRate"]
    print("learning rate #################################", lr)
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
          clip, log_frequency, logs_dir, model_params_dir, checkpoints, loss_log, val_loss_log, device)


def train(model, train_set, val_set, start_epoch, epochs, batch_size, optimizer, clip, log_frequency,
          logs_dir, model_params_dir, checkpoints, loss_log, val_loss_log, device):

    for epoch in tqdm(range(start_epoch, epochs+1)):
        print('Doing epoch {} of {}'.format(epoch, epochs))
        e_start_time = time.time()
        epoch_loss = 0
        epoch_val_loss = 0

        # Looping through the whole dataset
        for inputs in model.dataloader(train_set, batch_size):
            inputs = inputs.to(device).float()
            # zero accumulated gradients
            model.zero_grad()
            outputs = model(inputs)
            # print("outputs", outputs)
            # print("inputs", inputs)

            # calculate the loss and perform backprop
            loss = loss_function(outputs, inputs)
            epoch_loss += loss.item()
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem.
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        e_end_time = time.time()
        print('Total epoch {} finished in {} seconds.'
              .format(epoch, e_end_time - e_start_time))
        # loss_log.append(epoch_loss/len(train_set))
        print("epoch loss", epoch_loss/len(train_set))
        loss_log.append(epoch_loss)

        # Get validation loss
        model.eval()
        for val_inputs in model.dataloader(val_set, batch_size):
            val_inputs = val_inputs.to(device).float()

            val_outputs = model(val_inputs)
            val_loss = loss_function(val_outputs, val_inputs)
            epoch_val_loss += val_loss.item()

        # val_loss_log.append(epoch_val_loss/len(val_set))
        val_loss_log.append(epoch_val_loss)
        print("epoch val loss", epoch_val_loss/len(val_set))

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
