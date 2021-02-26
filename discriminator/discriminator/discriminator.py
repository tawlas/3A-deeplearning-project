import argparse
import os
import numpy as np
import math

import torch.nn as nn
import torch.nn.functional as F
import torch
import utils.workspace as ws



class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.Dropout(0.05),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.2),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(256, 128),
            # nn.Dropout(0.2),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.model.apply(self.init_weights)

    def forward(self, input):
        validity = self.model(input)
        return validity.squeeze()

    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.zero_()

    def dataloader(self, dataset, batch_size):
        ''' Loads the dataset '''
        n = len(dataset)
        if batch_size > n:
            raise Exception('Batch size must be less than dataset size')
        i = 0
        while i < n:
            if i + batch_size <= n:
                yield dataset[i:i+batch_size]
                i += batch_size
            else:
                # Dropping last uncomplete batch
                i = n
    

    def save_model(self, model_params_dir, filename, epoch):
        ''' Saves the weiths of the model '''

        if not os.path.isdir(model_params_dir):
            os.makedirs(model_params_dir)

        torch.save(
            {"epoch": epoch, "model_state_dict": self.state_dict()},
            os.path.join(model_params_dir, filename),
        )

    def save_logs(self, logs_dir, loss_log, val_loss_log,precision_log,
				val_precision_log,
				recall_log,
				val_recall_log,
				accuracy_log,
				val_accuracy_log,
				roc_auc_score_log,
				val_roc_auc_score_log, epoch):
        ''' Saves the logs of the model '''

        if not os.path.isdir(logs_dir):
            os.makedirs(logs_dir)
        torch.save(
            {"epoch": epoch, "loss": loss_log, "val_loss": val_loss_log, "precision":precision_log,
				"val_precision":val_precision_log,
				"recall":recall_log,
				"val_recall":val_recall_log,
				"accuracy":accuracy_log,
				"val_accuracy":val_accuracy_log,
				"roc_auc_score":roc_auc_score_log,
				"val_roc_auc_score":val_roc_auc_score_log},
            os.path.join(logs_dir, ws.logs_filename),
        )

    def load_model_parameters(self, model_params_dir, checkpoint):
        ''' Loads the weiths of the model and return the corresponding epoch number'''

        filename = os.path.join(model_params_dir, checkpoint + ".pth")

        if not os.path.isfile(filename):
            raise Exception(
                'model state dict "{}" does not exist'.format(filename))

        data = torch.load(filename)

        self.load_state_dict(data["model_state_dict"])

        return data["epoch"]
