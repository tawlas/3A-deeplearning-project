import argparse
import os
import numpy as np
import math
import utils.workspace as ws

import torch.nn as nn
import torch.nn.functional as F
import torch

# if torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
#     device = torch.device("cpu")


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers):
        super().__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # self.embedding = nn.Embedding(output_dim, input_dim)

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)

        self.fc_out = nn.Linear(hidden_dim, output_dim)
        nn.init.xavier_uniform_(self.fc_out.weight)

        # self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden):

        # input = [batch size, seq_len, input_dim]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]

        output, hidden = self.lstm(input, hidden)

        # output = [batch size, seq len, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]


        prediction = torch.sigmoid(self.fc_out(output.squeeze(0)))

        # prediction = [batch size,seq len,  output dim]

        return prediction, hidden

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden

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

    def save_logs(self, logs_dir, loss_log, val_loss_log, epoch):
        ''' Saves the logs of the model '''

        if not os.path.isdir(logs_dir):
            os.makedirs(logs_dir)
        torch.save(
            {"epoch": epoch, "loss": loss_log, "val_loss": val_loss_log},
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
