import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init
from torch.autograd import Variable
import os
import utils.workspace as ws


class AutoEncoder(nn.Module):
    def __init__(self,input_dim, latent_dim, dropout):
        super().__init__()
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.encoder = nn.Sequential(
            *block(input_dim, 512, normalize=False),
            *block(512, 512),
            *block(512, 256),
            *block(256, 256),
            nn.Linear(256, latent_dim),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            *block(latent_dim, 256, normalize=False),
            *block(256, 256),
            *block(256, 512),
            *block(512, 512),
            nn.Linear(512, input_dim),
            
        )


    def forward(self, x):
        return self.decode(self.encode(x))


    def encode(self, x):
        x = self.encoder(x)
        return x

    def decode(self, x):
        ## decode ##
        x = self.decoder(x)
        return x

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
            { "epoch": epoch, "loss": loss_log, "val_loss": val_loss_log},
            os.path.join(logs_dir, ws.logs_filename),
        )

    def load_model_parameters(self, model_params_dir, checkpoint):
        ''' Loads the weiths of the model and return the corresponding epoch number'''

        filename = os.path.join(model_params_dir, checkpoint + ".pth")

        if not os.path.isfile(filename):
            raise Exception('model state dict "{}" does not exist'.format(filename))

        data = torch.load(filename)

        self.load_state_dict(data["model_state_dict"])

        return data["epoch"]
