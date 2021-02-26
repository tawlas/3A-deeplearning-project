# Import statements
import numpy as np
import os
import json
from glob import glob
from tqdm import tqdm
import torch
import utils.workspace as ws
from autoencoder.autoencoder import AutoEncoder


def main(experiment_directory, checkpoint):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    specs = ws.load_experiment_specifications(experiment_directory)

    
    input_dim = specs["InputDim"]
    latent_dim = specs["LatentDim"]
    dropout = specs["Dropout"]
    model = AutoEncoder(input_dim, latent_dim, dropout).to(device).float()

    path_to_model_dir = os.path.join(experiment_directory, ws.model_params_dir)
    print("Loading checkpoint {} model from: {}".format(
        checkpoint, os.path.abspath(path_to_model_dir)))
    model.load_model_parameters(path_to_model_dir, checkpoint)

    print('######################################################')
    print('######################################################')
    print('############# AutoEncoder Model: #####################')
    print(model)
    print('######################################################')
    print('######################################################')
    model.to(device)
    model.eval()

    # Prediction
    ####
    def predict(filename):
        dataset = []
        # Loading the data
        file = json.load(open(filename, 'r'))
        for k in sorted(file.keys()):
            dataset.append(file[k])
        dataset = np.array(dataset)
        # Stacking all the points (x,y, x,y etc...)
        inputs = dataset.reshape(dataset.shape[0], -1)
        inputs = torch.from_numpy(inputs).to(device).float()
        latent_vectors = model.encode(inputs).cpu().detach().numpy()
        return latent_vectors
    #####

    pred_dir = specs['eval']["PredictionDir"]
    data_folder = specs["eval"]["DataFolder"]
    if not os.path.isdir(pred_dir):
        os.makedirs(pred_dir)
  #############
    filenames = sorted(glob(os.path.join(data_folder, '*.json')))
    if len(filenames) == 0:
        print('No json file was found in {}'.format(data_folder))
        exit()
    n_latent_vectors = 0
    for filename in tqdm(filenames):
        latent_vectors = predict(filename)
        pred_path = os.path.join(
            pred_dir, os.path.split(filename)[1][:-5] + '.npy')
        np.save(pred_path, latent_vectors)
        n_latent_vectors += len(latent_vectors)

    print('Successfuly saved {} latent vectors in \"{}\" '.format(
        n_latent_vectors, os.path.abspath(pred_dir)))

  #############


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include "
        + "experiment specifications in 'specs.json"
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="A snapshot to continue from. This can be 'latest' to continue"
        + "from the latest running snapshot, or an integer corresponding to "
        + "an epochal snapshot.",
    )

    args = arg_parser.parse_args()

    main(args.experiment_directory, args.checkpoint)
