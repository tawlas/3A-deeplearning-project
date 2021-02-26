# Import statements
import numpy as np
import os
import json
from glob import glob
from tqdm import tqdm
import torch
import utils.workspace as ws
from discriminator.discriminator import Discriminator


def main(experiment_directory, checkpoint):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    specs = ws.load_experiment_specifications(experiment_directory)

    # Instantiating the model
    dropout = specs["Dropout"]
    input_dim = specs["InputDim"]
    model = Discriminator(input_dim).to(device)

    path_to_model_dir = os.path.join(experiment_directory, ws.model_params_dir)
    print("Loading checkpoint {} model from: {}".format(
        checkpoint, os.path.abspath(path_to_model_dir)))
    model.load_model_parameters(path_to_model_dir, checkpoint)

    print('######################################################')
    print('######################################################')
    print('############# Discriminator Model: #####################')
    print(model)
    print('######################################################')
    print('######################################################')
    model.to(device)
    model.eval()

    # Prediction
    # TODO: Complete the inference


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
