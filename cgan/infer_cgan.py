# Import statements
import numpy as np
import os
import json
from glob import glob
from tqdm import tqdm
import torch
import utils.workspace as ws
from generator.generator import Generator


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
    input_dim = specs["InputDim"]
    output_dim = specs["OutputDim"]
    hid_dim = specs["HiddenDim"]
    n_layers = specs["NLayers"]
    dropout = specs["Dropout"]
    model = Generator(input_dim, output_dim, hid_dim,
                      n_layers, dropout).to(device)

    path_to_model_dir = os.path.join(experiment_directory, ws.model_params_dir)
    print("Loading checkpoint {} model from: {}".format(
        checkpoint, os.path.abspath(path_to_model_dir)))
    model.load_model_parameters(path_to_model_dir, checkpoint)

    print('######################################################')
    print('######################################################')
    print('############# Generator Model: #####################')
    print(model)
    print('######################################################')
    print('######################################################')
    model.to(device)
    model.eval()

    # I/O paths
    env_data_folder = specs["eval"]["EnvDataFolder"]
    start_goal_path = specs["eval"]["StartGoalPath"]

    filenames_env = sorted(glob(os.path.join(env_data_folder, '*.npy')))
    start_goal_all = json.load(start_goal_path)
    # start_goal keys and env codes have the same radical name

    # concat all env vectors with all corresponding start and appending them into one batch
    starts = []
    goals = []
    name_list = []
    env_pre = []
    for filename_env in filenames_env:
        env = np.load(filename_env)
        env_name = os.path.split(filename_env)[1][:-4] + '.jpg'
        sg_list = start_goal_all.get(env_name, None)
        if sg_list is None:
            pass
        else:
            name_list.append(tuple(env_name, len(sg_list)))
            env_pre.append(np.broadcast_to(env, (len(sg_list), env.shape[0])))
            for sg in sg_list:
                starts.append(sg[0])
                goals.append(sg[1])

    starts = np.array(starts)
    goals = np.array(goals)

    inputs = torch.from_numpy(inputs)
    env_pre = np.array(env_pre)
    env_pre = torch.from_numpy(env_pre)

    # generating trajectories
    seq_length = specs["SeqLength"]
    h = model.init_hidden(len(inputs))
    trajectories_points = []

    for timestep in range(seq_length-2):
        inputs = torch.cat([env_pre, inputs], dim=-1)
        inputs, h = model(inputs, h)
        trajectories_points.append(inputs.detach().cpu().numpy())

    # prepending and appending to intermediate points respectively start and goal
    trajectories_points = np.array(trajectories_points)
    starts = np.expand_dims(starts, 1)
    goals = np.expand_dims(goals, 1)
    trajectories_all = np.concatenate(
        [starts, trajectories_points, goals], axis=1)

    # Linking the trajectories to the corresponding environments
    trajectories_dict = {}
    current = 0
    for k in name_list:
        name, count = k
        trajectories_dict[name] = trajectories_all[current: current+count]
        current += count

    # Saving the trajectories to json format
    output_path = specs['eval']["GenerationPath"]
    json.dump(trajectories_dict, open(output_path))

    print('Successfuly saved {} trajectories in total for {} environment(s) to {} '.format(
        len(trajectories_all), len(trajectories_dict), os.path.abspath(output_path)))

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
