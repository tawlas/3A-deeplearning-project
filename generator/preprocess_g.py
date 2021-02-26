from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json


def main(path_to_traj_folder, path_to_env_folder, out_dir):
    """ Concatenates a latent environment and the dataset of interpolated trajectories (for each point) inside that env.
         Saves a numpy file of several concatenations for each environment. """
    filenames_env = sorted(glob(os.path.join(path_to_env_folder, '*.npy')))
    filenames_traj = sorted(
        glob(os.path.join(path_to_traj_folder, '*.json')))
    print("Found {} trajectories files and {} environment files".format(
        len(filenames_traj), len(filenames_env)))
    print("**************************Starting work**********************************")

    # Create a dict whose keys are the name of the environments (or images) and values the set of latent trajectories inside of them.
    traj_all = {}
    n_saved = 0  # to know the number of concatenations made.
    for filename in tqdm(filenames_traj):
        trajectory_dict = json.load(open(filename))
        key = os.path.split(filename)[1][:-5]
        traj_array = np.array([trajectory_dict[k] for k in trajectory_dict])
        traj_all[key] = traj_array

    for filename in tqdm(filenames_env):
        env = np.load(filename).squeeze()
        key = os.path.split(filename)[1][:-4]
        try:
            trajs = traj_all[key]
        except:
            print("Trajectories file {} does not exist in {}".format(
                    key+'.npy', path_to_traj_folder))
        env_new = np.broadcast_to(
            env, (trajs.shape[0], trajs.shape[1], env.shape[0]))
        traj_env = np.concatenate(
            (env_new, trajs), axis=2)
        out_path = os.path.join(out_dir, key + '.npy')
        np.save(out_path, traj_env)
        n_saved += len(traj_env)
        
    print("Saved {} total data to {}".format(n_saved, out_dir))


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser(
        description="Concatenate Traj and Env codes")
    arg_parser.add_argument(
        "--traj_folder",
        "-t",
        dest="traj_folder",
        required=True,
        help="The interpolated trajectories directory.  "
    ),
    arg_parser.add_argument(
        "--env_folder",
        "-e",
        dest="env_folder",
        required=True,
        help="The interpolated trajectories directory.  "
    ),

    arg_parser.add_argument(
        "--output_folder",
        "-o",
        dest="out_dir",
        required=True,
        help="The directory the input to the autoencoder is to be saved in"

    )

    args = arg_parser.parse_args()

    main(args.traj_folder, args.env_folder, args.out_dir)
