from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json


def main(path_to_latent_traj_folder, path_to_env_folder, label, out_dir):
    """ Concatenates a latent trajectory, its corresponding latent environment and its label (avoid or collide with obstacles).
         Saves a numpy file of many concatenations for each environment. """
    filenames_env = sorted(glob(os.path.join(path_to_env_folder, '*.npy')))
    filenames_lat_traj = sorted(
        glob(os.path.join(path_to_latent_traj_folder, '*.npy')))
    print("Found {} trajectories files and {} environment files".format(
        len(filenames_lat_traj), len(filenames_env)))
    print("**************************Starting work**********************************")

    # Create a dict whose keys are the name of the environment abd values the set of latent trajectories
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    traj = {}
    label = np.array([label])
    n_saved = 0
    for filename in tqdm(filenames_lat_traj):
        trajectories = np.load(filename)
        key = os.path.split(filename)[1][:-4]
        traj[key] = trajectories

    for filename in tqdm(filenames_env):
        env = np.load(filename).squeeze()
        key = os.path.split(filename)[1][:-4]
        try:
            lat_traj = traj[key]
        except:
            print("Trajectories file {} does not exist in {}".format(
                key+'.npy', path_to_latent_traj_folder))
            continue 

        env_new = np.broadcast_to(env, (lat_traj.shape[0], env.shape[0]))
        label_new = np.broadcast_to(
            label, (lat_traj.shape[0], label.shape[0]))
        lat_traj_env = np.concatenate(
            (env_new, lat_traj, label_new), axis=1)
        out_path = os.path.join(out_dir, key + '.npy')
        np.save(out_path, lat_traj_env)
        n_saved += len(lat_traj_env)
        
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
        "--label",
        "-l",
        dest="label",
        type=int,
        choices={0, 1},
        required=True,
        help="The label of the trajectories"
    ),
    arg_parser.add_argument(
        "--output_folder",
        "-o",
        dest="out_dir",
        required=True,
        help="The directory the input to the autoencoder is to be saved in"

    )

    args = arg_parser.parse_args()

    main(args.traj_folder, args.env_folder, args.label, args.out_dir)
