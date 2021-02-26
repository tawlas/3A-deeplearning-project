from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json


def main(path_to_traj_folder, path_to_env_folder, out_dir, fake):
    """ Concatenates an environment, and an interpolated trajectory.
         Saves a numpy file of many concatenations for each environment. """

    filenames_env = sorted(glob(os.path.join(path_to_env_folder, '*.npy')))
    if fake:
        filenames_traj = sorted(
        glob(os.path.join(path_to_traj_folder, '*.npy')))
    else:
        filenames_traj = sorted(
            glob(os.path.join(path_to_traj_folder, '*.json')))
    print("Found {} trajectories files and {} environment files".format(
        len(filenames_traj), len(filenames_env)))
    print("**************************Starting work**********************************")

    # Create a dict whose keys are the name of the environment abd values the set of latent trajectories
    trajectories_all = {}
    n_saved = 0
    if fake:
        for filename in tqdm(filenames_traj):
            trajectories = np.load(filename)
            key = os.path.split(filename)[1][:-4]
            trajectories_all[key] = trajectories
    else:
        for filename in tqdm(filenames_traj):
            trajectories = json.load(open(filename))
            trajectories = np.array([trajectories[k] for k in trajectories])
            # concat all traj points
            trajectories = trajectories.reshape(trajectories.shape[0], -1)
            key = os.path.split(filename)[1][:-5]
            trajectories_all[key] = trajectories

    for filename in tqdm(filenames_env):
        env = np.load(filename).squeeze()
        key = os.path.split(filename)[1][:-4]
        try:
            trajectories = trajectories_all[key]
        except:
            print("Trajectories file {} does not exist in {}".format(
                key+'.npy', path_to_traj_folder))
            continue
        env_new = np.broadcast_to(env, (trajectories.shape[0], env.shape[0]))
        lat_traj_env = np.concatenate(
            (env_new, trajectories), axis=1)

        # Saving the data
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        out_path = os.path.join(out_dir, key + '.npy')
        np.save(out_path, lat_traj_env)
        n_saved += len(lat_traj_env)
        
    print("Saved {} total data of shape {} each to {}".format(n_saved, lat_traj_env[0].shape, out_dir))


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
    arg_parser.add_argument(
        "-f",
        action='store_true',
        help="if flag given, this script will concat an env and a lat traj instead of an interpolated traj "

    )

    args = arg_parser.parse_args()
    fake=False
    if args.f:
        fake = True

    main(args.traj_folder, args.env_folder, args.out_dir, fake)
