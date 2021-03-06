# Import statements
import numpy as np
import os
import json
from glob import glob
from tqdm import tqdm
import time
import h5py
import torch
import utils.workspace as ws
from models.generator.generator import Generator
from numpy.linalg import norm


def main(experiment_directory):

    # Loading the model
    ## Determining device to use
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # experiment_directory = "models"
    
    specs = ws.load_experiment_specifications(experiment_directory)
    checkpoint = str(specs["test"]["Checkpoint"])


    # Instantiating the generator
    specs_g = specs["generator"]
    input_dim = specs_g["InputDim"]
    n_points = specs_g["NPoints"]
    generator = Generator(input_dim, n_points).to(device)

    path_to_model_dir = os.path.join(experiment_directory,"generator", ws.model_params_dir)
    print("Loading checkpoint {} model from: {}".format(
        checkpoint, os.path.abspath(path_to_model_dir)))
    generator.load_model_parameters(path_to_model_dir, checkpoint)

    print('############# Generator Model: #####################')
    print(generator)
    print('######################################################')
    generator.to(device)
    generator.eval()


    # Loading Data 
    ## obstacles
    # obs_path = "../metrics/obstacle_zone_test.json"
    obs_path = specs["test"]["ObstaclePath"]
    obs_zone_all = json.load(open(obs_path))
    obs_zone_all = [obs_zone_all[k] for k in sorted(obs_zone_all.keys())]
    ## start_goal
    # sg_path = "../metrics/start_goal_test.json"
    start_goal_path = specs["test"]["StartGoalPath"]
    sg_all = json.load(open(start_goal_path))
    sg_all = [sg_all[k] for k in sorted(sg_all.keys())]
    ## environment codes
    # env_data_folder = "/home/user/Documents/Alassane/motionPlanning/2d/2d_trajectory_generator/2dDeepSDF/chomp256/Reconstructions/test/codes"
    env_data_folder = specs["test"]["EnvFolder"]
    filenames_env = sorted(glob(os.path.join(env_data_folder, '*.npy')))
    envs = [np.load(f).squeeze() for f in filenames_env ]

    # Evaluation function definition
    ## Generate trajectories and record computation time (to do)
    def generate_paths(sg_list_all, envs):
        """ Generate a set of trajectories for all the environments. Then record the computation time of a number of trajectories for each environment."""
        trajectories_all = []
        generation_time_all = []
        for k in tqdm(range(len(envs))):
            sg_list = sg_list_all[k]
            env = envs[k]
            sg_list = np.array(sg_list).reshape(-1, 4) / 63
            start, goal = sg_list[:, :2], sg_list[:, 2:]
            env_new = np.broadcast_to(env, (sg_list.shape[0], env.shape[0]))
            inputs = np.concatenate(
                (env_new, sg_list), axis=1)
            inputs = torch.from_numpy(inputs).to(device).float()
            with torch.no_grad():
                start_time = time.time()
                trajectory_intermediate = generator(inputs)
                end_time = time.time()
                generation_time_all.append(end_time-start_time)
                trajectory_intermediate = trajectory_intermediate.squeeze().cpu().numpy()
            trajectories = np.concatenate([start, trajectory_intermediate, goal], axis=1).reshape(-1,64,2) *63
            trajectories_all.append(trajectories)
        trajectories_all = np.stack(trajectories_all, axis=0)
        generation_time_all = np.stack(generation_time_all, axis=0)
        return trajectories_all, generation_time_all

    ## Function for checking collision
    def check_collision(traj, obs_list):
        """Check whether a trajectory collide with an environment"""
        n_collision = 0
        path_x = traj[:, 0]
        path_y = traj[:, 1]
        for o in tqdm(obs_list):
            ox, oy = o
            dx_list = [ox - x for x in path_x]
            dy_list = [oy - y for y in path_y]
            d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]
            n_collision += len([c for c in d_list if c < 0.4])
        if n_collision > 3:
            return 0  # collision
        return 1 # avoid

    ## success rate evaluation function
    def eval_success_rate(paths_all, obs_list_all):
        """Success rate of all environments"""
        # success_rate = {}
        success_rate = []
        #loop over all environments
        for k in tqdm(range(len(obs_list_all))):
            obs_list = obs_list_all[k]
            # sg_list = sg_all[k]
            paths = paths_all[k]
            # store n_avoid avoidance for each path
            n_avoid = [check_collision(path, obs_list) for path in paths]
            # success_rate[k] = n_avoid
            success_rate.append(n_avoid)
        # json.dump(success_rate, open(output_sr, "w"))
        success_rate = np.stack(success_rate, axis=0)
        return success_rate
    
    ## path cost
    def eval_path_cost(paths_all):
        """Compute the path cost of all trajectories and saves a 1-d npy array of all trajectories"""
        paths_all = paths_all 
        path_cost_all = []
        ###########
        def path_cost(paths):
            cost_list = []
            for path in paths:
                cost = 0 
                for k in range(len(path)-1):
                    p, q = path[k], path[k+1]
                    cost += norm(p-q)
                cost_list.append(cost)
            return np.array(cost_list)
        ###########
        path_cost_all = np.stack([path_cost(paths) for paths in paths_all], axis=0)
        return path_cost_all
        # np.save(output_path_cost, path_cost_all)
        # print("Saved paths costs to {}".format(output_path_cost))
    
    # Running eval
    ## getting the trajectories and generation time
    print("***********************Generating trajectories...***********************")
    paths_all, time_all = generate_paths(sg_all, envs)
    print("Shape of time array", time_all.shape)
    ## path cost
    print("***********************Computing path cost...***********************")
    path_cost_all = eval_path_cost(paths_all)
    print("Shape of path cost array", path_cost_all.shape)
    ## evaluating success rate
    print("***********************Evaluating success rate...***********************")
    # outpath_sr = specs["test"]["OutputSuccessRate"]
    success_rate = eval_success_rate(paths_all, obs_zone_all)
    print("Shape of success rate array", success_rate.shape)
    
    ## Saving all results into one file
    result_path = specs["test"]["ResultPath"]
    with h5py.File(result_path,  "w") as hf:
        hf.create_dataset('generation_time', data=time_all)
        hf.create_dataset('path_cost', data=path_cost_all)
        hf.create_dataset('success_rate', data=success_rate)
    print("Succesfully saved the evaluation results to {}".format(result_path))

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

    args = arg_parser.parse_args()

    main(args.experiment_directory)
