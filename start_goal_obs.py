import numpy as np
import matplotlib.pyplot as plt
import os
import json
from glob import glob
from tqdm import tqdm
import random


def sample_start_goal(zone_coords):
    # 2) Select randomly the start point inside the obstacle zone
    coords_all = random.sample(zone_coords, len(zone_coords))
    start = coords_all[0]
    # 3) Filtering points at a given distance d
    #     dmin = sqrt((n-1)**2)*d_factor
    goal = coords_all[1]
    d = np.linalg.norm(np.array(goal) - np.array(start))
    for k in range(2, len(coords_all)):
        goal_new = coords_all[k]
        d_new = np.linalg.norm(np.array(goal_new) - np.array(start))
        if d < d_new:
            goal = goal_new
            d = d_new
    return start, goal


def main(coords_path, output_dir, n_sg):
    coords_dict = json.load(coords_path)
    start_goal_dict = {}
    # print("Found {} images".format(len(filenames)))

    for img_name in coords_dict:
        zone_coords = coords_dict[img_name]
        # start goal
        sg_list = []
        for k in range(n_sg):
            start, goal = sample_start_goal(zone_coords)
            sg_list.append((start, goal))

        start_goal_dict[img_name] = sg_list

    output_path_sg = os.path.join(output_dir, 'start_goal_obs.json')
    json.dump(start_goal_dict, open(output_path_sg, 'w'))
    print("Saved {} start and goal for {} images".format(
        n_sg, len(start_goal_dict)))


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser(description="Start and Goal")
    arg_parser.add_argument(
        "--coords_path",
        "-c",
        dest="zone_coords",
        required=True
    )
    arg_parser.add_argument(
        "--output_dir",
        "-o",
        dest="output_dir",
        required=True
    ),
    arg_parser.add_argument(
        "--n_sg",
        "-n",
        dest="n_sg",
        type=int,
        required=True,
        help="Number of start and goal per image"
    )

    args = arg_parser.parse_args()
    main(args.coords_folder, args.output_dir, args.n_sg)
