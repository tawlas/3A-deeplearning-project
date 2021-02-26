import numpy as np
import matplotlib.pyplot as plt
import os
import json
from glob import glob
from tqdm import tqdm
import random
from math import sqrt


def sample_start_goal(free_zone, d_factor=0.8):
    
    # 2) Select randomly the start point outside the obstacle zone
    l = len(free_zone)
    random_samples = random.sample(free_zone,l)
    start = random_samples[0]
    # 3) Filtering points at a given distance d
    n = 64
    dmin = sqrt((n-1)**2)*d_factor
    goal = random_samples[1]
    d0 = np.linalg.norm(np.array(goal) - np.array(start))
    d = d0
    sample_goal = tuple(goal)
    it = 0
    for k in range(2, l):
        sample_goal = random_samples[k]
        d = np.linalg.norm(np.array(sample_goal) - np.array(start))
        if d >= dmin:
            goal = sample_goal
            return (start, goal)
        if d0 < d:
            d0 = d
            goal = sample_goal

    return (start, goal)


def main(coords_path, output_dir, n_sg, suffix=""):
    coords_dict = json.load(open(coords_path))
    start_goal_dict = {}
    print("Found {} env coords list".format(len(coords_dict)))

    for img_name in tqdm(coords_dict):
        zone_coords = coords_dict[img_name]
        # start goal
        sg_list = []
        for k in range(n_sg):
            start, goal = sample_start_goal(zone_coords)
            sg_list.append((start, goal))

        start_goal_dict[img_name] = sg_list
    if args.suffix:
        suffix = "_" + suffix
    output_path_sg = os.path.join(output_dir, 'start_goal' + suffix + '.json')
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
    arg_parser.add_argument(
        "--suffix",
        "-s",
        dest="suffix",
        default="",
        help="Number of start and goal per image"
    )

    args = arg_parser.parse_args()
    main(args.zone_coords, args.output_dir, args.n_sg, args.suffix)
