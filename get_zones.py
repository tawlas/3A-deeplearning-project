import numpy as np
import matplotlib.pyplot as plt
import os
import json
from glob import glob
from tqdm import tqdm
import random


def get_zone(img):
    free_zone = []
    obstacle_coords = []
    h = img.shape[0]  # heigth of the image
    w = img.shape[1]  # width of the image
    for i in range(h):
        for j in range(w):
            if img[i, j] == 0:
                free_zone.append((i, j)) # j, i order cause otherwise x and y coordinates are reverted
            else:
                obstacle_coords.append((i, j)) # j, i order cause otherwise x and y coordinates are reverted
    return free_zone, obstacle_coords


def main(img_folder, output_dir, suffix):
    filenames = sorted(glob(os.path.join(img_folder, '*.jpg')))
    print("Found {} images".format(len(filenames)))
    free_zone_dict = {}
    obsctacle_zone_dict = {}
    for img_path in tqdm(filenames):
        img = np.array(plt.imread(img_path))
        # normalize img
        for k in img:
            for i in range(len(k)):
                if k[i] > 50:
                    k[i] = 255
                else:
                    k[i] = 0
        img = img//255

        # get zones
        free_zone, obstacle_coords = get_zone(img)

        img_name = os.path.split(img_path)[1]
        free_zone_dict[img_name] = free_zone
        obsctacle_zone_dict[img_name] = obstacle_coords
    if suffix:
        suffix = "_" + suffix
    output_path_fz = os.path.join(output_dir, 'free_zone' + suffix + '.json')
    output_path_oz = os.path.join(output_dir, 'obstacle_zone' + suffix + '.json')
    json.dump(free_zone_dict, open(output_path_fz, 'w'))
    json.dump(obsctacle_zone_dict, open(output_path_oz, 'w'))
    print("Saved free zone and obstacle zone for {} images".format(
        len(free_zone_dict)))


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser(description="Start and Goal")
    arg_parser.add_argument(
        "--img_folder",
        "-i",
        dest="img_folder",
        required=True
    )
    arg_parser.add_argument(
        "--output_dir",
        "-o",
        dest="output_dir",
        required=True
    )
    arg_parser.add_argument(
        "--suffix",
        "-s",
        dest="suffix",
        default="",
        help="Suffix to add to the name of the free and obstacle zone files before the .json to help differentiate training and testing files"
    )
    args = arg_parser.parse_args()
    main(args.img_folder, args.output_dir, args.suffix)
