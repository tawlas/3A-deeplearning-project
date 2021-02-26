from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json


def main(data_folder, out_path):
    filenames = sorted(glob(os.path.join(data_folder, '*.json')))

    dataset = []
    for filename in tqdm(filenames):
        file = json.load(open(filename, 'r'))
        for k in sorted(file.keys()):
            dataset.append(file[k])
    dataset = np.array(dataset)
    print(dataset.shape)
    print(dataset.max())
    np.save(out_path, dataset)
    print('Successfuly saved {} shape data trajectories at \"{}\" '.format(
        dataset.shape, out_path))


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser(
        description="Interpolate trajectories")
    arg_parser.add_argument(
        "--traj_folder",
        "-t",
        dest="traj_folder",
        required=True,
        help="The interpolated trajectories directory.  "
    )
    arg_parser.add_argument(
        "--output_folder",
        "-o",
        dest="out_path",
        required=True,
        help="The directory the input to the autoencoder is to be saved in"

    )

    args = arg_parser.parse_args()

    main(args.traj_folder, args.out_path)
