from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d


def main(data_folder, output_folder, x_size, y_size, n_points):

    filenames = sorted(
        glob(os.path.join(data_folder, '*.json')))

    print("Found {} files".format(len(filenames)))

    def extrapolate_traj(sample, n_points):
        """ Extrapolates a trajectory """
        ###############
        def get_spline(sample):
            """ Fits a spline to a trajectory """
            sample_sorted = np.array(sorted(sample, key=lambda x: x[0]))
            s = InterpolatedUnivariateSpline(
                sample_sorted[:, 0], sample_sorted[:, 1], k=1)
            # s = interp1d(
            #     sample_sorted[:, 0], sample_sorted[:, 1])
            return s
        ###############

        def get_xnew(x_size, y_size):
            """
            Gets the x range trajectory
            """

            x_min = 1
            while spline(x_min) > y_size-1 or spline(x_min) < 0:
                x_min += 1

            # right
            x_max = x_size - 1
            while spline(x_max) > y_size-1 or spline(x_max) < 0:
                x_max -= 1

            x_new = np.linspace(x_min, x_max, num=n_points)

            return x_new

        ###############

        spline = get_spline(sample)
        x_new = get_xnew(x_size, y_size)
        y_new = spline(x_new)
        if y_new.max() > 63:
            # print(sample)
            print(y_new.max())
           
        # normalizing the data for training after.
        x_new = x_new / x_size
        y_new = y_new / y_size
        sample_new = np.stack([y_new, x_new], axis=1) # y then x because rrtstar algo do it the other way around so we need to restore
        return sample_new

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    traj_logger = 0
    for filename in tqdm(filenames):
        file = json.load(open(filename, 'r'))
        file_new = {}
        for k in sorted(file.keys()):
            sample = np.array(file[k])
            sample_new = extrapolate_traj(sample, n_points)
            file_new[k] = sample_new.tolist()
        traj_logger += len(file_new)
        with open(os.path.join(output_folder, os.path.split(filename)[1]), 'w') as out_path:
            json.dump(file_new, out_path)
    print("Saved a total of {} interpolated trajectories to {}".format(
        traj_logger, output_folder))


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser(
        description="Interpolate trajectories")
    arg_parser.add_argument(
        "--data_folder",
        "-d",
        dest="data_folder",
        required=True,
        help="The data directory. This directory must contain the trajectories folder and a trajectories_interpolated folder "
    ),
    arg_parser.add_argument(
        "--output_folder",
        "-o",
        dest="output_folder",
        required=True,
        help=" "
    ),

    arg_parser.add_argument(
        "--n_points",
        "-n",
        dest="n_points",
        default=64,
        type=int,
        help="The number of points a trajectory will be composed of"

    ),
    arg_parser.add_argument(
        "--x_size",
        "-x",
        dest="x_size",
        default=64,
        type=int,
        help="The size of the image in x dimension (number of pixel in x dim)"
    ),
    arg_parser.add_argument(
        "--y_size",
        "-y",
        dest="y_size",
        default=64,
        type=int,
        help="The size of the image in x dimension (number of pixel in x dim)"
    )

    args = arg_parser.parse_args()

    main(args.data_folder, args.output_folder,
         args.x_size, args.y_size, args.n_points)
