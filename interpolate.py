from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d, UnivariateSpline


def main(data_folder, output_folder, n_points):
    x_size = y_size = 64

    filenames = sorted(glob(os.path.join(data_folder, '*.json')))

    # def interpolate_traj(sample, n_points):
        
    #     ###############
    #     def spline(sample):
    #         data_sorted = np.array(sorted(sample, key=lambda x: x[0]))
    #         s = interp1d(data_sorted[:, 0], data_sorted[:, 1])
    #         # try:
    #         #     s = interp1d(data_sorted[:, 0], data_sorted[:, 1], kind="quadratic")
    #         # except:
    #         #     s = interp1d(data_sorted[:, 0], data_sorted[:, 1], kind="slinear")
    #         #     print("exc")
                
    #         return s
    #     ###############

    #     s = spline(sample)
    #     x = sample[:, 0]
    #     x_new = np.linspace(np.min(x), np.max(x), num=n_points)
    #     y_new = s(x_new)
    #     # if y_new.max() > 63:
    #     #     # print(sample)
    #     #     print(y_new.max())
    #     #     # exit()
    #     # normalizing the data for training after.
    #     x_new = x_new / x_size
    #     y_new = y_new / y_size
    #     sample_new = np.stack([y_new, x_new], axis=1) # y then x because rrtstar algo do it the other way around so we need to restore
    #     return sample_new
    
    def interpolate_traj(sample, n_points):
        
        ###############
        def spline(sample):
            s = interp1d(np.arange(len(sample)), sample)

            return s
        ###############

        x = sample[:,0]
        y = sample[:,1]
        s_x = spline(x)
        s_y = spline(y)
        t_new = np.linspace(0, len(sample)-1, num=n_points)
        x_new = s_x(t_new) / x_size # normalizing the data for training after.
        y_new = s_y(t_new) / y_size # normalizing the data for training after.
        new_sample = np.stack([y_new, x_new], axis=-1)
        
        return new_sample

    for filename in tqdm(filenames):
        file = json.load(open(filename, 'r'))
        file_new = {}
        for k in sorted(file.keys()):
            sample = np.array(file[k])
            sample_new = interpolate_traj(sample, n_points)
            file_new[k] = sample_new.tolist()
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        with open(os.path.join(output_folder, os.path.split(filename)[1]), 'w') as out_path:
            json.dump(file_new, out_path)


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
        required=True
    ),
    arg_parser.add_argument(
        "--n_points",
        "-n",
        dest="n_points",
        default=10,
        type=int,
        help="The number of points a trajectory will be composed of"

    )

    args = arg_parser.parse_args()

    main(args.data_folder, args.output_folder, args.n_points)
