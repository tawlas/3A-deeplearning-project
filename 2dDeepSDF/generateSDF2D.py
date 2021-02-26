import numpy as np
import cv2
from scipy.ndimage.morphology import distance_transform_edt as bwdist
import matplotlib.pyplot as plt
import glob
import argparse

import os

def main(imgs_dir, output_dir, single):
    path = imgs_dir
    pathSave = output_dir
    os.makedirs(pathSave, exist_ok=True)

    files = sorted(glob.glob(path + '/*.jpg'))
    if single:
        sdf_all = []
    
    num = 0
    for file in files:

        I = cv2.imread(file)
        I = cv2.resize(I,(64,64))
        bw = I[:,:,0]/255

        #bw = 1-bw
        D1 = bwdist(bw)
        D2 = -bwdist(1 - bw)
        D = -(D1 + D2)

        # plt.imshow(D)
        # plt.show()

        name = str(num).zfill(6)
        filename = pathSave + '/' + name + '.npy'
        if single:
            sdf_all.append(D)
        else:
            np.save(filename, D)
            # cv2.imwrite(filename, D)

        num = num + 1
    if single:
        sdf_all = np.array(sdf_all)
        out_path = os.path.join(pathSave, "sdf_labels.npy")
        np.save(out_path, sdf_all)
        print("Saved a single npy file of {} sdf maps at {}".format(len(sdf_all), out_path))
    else:
        print("Saved {} npy sdf maps in {}".format(num, output_dir))

if __name__ == "__main__":
    
    arg_parser = argparse.ArgumentParser(
        description="Use a trained DeepSDF decoder to reconstruct a shape given SDF "
        + "samples."
    )
    arg_parser.add_argument(
        "--images",
        "-i",
        dest="imgs_dir",
        required=True,
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--output",
        "-o",
        dest="output_dir",
        required=True,
        help="The Path where to save the sdf maps"
    )
    arg_parser.add_argument(
        "--single",
        "-s",
        action='store_true',
        help="If false, saves a npy file per image else saves all sdf maps for all images in a single npy file. Use single for training time and not use for eval time. See training and inference codes for further details"
    )

    args = arg_parser.parse_args()
    main(args.imgs_dir, args.output_dir, args.single)