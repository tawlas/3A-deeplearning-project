#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import json
import logging
import os
import random
import time
import torch
import numpy as np
from tqdm import tqdm

import deep_sdf
import deep_sdf.workspace as ws

# Determining which device to use
use_cuda = torch.cuda.is_available()
if use_cuda:
    device = torch.device("cuda")
    print("Training on GPU")
else:
    device = torch.device("cpu")
    print("Training on CPU")


def reconstruct(
    decoder,
    num_iterations,
    latent_size,
    test_sdf,
    stat,
    clamp_dist,
    num_samples=4096,
    lr=5e-4,
    l2reg=False,
):
    def adjust_learning_rate(
        initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every
    ):
        lr = initial_lr * ((1 / decreased_by) **
                           (num_iterations // adjust_lr_every))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    decreased_by = 10
    adjust_lr_every = int(num_iterations / 2)

    if type(stat) == type(0.1):
        latent = torch.ones(1, latent_size).normal_(
            mean=0, std=stat).to(device)
    else:
        latent = torch.normal(stat[0].detach(), stat[1].detach()).to(device)

    latent.requires_grad = True

    optimizer = torch.optim.Adam([latent], lr=lr)

    loss_num = 0
    loss_l1 = torch.nn.L1Loss()

    for e in range(num_iterations):

        decoder.eval()
        sdf_data = deep_sdf.data.create_single_sample_position(
            test_sdf).float().to(device)

        xy = sdf_data[:, 0:2]
        sdf_gt = sdf_data[:, 2].unsqueeze(1)

        adjust_learning_rate(lr, optimizer, e, decreased_by, adjust_lr_every)

        optimizer.zero_grad()

        latent_inputs = latent.expand(num_samples, -1)

        inputs = torch.cat([latent_inputs, xy], 1).to(device)

        pred_sdf = decoder(inputs)

        # TODO: why is this needed?
        if e == 0:
            pred_sdf = decoder(inputs)

        loss = loss_l1(pred_sdf, sdf_gt)
        if l2reg:
            loss += 1e-4 * torch.mean(latent.pow(2))
        loss.backward()
        optimizer.step()

        if e % 50 == 0:
            logging.debug(loss.cpu().data.numpy())
            logging.debug(e)
            logging.debug(latent.norm())
        loss_num = loss.cpu().data.numpy()

    return loss_num, latent, pred_sdf


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Use a trained DeepSDF decoder to reconstruct a shape given SDF "
        + "samples."
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint weights to use. This can be a number indicated an epoch "
        + "or 'latest' for the latest weights (this is the default)",
    )

    arg_parser.add_argument(
        "--iters",
        dest="iterations",
        default=800,
        help="The number of iterations of latent code optimization to perform.",
    )
    arg_parser.add_argument(
        "--skip",
        dest="skip",
        action="store_true",
        help="Skip meshes which have already been reconstructed.",
    )
    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    def empirical_stat(latent_vecs, indices):
        lat_mat = torch.zeros(0).to(device)
        for ind in indices:
            lat_mat = torch.cat([lat_mat, latent_vecs[ind]], 0)
        mean = torch.mean(lat_mat, 0)
        var = torch.var(lat_mat, 0)
        return mean, var

    specs_filename = os.path.join(args.experiment_directory, "specs.json")

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    latent_size = specs["CodeLength"]

    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"]).to(device)
    
    decoder = torch.nn.DataParallel(decoder, device_ids=[1])

    saved_model_state = torch.load(
        os.path.join(
            args.experiment_directory, ws.model_params_subdir, args.checkpoint + ".pth"
        ), map_location=torch.device('cpu')
    )
    saved_model_epoch = saved_model_state["epoch"]

    decoder.load_state_dict(saved_model_state["model_state_dict"])

    decoder = decoder.module.to(device)
    # decoder = decoder.to(device)

    # Getting sdf filenames
    # fetches a list of sdf filenames. one per image
    data_source = specs['eval']['DataSource']
    sdf_filenames = sorted(deep_sdf.data.get_instance_filenames(data_source))
    # print(len(sdf_filenames))

    logging.debug(decoder)

    err_sum = 0.0
    repeat = 1
    rerun = 0

    reconstruction_dir = os.path.join(
        args.experiment_directory, ws.reconstructions_subdir \
        # , str(saved_model_epoch)
    )

    if not os.path.isdir(reconstruction_dir):
        os.makedirs(reconstruction_dir)

    reconstruction_sdf_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_sdf_subdir
    )
    if not os.path.isdir(reconstruction_sdf_dir):
        os.makedirs(reconstruction_sdf_dir)

    reconstruction_codes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_codes_subdir
    )
    if not os.path.isdir(reconstruction_codes_dir):
        os.makedirs(reconstruction_codes_dir)
    for ii, npy in enumerate(tqdm(sdf_filenames)):

        if "npy" not in npy:
            continue

        full_filename = npy

        logging.debug("loading {}".format(npy))

        for k in range(repeat):

            if rerun > 1:
                sdf_filename = os.path.join(
                    reconstruction_sdf_dir, npy[:-4] + "-" + str(k + rerun)
                )
                latent_filename = os.path.join(
                    reconstruction_codes_dir, npy[:-4] +
                    "-" + str(k + rerun) + ".pth"
                )
            else:
                # sdf_filename = os.path.join(
                #     reconstruction_sdf_dir, os.path.split(npy)[1])
                # latent_filename = os.path.join(
                #     reconstruction_codes_dir, os.path.split(npy)[1])
                pred_dir = specs['eval']['PredDir']
                sdf_filename = os.path.join(
                    pred_dir,"sdf", os.path.split(npy)[1])
                latent_filename = os.path.join(
                    pred_dir,"codes", os.path.split(npy)[1]
                )

            if (
                args.skip
                and os.path.isfile(latent_filename)
            ):
                continue

            logging.info("reconstructing {}".format(npy))

            start = time.time()
            err, latent, pred_sdf = reconstruct(
                decoder,
                int(args.iterations),
                latent_size,
                full_filename,
                0.01,  # [emp_mean,emp_var],
                0.1,
                num_samples=4096,
                lr=5e-3,
                l2reg=True,
            )
            err_sum += err
            logging.debug("current_error avg: {}".format((err_sum / (ii + 1))))
            logging.debug(ii)

            logging.debug("latent: {}".format(latent.detach().cpu().numpy()))

            decoder.eval()

            if not os.path.exists(os.path.dirname(sdf_filename)):
                os.makedirs(os.path.dirname(sdf_filename))

            if not os.path.exists(os.path.dirname(latent_filename)):
                os.makedirs(os.path.dirname(latent_filename))

            np.save(sdf_filename, pred_sdf.detach().cpu().numpy())
            np.save(latent_filename, latent.detach().cpu().numpy())
            # torch.save(latent.unsqueeze(0), latent_filename)
