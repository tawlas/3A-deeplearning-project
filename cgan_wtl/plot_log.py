# import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import utils.workspace as ws


def load_logs(experiment_directory, type):
    specs = ws.load_experiment_specifications(experiment_directory)
    if type == "g":
        sub_dir = "generator"
    elif type == "d":
        sub_dir = "discriminator"
    logs = torch.load(os.path.join(specs[sub_dir]["LogsDir"], ws.logs_filename))

    print("latest epoch is {}".format(logs["epoch"]))

    num_epoch = len(logs["loss"])

    fig, ax = plt.subplots()

    if type == "g":
    
        ax.plot(
        np.arange(1, num_epoch+1),
        logs["loss"],
        "#000000",
        label="Generator Training Loss"
        )
        
        ax.plot(
            np.arange(1, num_epoch+1),
            logs["val_loss"],
            "#999999",
            label="Generator Validation Loss"
        )
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)

        ax.set(xlabel="Epoch", ylabel="Loss", title="Generator　Training and Validation Losses")

    elif type == "d":
        
        ax.plot(
        np.arange(1, num_epoch+1),
        logs["loss"],
        "#000000",
        label="Discriminator Training Loss"
        )
        
        ax.plot(
            np.arange(1, num_epoch+1),
            logs["val_loss"],
            "#999999",
            label="Discriminator Validation Loss"
        )
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)

        ax.set(xlabel="Epoch", ylabel="Loss", title=" Discriminator Training and Validation Losses")


    else:
        raise Exception('unrecognized plot type "{}"'.format(type))

    ax.grid()
    plt.show()


if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="Plot DeepSDF training logs")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include experiment "
        + "specifications in 'specs.json', and logging will be done in this directory "
    )
    arg_parser.add_argument("--type", "-t", dest="type", default="g")


    args = arg_parser.parse_args()

    load_logs(args.experiment_directory, args.type)
