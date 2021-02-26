# import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import utils.workspace as ws


def load_logs(experiment_directory, type):
    specs = ws.load_experiment_specifications(experiment_directory)
    logs = torch.load(os.path.join(specs["LogsDir"], ws.logs_filename))

    print("latest epoch is {}".format(logs["epoch"]))

    num_epoch = len(logs["val_loss"])

    fig, ax = plt.subplots()

    if type == "loss":
    
        ax.plot(
        np.arange(1, num_epoch+1),
        logs["loss"],
        "#000000",
        label="Training Loss"
        )
        
        ax.plot(
            np.arange(1, num_epoch+1),
            logs["val_loss"],
            "#999999",
            label="Validation Loss"
        )
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)

        ax.set(xlabel="Epoch", ylabel="Loss", title="Training and Validation Losses")

    elif type == "precision":
        ax.plot(
        np.arange(1, num_epoch+1),
        logs["precision"],
        "#000000",
        label="Training Precision"
        )
        
        ax.plot(
            np.arange(1, num_epoch+1),
            logs["val_precision"],
            "#999999",
            label="Validation Precision"
        )
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)

        ax.set(xlabel="Epoch", ylabel="Precision", title="Training and Validation Precisions")

    elif type == "recall":
        ax.plot(
        np.arange(1, num_epoch+1),
        logs["recall"],
        "#000000",
        label="Training Recall"
        )
        
        ax.plot(
            np.arange(1, num_epoch+1),
            logs["val_recall"],
            "#999999",
            label="Validation Recall"
        )
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)

        ax.set(xlabel="Epoch", ylabel="Recall", title="Training and Validation Recalls")

    elif type == "accuracy":
        ax.plot(
        np.arange(1, num_epoch+1),
        logs["accuracy"],
        "#000000",
        label="Training Accuracy"
        )
        
        ax.plot(
            np.arange(1, num_epoch+1),
            logs["val_accuracy"],
            "#999999",
            label="Validation Accuracy"
        )
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)

        ax.set(xlabel="Epoch", ylabel="accuracy", title="Training and Validation Accuracies")

    elif type == "roc_auc_score":
        ax.plot(
        np.arange(1, num_epoch+1),
        logs["roc_auc_score"],
        "#000000",
        label="Training roc_auc_score"
        )
        
        ax.plot(
            np.arange(1, num_epoch+1),
            logs["val_roc_auc_score"],
            "#999999",
            label="Validation roc_auc_score"
        )
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)

        ax.set(xlabel="Epoch", ylabel="roc_auc_score", title="Training and Validation ROC AUCs")

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
    arg_parser.add_argument("--type", "-t", dest="type", default="loss")


    args = arg_parser.parse_args()

    load_logs(args.experiment_directory, args.type)
