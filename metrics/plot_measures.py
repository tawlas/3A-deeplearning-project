import numpy as np
import os
import json
from glob import glob
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import h5py
from numpy.linalg import norm
import pandas as pd
import seaborn as sns

if __name__ == "__main__":
    #Loading results files
    ## cgan
    cgan_path = "../cgan/data/test_results_cgan.h5"
    hf_cgan = h5py.File(cgan_path, 'r')
    ## cgan length constrained
    cgan_c_path = "../cgan_c/data/test_results_cgan_c.h5"
    hf_cgan_c = h5py.File(cgan_c_path, 'r')
    
    ## cgan_wtl
    cgan_wtl_path = "../cgan_wtl/data/test_results_cgan_wtl.h5"
    hf_cgan_wtl = h5py.File(cgan_wtl_path, 'r')
    ## generator_fc_single
    generator_fc_single_path = "../generator_fc_single/data/test_results_generator_fc_single.h5"
    hf_generator_fc_single = h5py.File(generator_fc_single_path, 'r')
    # lsq chomp
    lsqchomp_path = "../cgan_c/lsq_chomp.h5"
    hf_lsqchomp = h5py.File(lsqchomp_path, 'r')





    # success rate
    sr_lsqchomp = np.array(hf_lsqchomp["success_rate"])
    sr_cgan = np.array(hf_cgan["success_rate"])
    sr_cgan_c = np.array(hf_cgan_c["success_rate"])
    sr_cgan_wtl = np.array(hf_cgan_wtl["success_rate"])
    sr_generator_fc_single = np.array(hf_generator_fc_single["success_rate"])

    # Mean over all trajectories in all environments to plot bar chart 
    sr_lsqchomp_mean = np.mean(sr_lsqchomp)
    sr_cgan_mean = np.mean(sr_cgan)
    sr_cgan_c_mean = np.mean(sr_cgan_c)
    sr_cgan_wtl_mean = np.mean(sr_cgan_wtl)
    sr_generator_fc_single_mean = np.mean(sr_generator_fc_single)

    #path cost
    path_cost_lsqchomp = np.array(hf_lsqchomp["path_cost"])
    path_cost_cgan = np.array(hf_cgan["path_cost"])
    path_cost_cgan_c = np.array(hf_cgan_c["path_cost"])
    path_cost_cgan_wtl = np.array(hf_cgan_wtl["path_cost"])
    path_cost_generator_fc_single = np.array(hf_generator_fc_single["path_cost"])

    # Mean over all trajectories in all environments to plot bar chart 
    path_cost_lsqchomp_mean = np.mean(path_cost_lsqchomp)
    path_cost_cgan_mean = np.mean(path_cost_cgan)
    path_cost_cgan_c_mean = np.mean(path_cost_cgan_c)
    path_cost_cgan_wtl_mean = np.mean(path_cost_cgan_wtl)
    path_cost_generator_fc_single_mean = np.mean(path_cost_generator_fc_single)


    
    #time
    # computation_time_lsqchomp = np.array(hf_lsqchomp["generation_time"])
    computation_time_cgan = np.array(hf_cgan["generation_time"])
    computation_time_cgan_c = np.array(hf_cgan_c["generation_time"])
    computation_time_cgan_wtl = np.array(hf_cgan_wtl["generation_time"])
    computation_time_generator_fc_single = np.array(hf_generator_fc_single["generation_time"])

    # Mean over all trajectories in all environments to plot bar chart 
    # computation_time_lsqchomp_mean = np.mean(computation_time_lsqchomp)
    computation_time_cgan_mean = np.mean(computation_time_cgan)
    computation_time_cgan_c_mean = np.mean(computation_time_cgan_c)
    computation_time_cgan_wtl_mean = np.mean(computation_time_cgan_wtl)
    computation_time_generator_fc_single_mean = np.mean(computation_time_generator_fc_single)


    hf_cgan.close()
    hf_cgan_c.close()
    hf_cgan_wtl.close()
    hf_generator_fc_single.close()
    hf_lsqchomp.close()
    #
    models = ["Model"] + ["PathNet"] + ["PathNet_\length_const", "PathNet_\DetNet", "PathNet_\\traj_loss", "lsq_chomp"]
    success_rates_mean = ["Mean Success Rate"] + [sr_cgan_c_mean] + [sr_cgan_mean, sr_generator_fc_single_mean, sr_cgan_wtl_mean, sr_lsqchomp_mean]  
    path_costs_mean = ["Mean Path Cost"] + [path_cost_cgan_c_mean]+ [path_cost_cgan_mean, path_cost_generator_fc_single_mean, path_cost_cgan_wtl_mean, path_cost_lsqchomp_mean]  
    computation_times_mean = ["Mean Computation Time"] + [computation_time_cgan_c_mean*1000] + [computation_time_cgan_mean*1000, computation_time_generator_fc_single_mean*1000, computation_time_cgan_wtl_mean*1000, 0]  


    data = pd.DataFrame(list(zip(models[1:], success_rates_mean[1:], path_costs_mean[1:], computation_times_mean[1:])),
                        columns=[models[0], success_rates_mean[0], path_costs_mean[0], computation_times_mean[0]],
                    index=models[1:])

    my_dpi = 200
    fig = plt.figure(figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])

    ax = sns.barplot(y= "Mean Success Rate", x = "Model", data = data, palette=("Blues_d"))
    # ax = sns.barplot(y= "Mean Path Cost", x = "Model", data = data, palette=("Blues_d"))
    # ax = sns.barplot(y= "Mean Computation Time", x = "Model", data = data, palette=("Blues_d"))
    # ax.set(ylabel='Mean Computation Time (millisec)')
    # sns.set_context("poster")


    # Matplotlib Plot Bars
    # plt.figure(figsize=(16,16), dpi= 100)
    # plt.bar(data["Model"], data["Mean Success Rate"], color="blue", width=.4)
    # plt.bar(data["Model"], data["Mean Path Cost"], color="blue", width=.4)
    # plt.ylabel('Path Cost', fontsize=24, fontweight=800)
    # plt.bar(data["Model"], data["Mean Computation Time"], color="blue", width=.4)
    # plt.ylabel('Computation Time (millisec)', fontsize=24, fontweight=800)
    # for i, val in enumerate(data["Mean Computation Time"].values):
    #     plt.text(i, val, np.round(float(val), 2), horizontalalignment='center', verticalalignment='bottom', fontdict={'fontweight':700, 'size':24})

    # # Decoration
    # plt.gca().set_xticklabels(data["Model"], rotation=-10, horizontalalignment= 'center', fontsize=24, fontweight=800)
    # # plt.gca().set_yticklabels(np.round(np.linspace(0,0., 11), 2), fontsize=20)
    # # plt.title("Su", fontsize=22)
    # plt.yticks(fontsize=24, fontweight=700)
    # plt.ylabel('# Success Rate', fontsize=24, fontweight=800)
    # plt.ylim(0, 45)
    plt.show()

    # Add patches to color the X axis labels
    # p1 = patches.Rectangle((.57, -0.005), width=.33, height=.13, alpha=.1, facecolor='green', transform=fig.transFigure)
    # p2 = patches.Rectangle((.124, -0.005), width=.446, height=.13, alpha=.1, facecolor='red', transform=fig.transFigure)
    # fig.add_artist(p1)
    # fig.add_artist(p2)
    # plt.show()




    # y_pos = np.arange(len(data["Mean Success Rate"]))
    # plt.bar(y_pos, list(data["Mean Success Rate"]))
    # ax.xticks(y_pos, data["Model"])
    # plt.show()







