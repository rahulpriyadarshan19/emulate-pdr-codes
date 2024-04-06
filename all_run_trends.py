# General imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.stats import norm
import argparse
import math
import matplotlib as mpl

# For timing and progress
from tqdm import tqdm

# Pretty plots in matplotlib
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
colors = {"blue":"#4477aa", "cyan":"#66ccee", "green":"#228833", "yellow":"#ccbb44",
          "red":"#ee6677", "purple":"#aa3377", "grey":"#bbbbbb"}
fill_colors = {"pale_blue": '#bbccee', "pale_cyan": '#cceeff', "pale_green": '#ccddaa', 
               "pale_yellow": '#eeeebb', "pale_red": '#ffcccc', "pale_grey": '#dddddd'}
# np.seterr(all = 'raise')

# Function to return the input parameters of a given model
def parameters(model_index):
    """model_index is an integer
    """
    df = pd.read_csv("samples.csv")
    return df.iloc[model_index]
    
# Function to restrict data within bounds
def apply_bounds(data_array, lower_bound, upper_bound):
    """
    data_array is a general array of floats on which bounds need to be applied.
    """
    greater_than_mask = data_array >= lower_bound
    less_than_mask = data_array < upper_bound
    restricted_data = data_array * greater_than_mask.astype("int") * less_than_mask.astype("int")
    # data_temp = data_array[data_array >= lower_bound]
    # restricted_data = data_temp[data_temp < upper_bound]
    return restricted_data

# Function to find confidence intervals about the median
def confidence_interval(array, frac):
    """
    array is the data array, and frac is the required fraction (i.e. 0.68 corresponds 
    to 68% of all values lying inside the returned boundaries)
    """
    sorted_array = np.sort(array)
    return sorted_array[int(0.5*len(array)*(1 - frac))], sorted_array[int(0.5*len(array)*(1 + frac))]

# Function to find nearest value to a given number in an array
def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]
    
# Function to return three specific models with varying parameter ranges - mainly for plotting
def return_models(df, param):
    """df: dataframes of samples with initial conditions
       param: one of g_uv, n_H, zeta_CR
       bounds: lower and upper bounds (float64)
    """
    model_indices = []
    order_of_mag_values = {
        "g_uv": np.array([1e0, 1e2, 1e5]),
        "n_H": np.array([1e2, 1e5, 1e7]),
        "zeta_CR": np.array([1e-17, 1e-16, 1e-15])
        }
    rounded_values = order_of_mag_values[param] # array corresponding to parameter
    df_param = df[param]
    for rounded_value in rounded_values:
        rounded_value_array = rounded_value*np.ones(len(df_param),)
        param_value = df_param.iloc[np.argmin(np.abs(df_param - rounded_value_array))]
        model_index = df[df[param] == param_value].index.item()
        model_indices.append(model_index)
    return model_indices

# Function to return a histogram provided input data
def hist_data(data_array, bin_width_log):
    """
    data_array is the array corresponding to the variation of a parameter across all runs.
    bin_width_log: bin width of log histogram
    """
    # Finding minimum and maximum values of the array to get bin edges
    array_min, array_max = np.min(data_array), np.max(data_array)
    bin_min = np.floor(array_min)
    bin_max = np.floor(array_max)
    
    # Defining the bins and generating the histogram
    log_bins = np.arange(bin_min, bin_max + bin_width_log, bin_width_log)
    counts, bins = np.histogram(data_array, bins = 10**log_bins)
    bin_centers = 0.5*(bins[1:] + bins[:-1])
    bin_widths = bins[:-1] - bins[1:]
    return counts, bins, bin_centers    
    
# Function to plot histograms of different quantities across all runs to give a sense of their order of magnitude
def qty_hist(xlabels, shape, figsize, *all_data):
    """
    all_data is the variation of a specific quantity across Av for all runs
    shape: (number_of_runs, number_of_values_for_each_run)
    """
    fig, ax_array = plt.subplots(nrows = shape[0], ncols = shape[1], figsize = figsize)
    for i in tqdm(range(len(all_data))):
        
        # Choosing the right subplot
        if i%2 == 0:
            a = int(i//2)
            b = 1
        else:
            a = int(np.floor(i/2))
            b = 0
        ax_hist = ax_array[a, b]
        array = all_data[i]
        color = list(colors.keys())[i]
        
        # Binning the data and generating a histogram
        counts, bins, bin_centers = hist_data(array, bin_width_log = 1)
        
        # Plotting the histogram
        ax_hist.bar(bin_centers, height = counts/len(array), width = np.diff(bins),
                    color = color)
        ax_hist.set_xlabel(xlabels[i])
        ax_hist.set_ylabel("Counts")
        ax_hist.set_xscale("log")
        ax_hist.grid()
        
    fig.tight_layout()
    fig.savefig("all_runs/histograms.png", dpi = 1000)
    
# Function to return confidence intervals of abundances for a given species
# using information from all runs
def abund_conf_intervals(labels, df, shape, figsize, bin_width_log, show_param_trends,
                         index_range, path, param, all_data):
    """
    labels: Labels of species
    df: dataframe of samples in the validation set
    shape: (number_of_runs, number_of_values_for_each_run)
    figsize: figsize
    bin_width_log: size of bins in log space
    show_param_trends: if True, some specific models corresponding to a chosen parameter
    are also plotted along with the confidence intervals
    index_range: range of (time) Av-series
    path: path where the figure is saved
    param: taken as an argument if show_param_trends = True, represents parameter 
    based on which confidence plots are obtained 
    all_data: the variation of a specific quantity across Av for all runs
    """
    if shape[1] == 1:
        fig, ax_array = plt.subplots(nrows = shape[0], ncols = 1, sharex = True, 
                                     figsize = figsize)
    else:
        fig, ax_array = plt.subplots(nrows = shape[0], ncols = shape[1], figsize = figsize)
    av_array = all_data[0]  # (164, 199)
                
    if show_param_trends is True:
        linestyles = ["--", "-.", ":"]
        std_values = {"g_uv": 1e4, "n_H": 1e4, "zeta_CR": 1.3e-17}
        param_index = {"g_uv": 0, "n_H": 1, "zeta_CR": -1}
        model_indices = return_models(df, param)
        std_value = std_values[param]
        for model_index, linestyle in zip(model_indices, linestyles):
            param_value = parameters(model_index)[param]
            rounded_param = np.round(param_value/std_value, 3)
            model_data = np.squeeze(true_data[df_val.index == model_index], axis = 0)
            model_av = np.squeeze(val_av[df_val.index == model_index], axis = 0)
            plot_labels = {"g_uv": rf"$G_{{UV}} = {rounded_param}G_{{UV_{{0}}}}$",
                           "n_H": rf"$n_{{H}} = {rounded_param}n_{{H_{{0}}}}$",
                           "zeta_CR": rf"$\zeta_{{CR}} = {rounded_param}\zeta_{{CR_{{0}}}}$"}
             
            for i, (model_data_, label) in enumerate(zip(model_data, labels)):
                if i == 0:
                    ax_array[i].plot(model_av, model_data_, label = plot_labels[param],
                                     linestyle = linestyle, color = "black", zorder = 3)
                    ax_array[i].set_ylabel(label)
                else:
                    ax_array[i].plot(model_av, model_data_, linestyle, color = "black",
                                     zorder = 3)
                    ax_array[i].set_ylabel(label)
                ax_array[i].grid()
            
            # av, tgas, tdust, HI, H2, CII, CI, CO = read_pdr_file(
            #     f"all_runs/{model_string}/{model_string}.pdr.fin",
            #     start_index = index_range[0], end_index = index_range[1])
            # ax_array[0].plot(av, HI, label = plot_labels[param],
            #                  linestyle = linestyle, color = "black", zorder = 3)
            # ax_array[1].plot(av, H2, linestyle = linestyle, color = "black", zorder = 3)
            # ax_array[2].plot(av, CII, linestyle = linestyle, color = "black", zorder = 3)
            # ax_array[3].plot(av, CI, linestyle = linestyle, color = "black", zorder = 3)
            # ax_array[4].plot(av, CO, linestyle = linestyle, color = "black", zorder = 3)
        ax_array[0].legend(loc = "upper right", borderpad = 0.5, fontsize = 10)
        ax_array[0].set_xlabel("$A_v$")
    
    for i in tqdm(range(1, len(all_data))):
    
        lower_bounds_68 = []
        upper_bounds_68 = []
        lower_bounds_95 = []
        upper_bounds_95 = []
        medians = []
        
        # Choosing the right subplot
        if shape[1] == 1:
            ax_hist = ax_array[i - 1]
        else:
            if i%2 == 0:
                a = int(i//2)
                b = 1
            else:
                a = int(np.floor(i/2))
                b = 0
            ax_hist = ax_array[a, b]
        array = (all_data[i])[:, 1:]    # (164, 199)
        color = list(colors.keys())[i]
        fill_color = list(fill_colors.keys())[i]
        
        # Binning the data and generating a histogram
        counts, bins, bin_centers = hist_data(av_array, bin_width_log = bin_width_log)

        # Obtaining and 68% and 95% confidence interval arrays
        for j in range(len(bins) - 1):
            av_bounded = apply_bounds(av_array, bins[j], bins[j+1])
            array_bounded = array[av_array == av_bounded]
            print(array_bounded)
            lower_bound_68, upper_bound_68 = confidence_interval(array_bounded, 0.68)
            lower_bound_95, upper_bound_95 = confidence_interval(array_bounded, 0.95)
            lower_bounds_68.append(lower_bound_68)
            upper_bounds_68.append(upper_bound_68)
            lower_bounds_95.append(lower_bound_95)
            upper_bounds_95.append(upper_bound_95)
            medians.append(np.median(array_bounded))
            
        lower_bounds_68 = np.asarray(lower_bounds_68)
        upper_bounds_68 = np.asarray(upper_bounds_68)
        lower_bounds_95 = np.asarray(lower_bounds_95)
        upper_bounds_95 = np.asarray(upper_bounds_95)
        medians = np.asarray(medians)
            
        # Plotting the median and confidence limits for all bins
        ax_hist.step(x = bin_centers, y = medians, color = "black", where = "mid")
        ax_hist.step(x = bin_centers, y = lower_bounds_68, where = "mid", color = color)
        ax_hist.step(x = bin_centers, y = upper_bounds_68, where = "mid", color = color)
        ax_hist.step(x = bin_centers, y = lower_bounds_95, where = "mid", color = color)
        ax_hist.step(x = bin_centers, y = upper_bounds_95, where = "mid", color = color)
        ax_hist.fill_between(bin_centers, lower_bounds_68, upper_bounds_68, alpha = 0.6, 
                             step = "mid", color = color,
                             where = upper_bounds_68 > lower_bounds_68)
        ax_hist.fill_between(bin_centers, lower_bounds_95, upper_bounds_95, alpha = 0.3, 
                             step = "mid", color = color,
                             where = upper_bounds_95 > lower_bounds_95)
        ax_hist.set_ylabel(labels[i])
        ax_hist.set_xscale("log")
        # ax_hist.set_yscale("log")
        ax_hist.grid()
        
    ax_hist.set_xlabel("$A_v$")
    ax_hist.set_xlim(1e-8, 1e4)
    fig.tight_layout()
    fig.savefig(f"{path}/abundance_confidence_{param}.png", dpi = 300)
        
    
# Function to visualize the distribution of Av values for each order of magnitude
def av_dbn(av_all):
    fig, ax_av = plt.subplots(nrows = 1, ncols = 1, figsize = (6,6))
    _, bins, bin_centers = hist_data(av_all)
    av_spread = []
    av_mean = []
    inverse_bins = 1/bins
    for i in range(len(bins) - 1):
        av_order_of_mag = apply_bounds(av_all, bins[i], bins[i+1])
        av_order_of_mag *= inverse_bins[i]
        av_spread.append(np.std(av_order_of_mag))
        av_mean.append(np.mean(av_order_of_mag))
    ax_av.bar(x = bin_centers, height = av_mean, yerr = av_spread, width = np.diff(bins), 
              color = "#4477aa")
    ax_av.set_xscale("log")
    ax_av.set_xlabel("$A_v$ bin")
    ax_av.set_ylabel("Mean/std $A_v$")
    fig.savefig("all_runs/av_value_dbn.png", dpi = 1000)
        
if __name__ in "__main__":
    
    # Defining the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("start_index", type = int, help = "Start index of timeseries")
    parser.add_argument("end_index", type = int, help = "End index of timeseries")
    parser.add_argument("parameter", type = str, help = "Visualizing variation of said parameter")
    args = parser.parse_args()
    index_range = np.array([args.start_index, args.end_index])
    param = args.parameter
    
    path_numbers = np.arange(1,6)
    paths = [f"predictions_{number}" for number in path_numbers]
    df = pd.read_csv("samples.csv")
    df = df.rename(columns = {"Unnamed: 0": "model_index"})
    
    for path in paths[:1]:
    
        # Loading some files
        pred_data = np.load(f"{path}/eval_data_pred.npy")
        true_data = np.load(f"{path}/val_data_true.npy")
        
        val_ind = np.load(f"{path}/val_ind.npy")
        df_val = df.loc[val_ind]
        val_av = np.load(f"{path}/val_av.npy")
        
        pred_data = np.swapaxes(pred_data, 1, 2)
        true_data = np.swapaxes(true_data, 1, 2)
        main_array = true_data
         
        HI_all = main_array[:, 0, :]
        H2_all = main_array[:, 1, :]
        CII_all = main_array[:, 2, :]
        CI_all = main_array[:, 3, :] 
        CO_all = main_array[:, 4, :]
        
        av_all_without_0 = val_av[:, 1:]
        all_data = [av_all_without_0, HI_all, H2_all, CII_all, CI_all, CO_all]
        # # qty_hist(av_all_without_0)
        # # qty_hist(["$A_v$", "[HI]", "[H2]", "[CII]", "[CI]", "[CO]"], (3,2), (15,7), 
        # #           av_all_without_0, HI_all, H2_all, CII_all, CI_all, CO_all)
        
        # # Histogram of Av along with error bars to denote spread in Av
        # # av_dbn(av_all_without_0)
        
        # Confidence intervals of abundances based on binning Av values
        bin_width_log = 0.2
        abund_conf_intervals(labels = ["[HI]", "[H2]", "[CII]", "[CI]", "[CO]"], 
                             df = df_val, 
                             shape = (5,1), 
                             figsize = (10,10), 
                             bin_width_log = bin_width_log, 
                             show_param_trends = True, 
                             index_range = index_range, 
                             path = path,
                             param = param, 
                             all_data = all_data)
