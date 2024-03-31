# Script to visualize NeuralODE runs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import time
from subprocess import call
from itertools import repeat
import matplotlib as mpl
from os.path import isfile
import corner
from scipy.stats import binned_statistic_dd 

# Pretty plots and a colour-blind friendly colour scheme
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"

colors = {"blue":"#4477aa", "cyan":"#66ccee", "green":"#228833", "yellow":"#ccbb44",
          "red":"#ee6677", "purple":"#aa3377", "grey":"#bbbbbb"}
labels = ["HI", "CII", "CI", "CO"]

def parameters(model_index):
    """model_index is an integer
    """
    df = pd.read_csv("samples.csv")
    return df.iloc[model_index]

def renormalize(log_scale_params, pred_data, true_data, val_av):
    """Renormalizing data back to original log-scale abundances and visual extinctions.
    """
    data_mean, data_std, av_mean, av_std = log_scale_params
    rescaled_pred_data = pred_data*data_std[:, np.newaxis, :] + data_mean[:, np.newaxis, :]
    rescaled_true_data = true_data*data_std[:, np.newaxis, :] + data_mean[:, np.newaxis, :]
    rescaled_val_av = val_av*av_std[:, np.newaxis] + av_mean[:, np.newaxis]
    # rescaled_pred_data = pred_data*data_std + data_mean
    # rescaled_true_data = true_data*data_std + data_mean
    # rescaled_val_av = val_av*av_std + av_mean
    return rescaled_pred_data, rescaled_true_data, rescaled_val_av

def display_loss(train_loss, val_loss, savefig_path):
    """Display training and validation loss functions
    savefig_path is the full path where the plot is saved.
    """
    epochs = len(train_loss)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True)
    ax.plot(np.arange(1, epochs + 1), train_loss, color = "red", label = "Train loss")
    ax.plot(np.arange(1, epochs + 1), val_loss, color = "green", label = "Val loss")
    ax.set_yscale("log")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("MSE Loss")
    ax.set_ylabel("MSE Loss")
    ax.grid()
    fig.legend()
    fig.savefig(savefig_path, dpi = 300)
    
def display_predictions(pred, true, val_ind, fracs_array, path, percentile = False, 
                        percentile_losses = None, percentiles = None, percentile_indices = None):
    """Display predictions and true values of time series in the validation set.
    val_ind is the list of model_indices of the validation set.
    if percentile = True, only plots samples corresponding to specific loss percentiles
    """
    frac = fracs_array[-1]
    labels = ["HI", "CII", "CI", "CO"]
    if percentile:
        # array = percentile_mask.nonzero()[0]    # sample (model) indices 
        array = percentile_indices 
    else:
        array = np.arange(len(val_ind))
    # for i in tqdm(range(len(val_ind))):
    for percentile_index, i in enumerate(tqdm(array)):
        fig, ax = plt.subplots(nrows = 1, ncols = 1)
        # val_index = val_ind[i]
        val_index = val_ind[i]
        params = parameters(val_index).to_numpy()
        std_params = np.array([1e4, 1e4, 1.3e-17])
        standardized_params = params[1:]/std_params
        rounded_params = [np.round(standardized_params[j], 3) for j in range(len(params[1:]))]
        true_data, pred_data = true[i].T, pred[i].T
        for true_, pred_, label, color_key in zip(true_data, pred_data, labels, colors.keys()):
            length = len(val_av[i])
            ax.plot(val_av[i][:int(frac*length)], true_, label = label, color = colors[color_key], linestyle = "-", linewidth = 1.0)
            ax.plot(val_av[i][:int(frac*length)], pred_, color = colors[color_key], linestyle = "--", linewidth = 1.0)
        for f in fracs_array[:-1]:
            ax.axvline(x = val_av[i][int(f*length)], color = "red", linestyle = "--", linewidth = 0.5)
        ax.set_xlabel(r"$\log_{10}{A_v}$")
        ax.set_ylabel(r"$\log_{10}{X}$")
        ax.legend(loc = "best")
        if percentile:
            percentile_loss = np.round(percentile_losses[percentile_index], 3)
            percentile_ = percentiles[percentile_index]
            ax.set_title(rf"$G_{{UV}} = {rounded_params[0]}G_{{UV_{{0}}}}, n_{{H}} = {rounded_params[1]}n_{{H_{{0}}}}, \zeta_{{CR}} = {rounded_params[2]}\zeta_{{CR_{{0}}}}$"
                         + "\n"
                         + rf"{percentile_}th percentile MSE loss: {percentile_loss}")
            fig.savefig(f"{path}/percentile_plots/model_{val_index}_pred_abundances.png", dpi = 300)
        else:
            ax.set_title(rf"$G_{{UV}} = {rounded_params[0]}G_{{UV_{{0}}}}, n_{{H}} = {rounded_params[1]}n_{{H_{{0}}}}, \zeta_{{CR}} = {rounded_params[2]}\zeta_{{CR_{{0}}}}$")
            fig.savefig(f"{path}/model_{val_index}_pred_abundances.png", dpi = 300)
        plt.close()
        
def plot_epoch_prediction(true_data, pred_data, epoch, path):
    """Plots predictions for a sample for one epoch and saves it in a given path.
    """
    true_data = np.swapaxes(true_data, 0, 1)
    pred_data = np.swapaxes(pred_data, 0, 1)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10,10))
    index = np.arange(np.shape(true_data)[1])
    for true_, pred_, label, color_key in zip(true_data, pred_data, labels, colors.keys()):
        ax.plot(index, true_, label = label, color = colors[color_key], linestyle = "-")
        ax.plot(index, pred_, color = colors[color_key], linestyle = "--")
    ax.set_xlabel(r"$\overline{\log_{10}{A_v}}$")
    ax.set_ylabel(r"$\overline{\log_{10}{X}}$")
    ax.set_title(f"Model 0: Epoch = {epoch}")
    fig.savefig(f"{path}/train_predictions/epoch_{epoch}.png", dpi = 300)
    plt.close()        

def display_predictions_all(model_00_true, model_00_pred, path):
    """Display model predictions as a function of epoch to see how the model trains.
    set_of_predictions contains predictions for all epochs.
    set_of_true_timeseries contains the true timeseries for all epochs.
    path is the folder where the plots are saved.
    """
    print("Plotting predictions obtained during training...")
    epochs = np.arange(1, len(model_00_pred) + 1, 1)
    # Defining the multiprocessing pool
    p = mp.Pool(processes = 8)
    start = time.time()
    p.starmap(plot_epoch_prediction, zip(repeat(model_00_true), model_00_pred, epochs,
                                         repeat(path)))
    end = time.time()
    print("Total time: ", end - start)
    
def visualize_weights(all_weights):
    """Display weights of different layers as a function of epoch. 
    """
    for layer, weight in enumerate(all_weights, start = 1):
        epoch, l1, l2 = np.shape(weight)
        weight = np.reshape(weight, (epoch, l1*l2))
        fig, ax = plt.subplots(nrows = 1, ncols = 1)
        img = ax.imshow(weight, cmap = "viridis", aspect = "auto")
        fig.colorbar(img)
        fig.savefig(f"weights/weight_{layer}.png", dpi = 300)
        
def sample_loss(pred_data, true_data):
    """Returns an array of MSE losses for each sample
    """
    sample_loss_ = np.mean((pred_data - true_data)**2, axis = (1,2))
    return sample_loss_

def get_percentile_samples(percentiles, sample_losses):
    """Returns index mask corresponding to loss percentiles in the validation set.
    Explicitly includes 0th and 100th percentiles.
    """
    sorted_indices = np.argsort(sample_losses)
    sorted_losses = sample_losses[sorted_indices]
    percentile_losses = [sorted_losses[int(len(sorted_losses)*p)] for p in percentiles]
    percentile_losses = [sorted_losses[0]] + percentile_losses
    percentile_losses.append(sorted_losses[-1])
    mask = np.in1d(sample_losses, percentile_losses)
    percentile_indices = np.nonzero(mask.astype(int))[0]
    
    # Argsorting the mask according to the losses
    unsorted_losses = sample_losses[mask]
    sorted_percentile_indices = np.argsort(unsorted_losses)
    sorted_percentile_indices = percentile_indices[sorted_percentile_indices]
    return mask, percentile_losses, sorted_percentile_indices

def loss_hist(sample_losses, percentiles, percentile_losses, path):
    """Generates a histogram of losses along with percentiles.
    """
    sample_losses = np.sort(sample_losses)
    loss_bins = np.logspace(np.log10(1e-3), np.log10(1e1), 11)
    zorders = [5, 10, 20]
    print("Visualizing sample losses...")
    fig, ax = plt.subplots(1,1)
    ax.hist(sample_losses, bins = loss_bins, zorder = 0, histtype = "step", color = "black")
    for percentile, loss, color_key, zorder in zip(percentiles[1:-1], 
                                                   percentile_losses[1:-1], 
                                                   colors.keys(), zorders):    # avoiding 0 and 100
        ax.axvline(x = loss, color = colors[color_key], linestyle = "--", linewidth = 1.0, 
                   label = rf"{percentile}th percentile", zorder = zorder)
    ax.set_xlabel("Sample loss bins")
    ax.set_ylabel("Sample counts")
    ax.set_title("Histogram of sample losses")
    ax.set_xscale("log")
    ax.legend()
    fig.savefig(f"{path}/percentile_plots/loss_hist.png", dpi = 300)
    
def species_plots(pred_data, true_data):
    """Generates predicted vs true abundance confidence plots for each molecule.
    pred_data is the set of predicted abundance curves.
    true_data is the set of true abundance curves.
    """
    HI_true, CII_true, CI_true, CO_true = true_data[:, :, 0], true_data[:, :, 1], true_data[:, :, 2], true_data[:, :, 3]
    HI_pred, CII_pred, CI_pred, CO_pred = pred_data[:, :, 0], pred_data[:, :, 1], pred_data[:, :, 2], pred_data[:, :, 3]
    # print("Shape of HI_true: ", np.shape(HI_true))
    CII_pred = np.sort(CII_pred, axis = 1)

    
def corner_plot(df, path, sample_losses):
    """Corner plot of samples along with parameters corresponding to percentile losses highlighted.
    """
    # ndim, nsamples = 3, 1024
    # np.random.seed(42)
    # samples = np.random.randn(ndim * nsamples).reshape([nsamples, ndim])
    # fig = corner.corner(samples)
    
    samples = np.log10(df[["g_uv", "n_H", "zeta_CR"]].to_numpy())
    g_uv, n_H, zeta_CR = samples[:,0], samples[:,1], samples[:,2]
    # n_bins = int(np.sqrt(len(samples)))
    n_bins = 12     # in each dimension
    g_uv_bins = np.linspace(np.min(g_uv), np.max(g_uv), n_bins + 1)
    n_H_bins = np.linspace(np.min(n_H), np.max(n_H), n_bins + 1)
    zeta_CR_bins = np.linspace(np.min(zeta_CR), np.max(zeta_CR), n_bins + 1)
    loss_prob, bin_edges, bin_number = binned_statistic_dd(samples, sample_losses, statistic = "sum",
                                                           bins = [g_uv_bins, n_H_bins, zeta_CR_bins],
                                                           expand_binnumbers = True) 
    
    loss_prob /= np.sum(loss_prob)
    print(np.shape(loss_prob))
    
    # Converting to a 2D array for representing on a corner plot (honestly there should be a better way of doing this)
    loss_prob_2D = loss_prob.flatten()
    bin_array = np.arange(len(loss_prob_2D))    # indices of bins in flattened array
    bin_left_indices = np.swapaxes(np.unravel_index(bin_array, np.shape(loss_prob)), 0, 1)  # indices of bin left edges in 3D array
    bin_right_indices = bin_left_indices + 1
    g_uv_left_indices, n_H_left_indices, zeta_CR_left_indices = bin_left_indices[:,0], bin_left_indices[:,1], bin_left_indices[:,2]
    g_uv_right_indices, n_H_right_indices, zeta_CR_right_indices = bin_right_indices[:,0], bin_right_indices[:,1], bin_right_indices[:,2]
    g_uv_left_edges, g_uv_right_edges = np.take(g_uv_bins, g_uv_left_indices), np.take(g_uv_bins, g_uv_right_indices)
    n_H_left_edges, n_H_right_edges = np.take(n_H_bins, n_H_left_indices), np.take(n_H_bins, n_H_right_indices)
    zeta_CR_left_edges, zeta_CR_right_edges = np.take(zeta_CR_bins, zeta_CR_left_indices), np.take(zeta_CR_bins, zeta_CR_right_indices) 
    g_uv_means = 0.5*(g_uv_left_edges + g_uv_right_edges)
    n_H_means = 0.5*(n_H_left_edges + n_H_right_edges)
    zeta_CR_means = 0.5*(zeta_CR_left_edges + zeta_CR_right_edges)
    samples_2D = np.vstack((loss_prob_2D, g_uv_means, n_H_means, zeta_CR_means))
    
    # Generating a corner plot
    fig = corner.corner(samples_2D.T, labels = [r"$G_{UV}$", r"$n_{H}$", r"$\zeta_{CR}$"])
    fig.savefig(f"{path}/percentile_plots/corner_plot.png", dpi = 300)
    

if __name__ in "__main__":
    
    path_numbers = np.arange(1,6)
    paths = [f"predictions_{number}" for number in path_numbers]
    df = pd.read_csv("samples.csv")
    df = df.rename(columns = {"Unnamed: 0": "model_index"})
    
    
    for path in paths[:1]:
        
        call(f"mkdir {path}/train_predictions", shell = True)   # Folder for storing train predictions
        call(f"mkdir {path}/percentile_plots", shell = True)    # Folder for storing percentile plots
        val_ind = np.load(f"{path}/val_ind.npy")
        val_av = np.load(f"{path}/val_av.npy")
        
        loss_functions = np.loadtxt(f"{path}/loss_function.csv", delimiter = ",")
        train_loss, val_loss = loss_functions[:, 0], loss_functions[:, 1]
        pred_data = np.load(f"{path}/eval_data_pred.npy")
        true_data = np.load(f"{path}/val_data_true.npy")
        loss_av = np.load(f"{path}/loss_av.npy")
        
        data_mean = np.load(f"{path}/scale_params/val_data_log_mean.npy")
        data_std = np.load(f"{path}/scale_params/val_data_log_std.npy")
        av_mean = np.load(f"{path}/scale_params/val_av_log_mean.npy")
        av_std = np.load(f"{path}/scale_params/val_av_log_std.npy")
        # log_scale_params = np.load(f"{path}/log_scale_params.npy")
        log_scale_params = [data_mean, data_std, av_mean, av_std]
        
        # Renormalizing back to log-scale data and av
        pred_data, true_data, val_av = renormalize(log_scale_params, pred_data, true_data, val_av)
        
        # Getting sample losses and percentiles
        sample_losses = sample_loss(pred_data, true_data)
        percentiles = [0.25, 0.5, 0.75]
        percentile_mask, percentile_losses, sorted_percentile_indices = get_percentile_samples(percentiles, sample_losses)
        # species_plots(pred_data, true_data)
        
        # print("Plotting the train and validation loss functions as a function of epoch...")
        # display_loss(train_loss, val_loss, savefig_path = f"{path}/loss_function.png")
        
        # print("Making predictions on the validation set...")
        # display_predictions(pred_data, true_data, val_ind, np.array([0.2, 0.4, 0.6, 0.8, 1.0]),
        #                     path, percentile = False)
        
        # print("Plotting samples corresponding to 0th, 25th, 50th, 75th and 100th percentile losses...")
        # percentiles = 100*np.array([0.0] + percentiles + [1.0]) 
        # display_predictions(pred_data, true_data, val_ind, np.array([0.2, 0.4, 0.6, 0.8, 1.0]), 
        #                     path, percentile = True, percentile_losses = percentile_losses, 
        #                     percentiles = percentiles, percentile_indices = sorted_percentile_indices)
        
        # print("Plotting a histogram of losses...")
        # loss_hist(sample_losses, percentiles, percentile_losses, path)
        
        print("Making corner plots of sample parameters...")
        df_val = df.loc[val_ind]
        corner_plot(df_val, path, sample_losses)
        
        # Making a movie of visualizations obtained during training if it doesn't exist already
        if not isfile(f"{path}/predictions_all.mp4"):
            print("Visualizing predictions while training and making a movie...")
            # set_of_predictions = np.swapaxes(set_of_predictions, 1, 2)
            # set_of_true_timeseries = np.swapaxes(set_of_true_timeseries, 1, 2)
            model_00_pred = np.load(f"{path}/model_00_pred.npy")
            model_00_true = np.load(f"{path}/model_00_true.npy")
            display_predictions_all(model_00_true, model_00_pred, path)
            call(f"ffmpeg -framerate 8 -i {path}/train_predictions/epoch_%d.png -r 30 {path}/predictions_all.mp4", shell = True)
        
        print("Plotting final loss as a function of visual extinction...")
        fig, ax = plt.subplots(nrows = 1, ncols = 1)
        index = np.arange(len(loss_av))
        ax.plot(index, loss_av)
        ax.set_xlabel("Index (scaled $A_v$)")
        ax.set_ylabel("MSE loss")
        ax.set_title("Loss vs scaled $A_v$")
        ax.grid()
        fig.savefig(f"{path}/loss_vs_av.png", dpi = 300)
    


