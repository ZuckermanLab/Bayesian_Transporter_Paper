import numpy as np 
import matplotlib.pyplot as plt
import yaml
import arviz as az
import math
import logging 
import os


import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import arviz as az

def plot_histograms(data_files, colors, bins, burn_in=0, ranges=None, references=None, run_labels=None, titles=None, fname=None, xscale=None):
    # Compute the number of rows and columns based on the number of columns in the data.


    loaded_data = np.loadtxt(data_files[0], delimiter=',')
    if len(loaded_data.shape) == 1:
        num_subplots = 1
    else:
        num_subplots = loaded_data.shape[1]
    grid_size = math.isqrt(num_subplots)
    while num_subplots % grid_size != 0:
        grid_size -= 1
    num_rows = num_subplots // grid_size
    num_cols = grid_size
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(8*num_cols, 6*num_rows), squeeze=False)

    # Load and plot data for each file.
    for i, file_name in enumerate(data_files):
        # Load the data file into a numpy array.
        data = np.loadtxt(file_name, delimiter=',')

        if data.ndim == 1:
            axs[0, 0].hist(data[burn_in:], bins=bins, range=ranges, density=True, color=colors[i], alpha=0.5, histtype='step')
            if references is not None:
                axs[0, 0].axvline(x=references[0], color='black', linestyle='--', linewidth=1)
            if titles is not None:
                axs[0, 0].set_title(titles[0])
            axs[0, 0].set_ylabel("density")
            if xscale is not None:
                axs[0,0].set_xscale(xscale)
        else:
            # Plot each column of data as a histogram on the corresponding subplot.
            for j in range(num_subplots):
                row_idx, col_idx = np.unravel_index(j, axs.shape)
                range_j = ranges[j] if ranges is not None else None
                axs[row_idx, col_idx].hist(data[burn_in:, j], bins=bins, range=range_j, density=True, color=colors[i], alpha=0.5, histtype='step')

                # Add vertical lines for reference values to each subplot.
                if references is not None:
                    ref = references[j]
                    axs[row_idx, col_idx].axvline(x=ref, color='black', linestyle='--', linewidth=1)

                # Set labels and titles for each subplot.
                #axs[row_idx, col_idx].set_ylabel("Density")
                # Set labels and titles for each subplot.
                if col_idx == 0:
                    axs[row_idx, col_idx].set_ylabel("density")
                if titles is not None:
                    axs[row_idx, col_idx].set_title(titles[j])
                if xscale is not None:
                    axs[row_idx, col_idx].set_xscale(xscale)

    # Create custom legend handles and labels
    legend_handles = [mpatches.Patch(color=colors[i], label=run_labels[i] if run_labels is not None else data_files[i], alpha=0.5) for i in range(len(data_files))]

    # Add the custom legend to the figure
    fig.legend(handles=legend_handles, loc="upper right")

    # Remove any unused subplots.
    for j in range(num_subplots, axs.size):
        row_idx, col_idx = np.unravel_index(j, axs.shape)
        fig.delaxes(axs[row_idx, col_idx])

    plt.tight_layout()
    if fname:
        plt.savefig(fname)
        return fig, axs
    else:
        plt.show()
        return fig, axs


def calculate_ess(datafile):
    data = np.loadtxt(datafile, delimiter=',')
    inference_data = az.convert_to_inference_data(np.transpose(data))
    ess = az.ess(inference_data)
    print(az.summary)
    return ess


if __name__ == '__main__':
    
    config_fname = "/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/new/antiporter_1_1_12D_cycle1/experiment1/antiporter_1_1_12D_cycle1_experiment1_config.yaml"  # cycle 1 model

    # sample_data_files = [
    #     '/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/run_12D_1_1_antiporter_cycle1_experiment1_r6_2023-04-12-04-08-26-210610/flat_samples.csv',
    #     '/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/run_12D_1_1_antiporter_cycle1_experiment1_r7_2023-04-12-04-10-04-884693/flat_samples.csv',
    #     '/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/run_12D_1_1_antiporter_cycle3_experiment1_from_cycle1_r6_2023-04-12-04-15-05-757988/flat_samples.csv',
    #     '/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/run_12D_1_1_antiporter_cycle3_experiment1_from_cycle1_r7_2023-04-12-04-15-30-057348/flat_samples.csv',
    #     '/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/run_12D_1_1_antiporter_cycle1_experiment1_pocoMC_r6_2023-04-13-01-13-57-817754/samples.csv',
    #     '/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/run_12D_1_1_antiporter_cycle1_experiment1_pocoMC_r7_2023-04-13-01-16-26-018895/samples.csv',
    #     '/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/run_12D_1_1_antiporter_cycle3_experiment1_from_cycle1_pocoMC_r6_2023-04-13-01-20-33-578223/samples.csv',
    #     '/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/run_12D_1_1_antiporter_cycle3_experiment1_from_cycle1_pocoMC_r7_2023-04-13-01-21-16-422018/samples.csv'
    # ]

    # log_likelihood_data_files = [
    #     '/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/run_12D_1_1_antiporter_cycle1_experiment1_r6_2023-04-12-04-08-26-210610/flat_log_likelihoods.csv',
    #     '/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/run_12D_1_1_antiporter_cycle1_experiment1_r7_2023-04-12-04-10-04-884693/flat_log_likelihoods.csv',
    #     '/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/run_12D_1_1_antiporter_cycle3_experiment1_from_cycle1_r6_2023-04-12-04-15-05-757988/flat_log_likelihoods.csv',
    #     '/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/run_12D_1_1_antiporter_cycle3_experiment1_from_cycle1_r7_2023-04-12-04-15-30-057348/flat_log_likelihoods.csv',
    #     '/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/run_12D_1_1_antiporter_cycle1_experiment1_pocoMC_r6_2023-04-13-01-13-57-817754/log_likelihood.csv',
    #     '/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/run_12D_1_1_antiporter_cycle1_experiment1_pocoMC_r7_2023-04-13-01-16-26-018895/log_likelihood.csv',
    #     '/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/run_12D_1_1_antiporter_cycle3_experiment1_from_cycle1_pocoMC_r6_2023-04-13-01-20-33-578223/log_likelihood.csv',
    #     '/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/run_12D_1_1_antiporter_cycle3_experiment1_from_cycle1_pocoMC_r7_2023-04-13-01-21-16-422018/log_likelihood.csv'
    # ]

    # colors = [
    #     'orange', 'orange', 'blue', 'blue', 'red', 'red', 'green', 'green',
    # ]

    # run_labels = ['cycle1_r6_emcee','cycle1_r7_emcee','cycle3_r6_emcee','cycle3_r7_emcee', 'cycle1_r6_poco','cycle1_r7_poco','cycle3_r6_poco','cycle3_r7_poco',]

    sample_data_files = [
     '/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/run_12D_1_1_antiporter_cycle1_experiment1_pyMC_r7_2023-04-14-04-48-33-006511/chain1_samples.csv',
     '/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/run_12D_1_1_antiporter_cycle1_experiment1_pyMC_r7_2023-04-14-04-48-33-006511/chain2_samples.csv'
    ]

    log_likelihood_data_files = [
     '/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/run_12D_1_1_antiporter_cycle1_experiment1_pocoMC_extended_r7_2023-04-13-22-19-58-566986/log_likelihood.csv',
     '/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/run_12D_1_1_antiporter_cycle1_experiment1_pocoMC_extended_r6_2023-04-13-22-13-23-807662/log_likelihood.csv'
     
    ]

    colors = [
        'orange', 'blue',  ]
    
    run_labels = ['cycle1_chain1_NUTS','cycle1_chain2_NUTS', 'cycle1_r6_poco','cycle1_r7_poco',]

    fname_samples = 'sampling_hist_cycle1_NUTS.png'
    fname_like = 'likelihood_hist_cycle1_NUTS.png'

    bins = 10
    # Load the config.yaml file
    with open(config_fname, "r") as f:
        config = yaml.safe_load(f)

    p_nom = [d['nominal'] for d in config['bayesian_inference']['parameters']]
    p_names = [d['name'] for d in config['bayesian_inference']['parameters']]
    p_lb = [d['bounds'][0] for d in config['bayesian_inference']['parameters']]
    p_ub = [d['bounds'][1] for d in config['bayesian_inference']['parameters']]
    p_bounds = list(zip(p_lb,p_ub))

    fig, axs = plot_histograms(sample_data_files, colors, bins, burn_in=int(0), run_labels=run_labels, references=p_nom, titles=p_names, fname=fname_samples)

    # likelihood_max = np.max([np.max(np.loadtxt(log_likelihood_data_files[i], delimiter=',')) for i in range(len(log_likelihood_data_files))])
    # print(likelihood_max)
    # likelihood_bounds = (likelihood_max*0.99, likelihood_max)
    # fig2, axs2 = plot_histograms(log_likelihood_data_files, colors, bins, ranges=likelihood_bounds,  titles=['log-likelihood distribution'], run_labels=run_labels, fname=fname_like, xscale='log')
