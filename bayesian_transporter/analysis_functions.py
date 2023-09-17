from sklearn.mixture import GaussianMixture
import numpy as np
from scipy.stats import uniform
import matplotlib.pyplot as plt
import math


def estimate_multivariate_density_w_GMM(samples, name, k_max=30, verbose=False, plot=False):
    """
    Estimate a multivariate density using a Gaussian Mixture Model (GMM). 
    Select the optimal number of components based on the minumum Akaike Information Criterion (AIC) 
    and Bayesian Information Criterion (BIC).

    Args:
        samples (array-like): Input data samples. Shape (n_samples, n_features) where each row is a sample.
        name (str): Name of the dataset or method for labeling figures.
        k_max (int, optional): Maximum number of components for GMM. Defaults to 30.
        verbose (bool, optional): If True, prints out the AIC and BIC for the best models. Defaults to False.
        plot (bool, optional): If True, plots AIC and BIC values against the number of components and saves the figure. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - gmm_best_aic (GaussianMixture): Best GMM model based on AIC.
            - gmm_best_bic (GaussianMixture): Best GMM model based on BIC.

    Notes:
        This function fits a GMM to the data for each number of components up to `k_max`.
        The optimal number of components is chosen as the one that minimizes the AIC or BIC.
    """
    
    n_components = range(1, k_max)  # Change range as necessary
    aics = []
    bics = []
    gmms = []

    for n in n_components:
        gmm = GaussianMixture(n_components=n, max_iter=1000)
        gmm.fit(samples)
        aics.append(gmm.aic(samples))
        bics.append(gmm.bic(samples))
        gmms.append(gmm)

    gmm_best_bic_idx = np.argmin(bics)
    gmm_best_bic = gmms[gmm_best_bic_idx]
    gmm_best_aic_idx = np.argmin(aics)
    gmm_best_aic = gmms[gmm_best_aic_idx]

    if plot:
        plt.figure(figsize=(8, 6))
        plt.plot(n_components, aics, label='AIC')
        plt.plot(n_components, bics, label='BIC')
        plt.xlabel('Number of Components')
        plt.ylabel('Information Criterion')
        plt.plot(gmm_best_aic_idx+1, aics[gmm_best_aic_idx], 'o', label='AIC min')
        plt.plot(gmm_best_bic_idx+1, bics[gmm_best_bic_idx], 'o', label='BIC min')
        plt.legend()
        plt.title(f'{name}')
        plt.savefig(f'{name}.png')
    if verbose:
        print(f"AIC min: k={gmm_best_aic_idx+1}, AIC={aics[gmm_best_aic_idx]}")
        print(f"BIC min: k={gmm_best_bic_idx+1}, AIC={bics[gmm_best_bic_idx]}")
    return gmm_best_aic, gmm_best_bic


def kl_divergence_gmm_uniform(gmm, unfiform_prior_bounds, name, n_samples=10**6, verbose=False):
    """
    Compute the Kullback-Leibler (KL) divergence between a Gaussian Mixture Model (GMM) and a 
    uniform distribution using Monte Carlo sampling.

    Args:
        gmm (GaussianMixture): The fitted Gaussian Mixture Model.
        unfiform_prior_bounds (list of tuple): List of (min, max) bounds for each dimension of the uniform distribution.
        name (str): Label for the GMM dataset for verbose output.
        n_samples (int, optional): Number of samples to draw from the GMM for estimating the KL divergence. Defaults to 10**6.
        verbose (bool, optional): If True, prints out the estimated KL divergence. Defaults to False.

    Returns:
        float: Estimated KL divergence between the GMM and the uniform distribution.

    Notes:
        The function estimates the KL divergence by drawing samples from the GMM, computing the 
        log pdfs of both the GMM and the uniform distribution at these sample points, and then 
        computing the mean log pdf ratio.
        
        Samples outside the specified bounds for the uniform distribution are discarded.
    """

    samples = gmm.sample(n_samples)[0]
    valid_samples = np.all([(samples[:, i] >= r[0]) & (samples[:, i] <= r[1]) for i, r in enumerate(unfiform_prior_bounds)], axis=0)
    samples = samples[valid_samples]
    log_gmm_pdf = gmm.score_samples(samples)
    log_uniform_pdf = np.sum([uniform.logpdf(samples[:, i], loc=r[0], scale=r[1]-r[0]) for i, r in enumerate(unfiform_prior_bounds)], axis=0)
    kl_divergence = np.mean(log_gmm_pdf - log_uniform_pdf)
    if verbose:
        print(f"KL divergence of {name} = {kl_divergence}")
    return kl_divergence


def plot_1D_distributions(sample_arrays, sample_labels, parameter_names, parameter_ranges, parameter_nominals, bins=100, title="1D Parameter Distribution"):
    """
    Plots 1D parameter distributions with overlay of different samples and returns the figure object.

    Args:
        sample_arrays (list[np.ndarray]): List of 2D sample arrays to be plotted, where each row represents a sample and each column represents a parameter.
        sample_labels (list[str]): List of labels corresponding to each sample array.
        parameter_names (list[str]): List of parameter names.
        parameter_ranges (list[tuple]): List of (min, max) ranges for each parameter.
        parameter_nominals (list[float]): List of nominal values for each parameter.
        title (str, optional): Title of the plot. Default is "1D Parameter Distribution".

    Returns:
        matplotlib.figure.Figure: The figure object containing the plotted distributions.

    Raises:
        AssertionError: If the number of sample arrays does not match the number of labels.
    """

    # Ensure the number of labels match the number of sample arrays
    assert len(sample_arrays) == len(sample_labels), "Mismatch between number of sample arrays and labels."

    num_cols = math.ceil(math.sqrt(len(parameter_names)))
    num_rows = math.ceil(len(parameter_names) / num_cols)
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 2 * num_rows))
    axs = axs.flatten()

    for param_idx in range(len(parameter_names)):
        for i, sample_array in enumerate(sample_arrays):
            axs[param_idx].hist(sample_array[:, param_idx], bins=bins, alpha=0.5, density=True, histtype='step', label=sample_labels[i], range=parameter_ranges[param_idx])
        axs[param_idx].axvline(parameter_nominals[param_idx], linestyle='--', color='k', linewidth=1)
        axs[param_idx].set_xlabel(parameter_names[param_idx])
        axs[param_idx].set_ylabel('Density')

    axs[0].legend()
    fig.suptitle(title)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    return fig


def plot_2D_corner(sample_arrays, sample_labels, parameter_names, parameter_ranges, parameter_nominals=None, bins=100, title="Corner Plot"):
    """
    Plots a 2D corner plot with 2D density histograms off-diagonal and 1D histograms on the diagonal.

    Args:
        sample_arrays (list[np.ndarray]): List of 2D sample arrays to be plotted, where each row represents a sample and each column represents a parameter.
        sample_labels (list[str]): List of labels corresponding to each sample array.
        parameter_names (list[str]): List of parameter names.
        parameter_ranges (list[tuple]): List of (min, max) ranges for each parameter.
        parameter_nominals (list[float], optional): List of nominal (reference) values for each parameter.
        bins (int or list): Number of bins or a list of bin edges for the histograms.
        title (str, optional): Title of the plot. Default is "Corner Plot".

    Returns:
        matplotlib.figure.Figure: The figure object containing the plotted distributions.

    Raises:
        AssertionError: If the number of sample arrays does not match the number of labels.
    """

    # Ensure the number of labels match the number of sample arrays
    assert len(sample_arrays) == len(sample_labels), "Mismatch between number of sample arrays and labels."

    num_params = len(parameter_names)
    fig, axs = plt.subplots(num_params, num_params, figsize=(3 * num_params, 3 * num_params))

    for row in range(num_params):
        for col in range(num_params):
            ax = axs[row, col]
            
            # Hide plots in the upper triangle
            if row < col:
                ax.axis('off')
                continue
            
            # Diagonal: 1D histograms
            if row == col:
                for i, sample_array in enumerate(sample_arrays):
                    ax.hist(sample_array[:, col], bins=bins, alpha=0.5, density=True, histtype='step', label=sample_labels[i], range=parameter_ranges[col])
                ax.set_xlim(*parameter_ranges[col])
                ax.set_xlabel(parameter_names[col])
                if parameter_nominals:
                    ax.axvline(parameter_nominals[col], linestyle='--', color='k', linewidth=1)
            
            # Off-diagonal: 2D histograms
            else:
                for i, sample_array in enumerate(sample_arrays):
                    hist2d_params = {
                        "bins": bins,
                        "range": [parameter_ranges[col], parameter_ranges[row]],
                        "cmap": 'Blues',
                        "density": True
                    }
                    ax.hist2d(sample_array[:, col], sample_array[:, row], **hist2d_params)
                ax.set_xlim(*parameter_ranges[col])
                ax.set_ylim(*parameter_ranges[row])
                ax.set_xlabel(parameter_names[col])
                ax.set_ylabel(parameter_names[row])
                if parameter_nominals:
                    ax.axvline(parameter_nominals[col], linestyle='--', color='k', linewidth=1)
                    ax.axhline(parameter_nominals[row], linestyle='--', color='k', linewidth=1)

    # We set the legend on one of the diagonal plots for compactness
    axs[0,0].legend(loc='upper right')
    fig.suptitle(title)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    return fig


def plot_random_sample_predictions(samples, data_gen_func, observed_data, N, gen_func_args=None, title=None):
    """
    Draws N random samples from the given sample array, generates predicted datasets 
    for each sample, and plots them against the observed dataset.
    
    Args:
        samples (np.ndarray): Sample array where each row is a sample.
        data_gen_func (function): Data generation function that takes in a sample and additional arguments.
        observed_data (np.ndarray): Observed data set for reference.
        N (int): Number of random samples to be drawn.
        gen_func_args (dict, optional): Additional arguments to be passed to the data generation function.
        
    Returns:
        matplotlib.figure.Figure: The figure object containing the plotted predictions and observed data.
    """
    
    if gen_func_args is None:
        gen_func_args = {}
    
    # Check if the number of samples requested is valid
    total_samples = samples.shape[0]
    if N <= 0 or N > total_samples:
        raise ValueError(f"Invalid number of samples requested: {N}. It should be between 1 and {total_samples}.")
    
    # Select N random samples
    random_samples = samples[np.random.choice(total_samples, N, replace=False)]
    
    plt.figure(figsize=(12, 6))
    
    # Generate and plot predicted data for each random sample
    for sample in random_samples:
        predicted_data = data_gen_func(sample, **gen_func_args)
        plt.plot(predicted_data, 'b-', alpha=0.25)
    
    # Plot observed data for comparison
    plt.plot(observed_data, 'ro', label="Observed Data", alpha=0.45)
    plt.legend()
    if title:
        plt.title(f"{title}")
    else:
        plt.title(f"{N} Randomly Selected Predictions vs Observed Data")
    return plt.gcf()