from sklearn.mixture import GaussianMixture
import numpy as np
from scipy.stats import uniform
import matplotlib.pyplot as plt


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
        print(f"AIC min: k=gmm_best_aic_idx+1, AIC={aics[gmm_best_aic_idx]}")
        print(f"BIC min: k=gmm_best_bic_idx+1, AIC={bics[gmm_best_bic_idx]}")
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