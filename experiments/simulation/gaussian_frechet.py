"""
Frechet regression for Gaussian distributions under Wasserstein distance.

This module implements Frechet regression for multivariate Gaussian distributions
using the Wasserstein (Bures-Wasserstein) distance. This provides a baseline
comparison for the Nonparanormal Transport (NPT) Frechet regression.

For Gaussian distributions, the Wasserstein distance reduces to the
Bures-Wasserstein distance with the following form:
    W_2^2(N(mu_1, Sigma_1), N(mu_2, Sigma_2))
        = ||mu_1 - mu_2||^2 + B^2(Sigma_1, Sigma_2)

where B is the Bures-Wasserstein distance on covariance matrices:
    B^2(Sigma_1, Sigma_2)
        = Tr[Sigma_1 + Sigma_2 - 2(Sigma_1^{1/2} Sigma_2 Sigma_1^{1/2})^{1/2}]

The Frechet regression decomposes into:
1. Linear regression on the mean vectors (standard OLS)
2. BW Frechet regression on the covariance matrices (Riemannian optimization)

Author: Junyoung Park
Last modified: December 2025
"""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from functions import Bures_Wasserstein, corr_frechet, global_frechet_weights


def estimate_gaussian_params(Y):
    """
    Estimate Gaussian mean vectors and covariance matrices from empirical samples.

    Parameters
    ----------
    Y : list of numpy.ndarray
        List of n empirical distributions, where each element is an (N_i, d) array
        representing samples from the i-th multivariate distribution.

    Returns
    -------
    means : numpy.ndarray
        Array of shape (n, d) containing estimated mean vectors.
    covs : numpy.ndarray
        Array of shape (n, d, d) containing estimated covariance matrices.
    """
    n = len(Y)
    d = Y[0].shape[1]

    means = np.zeros((n, d))
    covs = np.zeros((n, d, d))

    for i in range(n):
        means[i] = np.mean(Y[i], axis=0)
        # Use sample covariance (rowvar=False since rows are observations)
        covs[i] = np.cov(Y[i], rowvar=False)

    return means, covs


def mean_frechet(X, means, Z=None):
    """
    Perform Frechet regression on mean vectors.

    For Gaussian mean vectors, Frechet regression under Euclidean distance
    reduces to standard linear regression:
        mu_F(z) = sum_i s(X_i, z) mu_i

    Parameters
    ----------
    X : numpy.ndarray
        Predictor matrix of shape (n, p).
    means : numpy.ndarray
        Mean vectors of shape (n, d).
    Z : numpy.ndarray, optional
        Prediction points of shape (m, p). If None, predictions at X.

    Returns
    -------
    numpy.ndarray
        Predicted mean vectors of shape (m, d).
    """
    if Z is None:
        Z = X.copy()

    # Compute Frechet weights
    weights = global_frechet_weights(X, Z)  # shape (m, n)

    # Weighted combination of means mu_F(z) = sum_i s(X_i, z) mu_i
    mean_preds = weights @ means  # shape (m, d)

    return mean_preds


def gaussian_frechetreg(X, Y, Z=None, means=None, covs=None, verbose=True):
    """
    Perform Frechet regression for multivariate Gaussian distributions under
    the Wasserstein (Bures-Wasserstein) distance.

    This function estimates the conditional Frechet mean of Gaussian distributions
    given predictor values. The Wasserstein distance between Gaussians decomposes
    into mean and covariance components:
        W_2^2(N(mu_1, Sigma_1), N(mu_2, Sigma_2))
            = ||mu_1 - mu_2||^2 + B^2(Sigma_1, Sigma_2)

    Consequently, the Frechet regression also decomposes:
        - Mean vectors: Linear regression (weighted mean)
        - Covariance matrices: BW Frechet regression (Riemannian barycenter)

    Parameters
    ----------
    X : numpy.ndarray
        Predictor matrix of shape (n, p), where n is the number of observations
        and p is the number of predictors.
    Y : list of numpy.ndarray
        Response distributions. A list of length n, where each element is an
        (N_i, d) array representing samples from the i-th Gaussian distribution.
    Z : numpy.ndarray, optional
        Points at which to compute regression predictions, of shape (m, p).
        If None, predictions are computed at the input points X.
    means : numpy.ndarray, optional
        Precomputed mean vectors of shape (n, d). If None, estimated from Y.
    covs : numpy.ndarray, optional
        Precomputed covariance matrices of shape (n, d, d). If None, estimated from Y.
    verbose : bool, default=True
        If True, prints progress messages.

    Returns
    -------
    dict
        A dictionary containing:
        - 'means': numpy.ndarray of shape (m, d)
            Predicted mean vectors at prediction points Z.
        - 'covs': numpy.ndarray of shape (m, d, d)
            Predicted covariance matrices at prediction points Z.
        - 'Z': numpy.ndarray
            The prediction points used.
        - 'input_means': numpy.ndarray of shape (n, d)
            The estimated/provided mean vectors of input distributions.
        - 'input_covs': numpy.ndarray of shape (n, d, d)
            The estimated/provided covariance matrices of input distributions.
    """
    # Input validation
    n = X.shape[0]
    if len(Y) != n:
        raise ValueError(f"Number of distributions ({len(Y)}) must match number of predictors ({n})")

    if Z is None:
        Z = X.copy()
    elif Z.ndim == 1:
        Z = Z.reshape(1, -1)

    if Z.shape[1] != X.shape[1]:
        raise ValueError("Z must have the same number of columns as X")

    # Step 1: Estimate Gaussian parameters if not provided
    if means is None or covs is None:
        if verbose:
            print("    Estimating Gaussian parameters from empirical distributions...")
        est_means, est_covs = estimate_gaussian_params(Y)
        means = means if means is not None else est_means
        covs = covs if covs is not None else est_covs

    # Step 2: Frechet regression on mean vectors (linear regression)
    if verbose:
        print("    Computing Frechet regression of mean vectors...")
    mean_preds = mean_frechet(X, means, Z)

    # Step 3: Frechet regression on covariance matrices (BW barycenter)
    if verbose:
        print("    Computing Frechet regression of covariance matrices...")
    cov_preds = corr_frechet(X, covs, Z, verbose=verbose, proj=False)

    if verbose:
        print("    Gaussian Frechet regression completed.")

    return {
        "means": mean_preds,
        "covs": cov_preds,
        "Z": Z,
        "input_means": means,
        "input_covs": covs,
    }


def result_to_nonparanormal(result, quantile_grid):
    """
    Convert Gaussian Frechet regression results to nonparanormal format
    (quantiles and latent correlations).

    Must specify the quantile grid used.
    """
    mean_preds = result["means"]
    cov_preds = result["covs"]
    m, d = mean_preds.shape

    quantiles = np.zeros((m, d, len(quantile_grid)))
    latent_corrs = np.zeros((m, d, d))

    for i in range(m):
        # Symmetric normalization transformation to latent correlation.
        cov_matrix = cov_preds[i]
        std_devs = np.sqrt(np.diag(cov_matrix))
        latent_corrs[i] = cov_matrix / np.outer(std_devs, std_devs)

        for j in range(d):
            quantiles[i, j] = norm.ppf(quantile_grid, loc=mean_preds[i, j], scale=std_devs[j])

    return {
        "Marginal_Quantiles": quantiles,
        "Latent_Corrs": latent_corrs,
    }


def get_nonparanormal_mse(result, true_Y, quantile_grid):
    """
    Compute mean squared error for nonparanormal distributions.

    `true_Y` is expected to be a tuple of quantile arrays and latent
    correlation matrices.
    """
    pred_nonparanormal = result_to_nonparanormal(result, quantile_grid)

    pred_quantiles = pred_nonparanormal["Marginal_Quantiles"]
    pred_latent_corrs = pred_nonparanormal["Latent_Corrs"]

    true_quantiles = true_Y[0]
    true_latent_corrs = true_Y[1]

    m, d, _ = pred_quantiles.shape

    # 1. Marginal MSEs
    marginal_mses = []
    for j in range(d):
        squared_diff = (pred_quantiles[:, j, :] - true_quantiles[:, j, :]) ** 2
        mse = np.trapz(squared_diff, x=quantile_grid, axis=1).mean()
        marginal_mses.append(mse)

    # 2. Latent correlation MSE
    sq_err = 0
    for i in range(m):
        sq_err += Bures_Wasserstein(true_latent_corrs[i], pred_latent_corrs[i])

    latentcor_mse = sq_err / m

    return {
        "marginal_mses": marginal_mses,
        "latentcor_mse": latentcor_mse,
    }
