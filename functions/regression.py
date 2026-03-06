"""
Frechet Regression Module for Multivariate Distributions

This module provides comprehensive functionality for performing Frechet regression
on multivariate distributions using the Nonparanormal Transport (NPT) metric.

Main Components:
- Core Frechet regression using NPT metric (decoupled marginals and latent correlations)
- Latent correlation computation via R's latentcor package
- Marginal distribution regression via R's fastfrechet package
- High-level wrapper functions for common analysis patterns
- Utility functions for data processing and validation

Dependencies:
- R packages: latentcor, fastfrechet
- Python packages: numpy, pandas, scipy

Author: Junyoung Park (junyoup@umich.edu)
Last modified: December 2025
"""

import numpy as np
import pandas as pd

# Internal module imports
from .corr_barycenter import riemannian_corr_barycenter
from .dist import Bures_Wasserstein
from .r_utils import setup_r_environment, import_r_packages, get_r_objects

# Standard normal CDF for convenience


# =============================================================================
# R ENVIRONMENT SETUP
# =============================================================================

# Initialize R environment and import required packages
setup_r_environment()
latentcor, fastfrechet = import_r_packages()
ro = get_r_objects()

# Get R base functions for data type conversions
r_matrix = ro.r.matrix
r_FloatVector = ro.FloatVector
r_StrVector = ro.StrVector


# =============================================================================
# CORE FRECHET REGRESSION FUNCTIONS
# =============================================================================

def npt_frechetreg(X, Y, Z=None, test_Y=None, M=200, latentcor=None, do_mse=True, do_cor_err=True,
                   verbose=True, bounds=None, oracle=False):
    """
    Perform Frechet regression using the Nonparanormal Transport (NPT) metric.

    This function implements Frechet regression for multivariate distributions
    within the nonparanormal (Gaussian copula) family. The regression is decoupled
    into two components based on the NPT metric structure:
    1. Marginal distributions: Regressed using univariate Wasserstein Frechet regression.
    2. Latent correlations: Regressed using Riemannian Frechet regression on the
       manifold of correlation matrices (via Bures-Wasserstein metric).

    Parameters
    ----------
    X : numpy.ndarray
        Predictor matrix of shape (n, p), where n is the number of observations
        and p is the number of predictors.
    Y : list of numpy.ndarray
        Response distributions. A list of length n, where each element is an
        (N_i, d) array representing the i-th multivariate distribution with
        d dimensions and N_i samples.
    Z : numpy.ndarray, optional
        Points at which to compute regression predictions, of shape (m, p).
        If None, predictions are computed at the input points X.
    test_Y : list of numpy.ndarray, optional
        Test response distributions. Used only if Z is provided and do_mse is True.
        A list of length m, where each element is an (N_i^test, d) array 
        representing the i-th multivariate distribution with d dimensions 
        and N_i^test samples.
    M : int or numpy.ndarray, default=200
        Number of quantile grid points used to represent marginal distributions.
        If an array of shape (M, ) is provided, it is used as the quantile grid directly.
    latentcor : numpy.ndarray, optional
        Precomputed latent correlation matrices of shape (n, d, d).
        If None, they are computed automatically from Y.
    do_mse : bool, default=True
        If True, computes mean squared errors (MSE) for marginal predictions
    verbose : bool, default=True
        If True, prints progress messages during computation.
    bounds : list of tuple, optional
        List of (lower, upper) bounds for the quantile functions of each dimension.
        If None, no bounds are applied.

    Returns
    -------
    dict
        A dictionary containing the regression results:
        - 'Marginals': list of numpy.ndarray
            Predicted marginal quantiles for each dimension.
        - 'LatentCor': numpy.ndarray
            Predicted correlation matrices at points Z, shape (m, d, d).
        - 'Z': numpy.ndarray
            The prediction points used.
        
        if do_mse is True, also includes:
        - 'marginal_frechet_means': list of numpy.ndarray
            Frechet means of marginals.
        - 'marginal_frechet_vars': list of float
            Frechet variances of marginals.
        - 'marginal_mses': list of float, optional
            Mean squared errors for marginals.

    Raises
    ------
    TypeError
        If inputs are not of the expected types.
    ValueError
        If input dimensions are inconsistent or contain invalid values.
    """

    # Validate inputs and set bounds if necessary
    bounds = _validate_frechet_inputs(X, Y, Z, test_Y, M, latentcor, bounds, oracle=oracle)

    # ========================================================================== 
    # MAIN COMPUTATION
    # ==========================================================================
    
    n = X.shape[0]
    d = Y.shape[1] if oracle else Y[0].shape[1]
    
    if Z is not None:
        if Z.shape[1] != X.shape[1]:
            raise ValueError("Z must have the same number of columns as X")
        if Z.ndim != 2:
            raise ValueError("Z must be a 2D array")
    else:
        # Use training data for MSE if no test data provided
        Z = X.copy()
        do_mse = True # Ensure we calculate MSE (for R^2 computation)
        test_Y = Y 

    # Marginal Frechet regression for each dimension
    if verbose: print("    Computing Frechet regression of marginals...", end="    ")
    
    marginal_predictions = []

    # Store Frechet means, variances, and mses for each marginal (if do_mse==True)
    frechet_means = []
    frechet_vars = []
    mses = []

    # Create quantile grid for marginal distributions
    if isinstance(M, int):
        quantile_grid = np.linspace(0, 1, M)
    else:
        quantile_grid = M

    def _is_simulation_test_Y(test_Y_value):
        return isinstance(test_Y_value, tuple) and len(test_Y_value) == 2

    sim_test_Y = _is_simulation_test_Y(test_Y)

    for i in range(d):
        if oracle:
            # Oracle mode: Y already contains quantile functions of shape (n, d, M)
            Y_i_quantiles = Y[:, i, :]
        else:
            # Extract i-th marginal from all distributions
            Y_i = [Y[j][:, i] for j in range(len(Y))]

            # Compute quantiles for each distribution's i-th marginal
            Y_i_quantiles = [np.quantile(Y_i[j], quantile_grid) for j in range(len(Y_i))]
            Y_i_quantiles = np.array(Y_i_quantiles)
        
        # Perform Frechet regression on the i-th marginal distributions
        Y_pred_i = marginal_frechet(X, Y_i_quantiles, Z=Z, lower=bounds[i][0], upper=bounds[i][1])

        # Compute MSE if requested
        if do_mse:
            # Store Frechet mean of the i-th marginal
            mean = np.mean(Y_i_quantiles, axis=0)
            frechet_means.append(mean)  

            # Store Frechet variance of the i-th marginal (using trapezoidal integration)
            var_squared = (Y_i_quantiles - mean) ** 2
            var = np.trapz(var_squared, x=quantile_grid, axis=1).mean()
            frechet_vars.append(var)
            
            # L^2 distance for the test distribution using trapezoidal rule on [0, 1]
            if sim_test_Y:
                # Simulation: test_Y[0] already contains quantiles of shape (m, d, M)
                assert test_Y is not None
                test_Y_i_quantiles = test_Y[0][:, i, :]
            elif np.array_equal(X, Z) or test_Y is Y:
                # Default: use the input data for MSE calculation
                test_Y_i_quantiles = Y_i_quantiles
            else:
                # test_Y is a list of empirical samples
                test_Y_i = [test_Y[j][:, i] for j in range(len(test_Y))]
                test_Y_i_quantiles = [np.quantile(test_Y_i[j], quantile_grid) for j in range(len(test_Y_i))]
                test_Y_i_quantiles = np.array(test_Y_i_quantiles)
            squared_diff = (Y_pred_i - test_Y_i_quantiles) ** 2
            mse = np.trapz(squared_diff, x=quantile_grid, axis=1).mean()
            mses.append(mse)
            
        marginal_predictions.append(Y_pred_i)

    if verbose: print("Completed.")

    # Latent correlation regression
    if verbose: print("    Computing regression of latent correlations ...", end="    ")

    latent_cors = get_latent_cor(Y) if latentcor is None else latentcor
    cor_preds = corr_frechet(X, latent_cors, Z=Z)
    
    test_latent_cors = None
    corr_mean = None

    if do_cor_err:
        # Frechet mean of correlations
        corr_mean = riemannian_corr_barycenter(latent_cors, record=False, verbose=False)

        if sim_test_Y:
            assert test_Y is not None
            test_latent_cors = test_Y[1]
        elif np.array_equal(X, Z) or test_Y is Y:
            test_latent_cors = latent_cors
        else:
            test_latent_cors = get_latent_cor(test_Y)


        sq_err, var_cor, dist_id = 0., 0., 0.
        for i in range(len(test_latent_cors)):
            # Distance from the latent correlation matrix to the frechet fit
            sq_err += Bures_Wasserstein(test_latent_cors[i], cor_preds[i])

            # Distance from the latent correlation matrix to the mean
            var_cor += Bures_Wasserstein(test_latent_cors[i], corr_mean)

            # Distance from the identity matrix to the prediction (for reference)
            dist_id += Bures_Wasserstein(test_latent_cors[i], np.eye(d))

    if verbose: print("Completed.")

    # Prepare results
    result = {
        'Marginals': marginal_predictions,
        'LatentCor': cor_preds,
        'Z': Z
    }
    
    if do_mse:
        result['marginal_mses'] = mses 
        result['marginal_frechet_means'] = frechet_means
        result['marginal_frechet_vars'] = frechet_vars

    if do_cor_err:
        assert test_latent_cors is not None
        assert corr_mean is not None
        n_test = len(test_latent_cors)
        result['latentcor_mse'] = sq_err / n_test
        result['latentcor_frechet_mean'] = corr_mean
        result['latentcor_frechet_var'] = var_cor / n_test
        result['latentcor_identity_mse'] = dist_id / n_test

    return result


# =============================================================================
# MARGINAL AND CORRELATION REGRESSION FUNCTIONS
# =============================================================================

def marginal_frechet(X, Y, Z=None, lower=None, upper=None):
    """
    Perform Frechet regression for univariate quantile functions.

    This function wraps the R function `fastfrechet::frechetreg_univar2wass`
    to compute Frechet regression for univariate distributions represented
    by their quantile functions.

    Parameters
    ----------
    X : numpy.ndarray
        Predictor matrix of shape (n, p).
    Y : numpy.ndarray
        Quantile representations of marginal distributions, of shape (n, M).
    Z : numpy.ndarray, optional
        Prediction points of shape (m, p). If None, predictions are computed at X.
    lower : float, optional
        Lower bound for the quantile functions.
    upper : float, optional
        Upper bound for the quantile functions.

    Returns
    -------
    numpy.ndarray
        Predicted quantile functions at points Z, of shape (m, M).

    Notes
    -----
    This function handles the interface between Python numpy arrays
    and R matrices, ensuring proper data type conversion and ordering.
    """
    if Z is None:
        Z = X.copy()
    
    # Convert numpy arrays to R matrices with proper ordering
    X_r = r_matrix(r_FloatVector(X.flatten()), nrow=X.shape[0], ncol=X.shape[1], byrow=True)
    Y_r = r_matrix(r_FloatVector(Y.flatten()), nrow=Y.shape[0], ncol=Y.shape[1], byrow=True)
    Z_r = r_matrix(r_FloatVector(Z.flatten()), nrow=Z.shape[0], ncol=Z.shape[1], byrow=True)

    # Call R function with appropriate parameters
    # Set R infinity values for None bounds (default)
    if lower is None:
        lower = ro.r('-Inf')[0]
    if upper is None:
        upper = ro.r('Inf')[0]
    
    Y_pred = fastfrechet.frechetreg_univar2wass(X_r, Y_r, Z_r, lower=lower, upper=upper)

    # Convert result back to numpy array
    Y_output = np.array(Y_pred.rx2('Qhat'))
    return Y_output


def corr_frechet(X, Y, Z=None, **params):
    """
    Perform Frechet regression on the manifold of correlation matrices.

    This function performs Frechet regression of correlation matrices under
    the Bures-Wasserstein metric, using Riemannian gradient descent with
    projection.

    Parameters
    ----------
    X : numpy.ndarray
        Predictor matrix of shape (n, p).
    Y : numpy.ndarray
        Response correlation matrices of shape (n, d, d).
    Z : numpy.ndarray, optional
        Prediction points of shape (m, p). If None, predictions are computed at X.
    params: dict, optional
        Additional parameters to pass to the riemannian_corr_barycenter function,
        such as step_size, max_iter, tol, proj=False (for covariance regression)

    Returns
    -------
    numpy.ndarray
        Predicted correlation matrices at points Z, of shape (m, d, d).
    """
    # Basic validation
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of rows (samples)")
    
    d = 2 if Y.ndim < 2 else Y.shape[1]
    
    # Handle prediction points
    if Z is not None:
        if Z.shape[1] != X.shape[1]:
            raise ValueError("Z must have the same number of columns as X")
        if Z.ndim != 2:
            raise ValueError("Z must be a 2D array")
    else:
        Z = X.copy()

    weights_mat = global_frechet_weights(X, Z)  # shape (m, n)
        
    # Compute Frechet regression using Riemannian optimization for barycenters
    Z_pred = np.zeros((Z.shape[0], d, d))

    params.setdefault('verbose', False)

    for i in range(Z.shape[0]):
        Z_pred[i] = riemannian_corr_barycenter(Y, weights_mat[i], record=False, **params)

    return Z_pred


def global_frechet_weights(X, Z, tol=1e-10):
    """
    Weight computation for global Frechet regression.
    Scaling of X and Z are performed for numerical stability
    to prevent heterogeneity in predictor scales across dimensions.
    """
    # Scale X and Z to have zero mean and unit variance
    X_mean = np.mean(X, axis=0)
    X_std = np.maximum(np.std(X, axis=0), tol)  # Avoid division by zero
    X_scaled = (X - X_mean) / X_std
    Z_scaled = (Z - X_mean) / X_std

    # Weights for Frechet regression (m x n)
    cov_inv_X = np.linalg.pinv(X_scaled.T @ X_scaled)
    weights_mat = Z_scaled @ cov_inv_X @ X_scaled.T
    weights_mat += np.ones_like(weights_mat) / X.shape[0]
    
    # Ensure weights sum to 1 (ensuring numerical stability)
    weights_mat /= np.sum(weights_mat, axis=1, keepdims=True)

    return weights_mat


# =============================================================================
# LATENT CORRELATION COMPUTATION
# =============================================================================

def get_latent_cor(indiv_samples):
    """
    Compute latent correlation matrices using the R 'latentcor' package.

    This function interfaces with R to compute Gaussian copula correlation
    matrices for each multivariate distribution in the input list.

    Parameters
    ----------
    indiv_samples : list of numpy.ndarray
        A list of length n, where each element is an (N_i, d) array representing
        an individual multivariate distribution sample.

    Returns
    -------
    numpy.ndarray
        Array of correlation matrices of shape (n, d, d), one for each input distribution.

    Notes
    -----
    - Adds small noise to constant columns to avoid numerical issues in 'latentcor'.
    - Uses identity matrices as a fallback if computation fails for a sample.
    """
    corr_mats = []
    failed_indices = []
    
    for i in range(len(indiv_samples)):
        X = indiv_samples[i]
        try:
            # Ensure proper data type
            X = np.asarray(X, dtype=np.float64)
            
            # Additional validation: check for constant columns to prevent issues in latentcor computation
            if np.any(np.std(X, axis=0) == 0):
                print(f"Warning: Sample {i} has constant columns. Adding small noise.")
                for col in range(X.shape[1]):
                    if np.std(X[:, col]) == 0:
                        X[:, col] += np.random.normal(0, 1e-10, X.shape[0])

            # Final validation
            X_flat = X.flatten()
            if not np.all(np.isfinite(X_flat)):
                raise ValueError(f"Sample {i} still contains non-finite values after cleaning")
                
            # Convert to R matrix and compute latent correlation
            X_r = r_matrix(r_FloatVector(X_flat), nrow=X.shape[0], ncol=X.shape[1], byrow=True)
            cor_matrix = latentcor.latentcor(X_r, types=r_StrVector(['con']))
            cor_output = np.array(cor_matrix.rx2('R'))
            corr_mats.append(cor_output)
            
        except Exception as e:
            print(f"Error processing sample {i}: {str(e)}")
            print(f"Sample shape: {X.shape}")
            print(f"Sample stats: min={np.min(X)}, max={np.max(X)}, mean={np.mean(X)}")
            failed_indices.append(i)
            
            # Add identity matrix as fallback
            corr_mats.append(np.eye(indiv_samples[i].shape[1]))

    if failed_indices:
        print(f"Warning: Failed to compute latent correlations for {len(failed_indices)}",
              f"samples: {failed_indices}")
        print("Using identity matrices as placeholders for failed samples.")

    return np.asarray(corr_mats)


# =============================================================================
# HIGH-LEVEL WRAPPER FUNCTIONS
# =============================================================================

def multivariate_frechet_regression(predictor_name, multivar_data, feature_names, M=200,
                    bounds=None, Z=None, space_interval=10, min_q=0., max_q=1.,
                    quadratic=False, r_squared=False,
                    dist_col='multivar_distribution', latentcor_col='latentcor', verbose=True,
                    **kwargs):
    """
    High-level interface for multivariate Frechet regression using the NPT metric.

    This function provides a convenient interface for performing Frechet regression
    on datasets with multiple predictors and multivariate response distributions.
    It handles data preparation, regression computation, and optional R^2 calculation.

    Parameters
    ----------
    predictor_name : str or list of str
        Name(s) of the predictor column(s) in `multivar_data`.
    multivar_data : pandas.DataFrame
        DataFrame containing the distributions and predictor variables.
    feature_names : list of str
        Names of the features (dimensions) in the multivariate distributions.
        The order must be consistent with the data in `dist_col`.
    M : int, default=200
        Number of quantile grid points used for marginal distributions.
    bounds : dict, optional
        Dictionary specifying bounds for quantile functions. Keys must be in
        `feature_names`, and values must be (lower, upper) tuples.
    Z : numpy.ndarray, optional
        Custom prediction points. If None, an evenly spaced grid is created.
    space_interval : int, default=10
        Number of evenly spaced points for the default prediction grid.
    min_q : float, default=0.0
        Minimum quantile for the default prediction grid (relative to predictor range).
    max_q : float, default=1.0
        Maximum quantile for the default prediction grid (relative to predictor range).
    quadratic : bool, default=False
        If True, includes quadratic terms of the predictors in the regression.
    r_squared : bool, default=False
        If True, computes and returns generalized R^2 values for model assessment.
    dist_col : str, default='multivar_distribution'
        Name of the column in `multivar_data` containing the multivariate distributions.
    latentcor_col : str, default='latentcor'
        Name of the column in `multivar_data` containing precomputed latent correlation matrices.
    verbose : bool, default=True
        If True, prints progress messages.

    Returns
    -------
    dict
        A dictionary containing:
        - 'fits': pandas.DataFrame
            DataFrame containing the prediction points (Z) and the predicted
            marginal fits ('marginal_fits') and latent correlation fits ('latentcor_fits').
        - 'feature_names': list of str
            The list of feature names used.
        - 'R_squares': pandas.DataFrame, optional
            DataFrame containing R^2 values for marginals, latent correlations, and total
            (only returned if r_squared=True).
    """
    # Validate predictor existence
    if isinstance(predictor_name, str):
        if predictor_name not in multivar_data.columns:
            raise ValueError(f"Predictor '{predictor_name}' not found in the data")
    else:
        for pred in predictor_name:
            if pred not in multivar_data.columns:
                raise ValueError(f"Predictor '{pred}' not found in the data")
    
    # Validate and process bounds
    bounds_list = _process_bounds_dict(bounds, feature_names)

    # Create predictor matrix
    X = multivar_data[predictor_name].values.reshape(len(multivar_data), -1)

    # Handle missing values in predictors
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        print("Warning: Predictor contains NaN or infinite values. Cleaning up...")
        valid_indices = ~np.isnan(X).any(axis=1) & ~np.isinf(X).any(axis=1)
        X = X[valid_indices]
        multivar_data = multivar_data.iloc[valid_indices]
        print(f"Predictor cleaned. Remaining sample size: {X.shape[0]}")
        
    # Set up prediction points
    if Z is not None:
        Z = Z.reshape(-1, X.shape[1]) if Z.ndim == 1 else Z
    elif space_interval is not None:
        if space_interval ** X.shape[1] > X.shape[0]:
            print("Warning: space_interval too large; for R^2 purposes, set space_interval to None.")
        # Create evenly spaced prediction grid
        mini, maxi = np.quantile(X, min_q, axis=0), np.quantile(X, max_q, axis=0)
        grids = [np.linspace(mini[i], maxi[i], space_interval) for i in range(X.shape[1])]
        Z = np.array(np.meshgrid(*grids)).T.reshape(-1, X.shape[1])
    else:
        # Take the first sample as prediction point
        Z = X[0, :].reshape(1, -1)

    # Add quadratic features if requested
    if quadratic:
        X = np.hstack((X, X**2))
        Z = np.hstack((Z, Z**2))
    
    # Extract multivariate distributions
    Y_list = multivar_data[dist_col].tolist()
    
    # Extract the precomputed latent correlation matrices if available
    if latentcor_col in multivar_data.columns:
        latent_cors = np.array([mat for mat in multivar_data[latentcor_col]])
    else:
        latent_cors = get_latent_cor(Y_list)
    
    ####  Perform Frechet regression ####
    if verbose:
        print(f"Performing Frechet regression using {predictor_name} as predictor...")
    frechet_fit = npt_frechetreg(X, Y_list, Z, latentcor=latent_cors, M=M, bounds=bounds_list, verbose=verbose,
                                 do_mse=False, do_cor_err=False, **kwargs)

    # Format the result DataFrame
    predictor_cols = [predictor_name] if isinstance(predictor_name, str) else predictor_name
    if quadratic:
        quad_cols = [f"{name}^2" for name in predictor_cols]
        predictor_cols = predictor_cols + quad_cols

    Z_df = pd.DataFrame(Z, columns=predictor_cols)
    result_df = Z_df.copy()
    result_df['marginal_fits'] = list(np.array(frechet_fit['Marginals']).swapaxes(0, 1))
    result_df['latentcor_fits'] = list(frechet_fit['LatentCor'])

    result = {
        'fits': result_df,
        'feature_names': feature_names
    }

    # Compute R^2 values if requested
    if r_squared:
        if verbose: print("    Computing R-squared values for marginal fits...", end=" ")
        sample_fit = npt_frechetreg(X, Y_list, Z=None, latentcor=latent_cors, M=M, verbose=False,
                                   bounds=bounds_list)

        # R^2 for marginal fits
        r2_marginals = []
        d = len(sample_fit['Marginals'])
        for j in range(d):
            r2_marginals.append(1 - sample_fit['marginal_mses'][j] / sample_fit['marginal_frechet_vars'][j])
        r2_df = pd.DataFrame([r2_marginals], columns=feature_names, index=["R2"])
        if verbose: print("Done")

        # R^2 for latent correlation fit
        if verbose: print("    Computing R-squared value for latent correlation fit...", end=" ")
        corr_mean = riemannian_corr_barycenter(latent_cors, record=False, verbose=False)

        num, denom = 0, 0
        for i in range(len(latent_cors)):
            # Distance from the latent correlation matrix to the frechet fit
            num += Bures_Wasserstein(latent_cors[i], sample_fit['LatentCor'][i])
            
            # Distance from the latent correlation matrix to the mean
            denom += Bures_Wasserstein(latent_cors[i], corr_mean)
        r2_df['latentcor'] = 1 - num / denom if denom > 0 else None
        if verbose: print("Done!")

        # Total R^2
        r2_df['total'] = 1 - (np.sum(sample_fit['marginal_mses']) + num) / (np.sum(sample_fit['marginal_frechet_vars']) + denom)

        result['R_squares'] = r2_df
    
    return result


# =============================================================================
# UTILITY AND VALIDATION FUNCTIONS
# =============================================================================

def _validate_frechet_inputs(X, Y, Z, test_Y, M, latentcor, bounds, oracle=False):
    """
    Internal helper to validate inputs for npt_frechetreg.

    Parameters
    ----------
    X : numpy.ndarray
        Predictor matrix.
    Y : list of numpy.ndarray
        Response distributions.
    Z : numpy.ndarray or None
        Prediction points.
    test_Y : list of numpy.ndarray or None
        Test response distributions.
    M : int or numpy.ndarray
        Number of quantile grid points.
    latentcor : numpy.ndarray or None
        Latent correlation matrices.
    bounds : list or None
        Bounds for quantile functions.

    Returns
    -------
    list
        Validated bounds list.

    Raises
    ------
    TypeError, ValueError
        If inputs are invalid.
    """
    # Validate predictor matrix X
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if X.shape[0] == 0:
        raise ValueError("X cannot be empty")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    
    # Validate response distributions Y
    if oracle:
        if not isinstance(Y, np.ndarray):
            raise TypeError("In oracle mode, Y must be a numpy array of shape (n, d, M)")
        if Y.ndim != 3:
            raise ValueError("In oracle mode, Y must be a 3D array (n, d, M)")
        if Y.shape[0] != X.shape[0]:
            raise ValueError(f"Y first dimension ({Y.shape[0]}) must match X rows ({X.shape[0]})")
        if Y.shape[2] == 0:
            raise ValueError("In oracle mode, Y must have M>0 quantile grid points")
        if np.any(np.isnan(Y)) or np.any(np.isinf(Y)):
            raise ValueError("Y contains NaN or infinite values")
        d = Y.shape[1]
    else:
        if not isinstance(Y, list) and not isinstance(Y, np.ndarray):
            raise TypeError("Y must be a list of numpy arrays or a numpy array")
        if len(Y) == 0:
            raise ValueError("Y cannot be empty")
        if len(Y) != X.shape[0]:
            raise ValueError(f"Y length ({len(Y)}) must match X rows ({X.shape[0]})")
        
        # Validate each distribution in Y
        d = Y[0].shape[1]
        for i, dist in enumerate(Y):
            if not isinstance(dist, np.ndarray):
                raise TypeError(f"Y[{i}] must be a numpy array")
            if dist.ndim != 2:
                raise ValueError(f"Y[{i}] must be a 2D array")
            if dist.shape[0] == 0:
                raise ValueError(f"Y[{i}] cannot be empty")
            if np.any(np.isnan(dist)) or np.any(np.isinf(dist)):
                raise ValueError(f"Y[{i}] contains NaN or infinite values")
            if dist.shape[1] != d:
                raise ValueError(f"All distributions in Y must have same dimension. "
                               f"Y[0] has {d}, Y[{i}] has {dist.shape[1]}")
    
    # Validate test distributions test_Y if not None
    if test_Y is not None:
        if len(test_Y) == 2:
            # For simulation purposes (hard-coded)
            pass
        else:
            if not isinstance(test_Y, list):
                raise TypeError("test_Y must be a list of numpy arrays or None")
            if Z is not None and len(test_Y) != Z.shape[0]:
                raise ValueError(f"test_Y length ({len(test_Y)}) must match Z rows ({Z.shape[0]})")
            for i, dist in enumerate(test_Y):
                if not isinstance(dist, np.ndarray):
                    raise TypeError(f"test_Y[{i}] must be a numpy array")
                if dist.ndim != 2:
                    raise ValueError(f"test_Y[{i}] must be a 2D array")
                if dist.shape[0] == 0:
                    raise ValueError(f"test_Y[{i}] cannot be empty")
                if np.any(np.isnan(dist)) or np.any(np.isinf(dist)):
                    raise ValueError(f"test_Y[{i}] contains NaN or infinite values")
                if dist.shape[1] != d:
                    raise ValueError(f"All distributions in test_Y must have same dimension. "
                                    f"test_Y[0] has {d}, test_Y[{i}] has {dist.shape[1]}")

    # Validate prediction points Z
    if Z is not None:
        if not isinstance(Z, np.ndarray):
            raise TypeError("Z must be a numpy array")
        if Z.ndim != 2:
            raise ValueError("Z must be a 2D array")
        if Z.shape[1] != X.shape[1]:
            raise ValueError(f"Z columns ({Z.shape[1]}) must match X columns ({X.shape[1]})")
        if np.any(np.isnan(Z)) or np.any(np.isinf(Z)):
            raise ValueError("Z contains NaN or infinite values")
    
    # Validate M
    if not (isinstance(M, int) and M > 0) and not (isinstance(M, np.ndarray) and M.ndim == 1):
        raise ValueError("M must be a positive integer or a 1D numpy array")
    if oracle:
        # At this point oracle-mode has validated Y is a numpy.ndarray
        expected_m = int(np.asarray(Y).shape[2])
        if isinstance(M, int) and M != expected_m:
            raise ValueError(f"In oracle mode, M={M} must match Y.shape[2]={expected_m}")
        if isinstance(M, np.ndarray) and M.shape[0] != expected_m:
            raise ValueError(f"In oracle mode, quantile grid length ({M.shape[0]}) must match Y.shape[2]={expected_m}")

    # Validate latentcor
    if latentcor is not None:
        if not isinstance(latentcor, np.ndarray):
            raise TypeError("latentcor must be a numpy array")
        if latentcor.ndim != 3:
            raise ValueError("latentcor must be a 3D array")
        if latentcor.shape[0] != len(Y):
            raise ValueError(f"latentcor first dimension ({latentcor.shape[0]}) "
                           f"must match Y length ({len(Y)})")
        if latentcor.shape[1] != d or latentcor.shape[2] != d:
            raise ValueError(f"latentcor dimensions must be (n, {d}, {d})")
    
    # Validate bounds
    if bounds is None:
        bounds = [(None, None)] * d
    else:
        if not isinstance(bounds, list):
            raise TypeError("bounds must be a list of tuples")
        if len(bounds) != d:
            raise ValueError(f"bounds length ({len(bounds)}) must match number of dimensions ({d})")
        for i, bound in enumerate(bounds):
            if not isinstance(bound, (tuple, list)) or len(bound) != 2:
                raise ValueError("Each bound must be a tuple or list of length 2")
            low, high = bound
            if low is not None and high is not None and low >= high:
                raise ValueError(f"Invalid bounds for dimension {i}: ({low}, {high})")
                
    return bounds


def _process_bounds_dict(bounds, feature_names):
    """
    Process bounds dictionary into a list of tuples consistent with feature_names.

    Parameters
    ----------
    bounds : dict or None
        Dictionary of bounds.
    feature_names : list of str
        List of feature names.

    Returns
    -------
    list
        List of (lower, upper) tuples.
    """
    if bounds is None:
        return [(None, None)] * len(feature_names)
        
    if not isinstance(bounds, dict):
        raise ValueError("bounds must be a dictionary with keys in feature_names")
        
    bounds_list = []
    for key in feature_names:
        if key not in bounds:
            bounds_list.append((None, None))
            continue
            
        val = bounds[key]
        if not isinstance(val, (list, tuple)) or len(val) != 2:
            raise ValueError(f"Bounds for '{key}' must be a list or tuple of length 2")
            
        if val[0] is not None and val[1] is not None and val[0] >= val[1]:
            raise ValueError(f"Lower bound must be less than upper bound for '{key}'")
            
        bounds_list.append(val)
        
    return bounds_list
