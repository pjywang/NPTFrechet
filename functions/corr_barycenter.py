import numpy as np
from numba import njit

from .dist import Bures_Wasserstein, positive_sqrtm

import time


# =============================================================================
# NUMBA-ACCELERATED FUNCTIONS
# =============================================================================

@njit(cache=True)
def _positive_sqrtm_numba(A, eps=1e-15):
    """Numba-accelerated positive square root of a PSD matrix."""
    A_sym = (A + A.T) / 2
    d, v = np.linalg.eigh(A_sym)
    d_sqrt = np.sqrt(np.maximum(d, eps))
    return v @ np.diag(d_sqrt) @ v.T


@njit(cache=True)
def _grad_barycenter_numba(A, samples, weights):
    """
    Numba-accelerated gradient of the barycenter functional.
    
    Parameters
    ----------
    A : np.ndarray (d, d) - current estimate
    samples : np.ndarray (n, d, d) - sample correlation matrices
    weights : np.ndarray (n,) - weights summing to 1
    
    Returns
    -------
    grad : np.ndarray (d, d)
    """
    d = A.shape[0]
    n = samples.shape[0]
    
    A_sym = (A + A.T) / 2
    eigvals, vecs = np.linalg.eigh(A_sym)
    eigvals_sqrt = np.sqrt(np.maximum(eigvals, 1e-10))
    
    A_sqrt = vecs @ np.diag(eigvals_sqrt) @ vecs.T
    A_sqrt_inv = vecs @ np.diag(1.0 / eigvals_sqrt) @ vecs.T
    
    grad = np.eye(d)
    
    for i in range(n):
        intermediate = A_sqrt @ samples[i] @ A_sqrt
        fidel = _positive_sqrtm_numba(intermediate)
        grad -= weights[i] * (A_sqrt_inv @ fidel @ A_sqrt_inv)
    
    return grad


@njit(cache=True)
def _BW_projection_numba(A):
    """Numba-accelerated BW projection to correlation matrix."""
    A_sym = (A + A.T) / 2
    d = A_sym.shape[0]
    std_diag = np.sqrt(np.diag(A_sym))
    
    C = np.empty((d, d))
    for i in range(d):
        for j in range(d):
            C[i, j] = A_sym[i, j] / (std_diag[i] * std_diag[j])
    
    for i in range(d):
        C[i, i] = 1.0
    
    return C


@njit(cache=True)
def _riemannian_barycenter_numba(samples, weights, step_size=1.0, max_iter=1000, tol=1e-6):
    """
    Numba-accelerated Riemannian correlation barycenter (for d >= 3).
    
    Parameters
    ----------
    samples : np.ndarray (n, d, d)
    weights : np.ndarray (n,)
    
    Returns
    -------
    A : np.ndarray (d, d) - barycenter correlation matrix
    """
    d = samples.shape[1]
    A = samples[0].copy()
    
    for iteration in range(max_iter):
        grad = _grad_barycenter_numba(A, samples, weights)
        
        S = step_size * (np.eye(d) - grad)
        exp_A_neg_grad = S @ A @ S
        
        A_new = _BW_projection_numba(exp_A_neg_grad)
        
        # Check convergence (Frobenius norm)
        diff = 0.0
        for i in range(d):
            for j in range(d):
                diff += (A_new[i, j] - A[i, j]) ** 2
        
        if np.sqrt(diff) < tol:
            return A_new
        
        A = A_new
    
    return A


# =============================================================================


def get_bivar_corr(rho):
    """Correlation matrix with given correlation rho"""
    return np.array([[1.0, rho], [rho, 1.0]])


def barycenter_bivar_corr(samples, weights):
    """
    Barycenter of bivariate correlation matrices using the closed formula

    Parameters
    ----------
    samples : list of 2D np.ndarrays or list of correlations
    weights : list of weights, summing to 1

    Returns
    ----------
    np.ndarray:  Barycenter correlation matrix
    """
    correlations = np.zeros(len(samples))
    # Check if samples are bivariate correlation matrices
    for i, sample in enumerate(samples):
        if type(sample) == np.ndarray:
            correlations[i] = sample[0, 1]
            if sample.shape != (2, 2): 
                raise ValueError("All samples must be 2x2 correlation matrices or univariate correlations.")
        elif sample < -1 or sample > 1:
            raise ValueError("Correlation values must be between -1 and 1.")
        else:
            correlations[i] = sample
    # Check if weights are valid
    if len(weights) != len(samples):
        raise ValueError("Weights must match the number of samples.")
    if not np.isclose(sum(weights), 1):
        raise ValueError("Weights must sum to 1.")

    # Compute the weighted average of the samples
    weights = np.array(weights)

    # a, b parameters in the paper
    a = np.sum(weights * np.sqrt(1 + correlations))
    b = np.sum(weights * np.sqrt(1 - correlations))

    if a < 0 and b >= 0:
        bary_corr = get_bivar_corr(-1)
    elif b < 0 and a >= 0:
        bary_corr = get_bivar_corr(1)
    elif a < 0 and b < 0:
        bary_corr = -1 if a > b else 1
        bary_corr = get_bivar_corr(bary_corr)
    else:
        bary_corr = get_bivar_corr((a**2 - b**2) / (a**2 + b**2))
                                   
    return bary_corr
    

def barycenter_functional(A, samples, weights):
    """
    The barycenter functional, defined by samples and weights, evaluated at the psd matrix A.

    Parameters
    ----------
    A : np.ndarray
        A positive semi-definite matrix.
    samples : list of np.ndarrays or correlations (if 2 dimensional)
    weights : list of weights, summing to 1

    Returns
    ----------
    float: The value of the barycenter functional at A.
    """
    if len(samples) != len(weights):
        raise ValueError("Weights must match the number of samples.")
    if not np.isclose(sum(weights), 1):
        raise ValueError("Weights must sum to 1.")
    if isinstance(samples[0], float) or isinstance(samples[0], int):
        samples = [get_bivar_corr(sample) for sample in samples]
    
    # Weighted sum of squared Bures-Wasserstein distances
    functional_value = 0.0
    for sample, weight in zip(samples, weights):
        functional_value += weight * Bures_Wasserstein(A, sample)
    
    return functional_value


def grad_barycenter_functional(A, samples, weights):   
    """
    Gradient of the barycenter functional at the psd matrix A (in the Euclidean geometry).
    The closed formula is provided in the paper:
        Bhatia et al. (2019) "On the Bures-Wasserstein distance between positive definite matrices"
    
    It is basically I_d - (weighted sum of geometric means of A^{-1} and samples).
        
    Parameters
    ----------
    A : np.ndarray
        A positive semi-definite matrix.
    samples : list of np.ndarrays or correlations (if 2 dimensional)
    weights : list of weights, summing to 1
    """
    if len(samples) != len(weights):
        raise ValueError("Weights must match the number of samples.")
    if not np.isclose(sum(weights), 1):
        raise ValueError("Weights must sum to 1.")
    if isinstance(samples[0], float) or isinstance(samples[0], int):
        samples = [get_bivar_corr(sample) for sample in samples]
    
    d = A.shape[0] 
    grad = np.eye(d)

    # Ensure symmetry of A (for numerical stability)
    A = (A + A.T) / 2

    # Square root calculations
    eigvals, vecs = np.linalg.eigh(A)
    eigvals_sqrt = np.sqrt(np.maximum(eigvals, 1e-10)) # filter small negative eigenvalues that may occur due to floating point errors
    A_sqrt = vecs @ np.diag(eigvals_sqrt) @ vecs.T
    A_sqrt_inv = vecs @ np.diag(1.0 / eigvals_sqrt) @ vecs.T

    # weighted sum of geometric means between A^{-1} and sample
    for sample, weight in zip(samples, weights):    
        fidel = positive_sqrtm(A_sqrt @ sample @ A_sqrt)
        grad -= weight * (A_sqrt_inv @ fidel @ A_sqrt_inv)
    
    return grad


def check_and_fix_correlation_matrices(samples, min_eig_threshold=1e-8, verbose=True):
    """
    Check the condition number of correlation matrices and fix those with 
    eigenvalues below the threshold.
    
    Parameters
    ----------
    samples : list of np.ndarray
        List of correlation matrices
    min_eig_threshold : float
        Minimum eigenvalue threshold
    verbose : bool
        Whether to print warnings
        
    Returns
    ----------
    list: List of fixed correlation matrices
    """
    fixed_samples = []
    for i, sample in enumerate(samples):
        # Check eigenvalues
        eig_vals = np.linalg.eigvalsh(sample)
        min_eig = np.min(eig_vals)
        max_eig = np.max(eig_vals)
        condition_number = max_eig / (min_eig if min_eig > 0 else min_eig_threshold)
        
        if min_eig < min_eig_threshold:
            if verbose:
                print(f"Warning: Sample {i} has minimum eigenvalue {min_eig:.2e} < {min_eig_threshold:.2e}")
                print(f"Condition number: {condition_number:.2e}")
            
            # Fix by clipping eigenvalues
            eig_vals_fixed = np.maximum(eig_vals, min_eig_threshold)
            eig_vecs = np.linalg.eigh(sample)[1]
            
            # Reconstruct the matrix with clipped eigenvalues
            fixed_matrix = eig_vecs @ np.diag(eig_vals_fixed) @ eig_vecs.T
            
            # Re-normalize to ensure it's still a correlation matrix
            fixed_matrix = BW_projection(fixed_matrix)
            fixed_samples.append(fixed_matrix)
            
            if verbose:
                new_cond = max_eig / min_eig_threshold
                print(f"Fixed condition number: {new_cond:.2e}")
        else:
            fixed_samples.append(sample)
            
    return fixed_samples


###########################################################
# Generation of correlation matrices

def onion_corr(n, d, eta=1, seed=None):
    """
    Onion (LKJ) distribution of correlation matrices using the onion method of 
        Lewandowski et al. (2009) "Generating random correlation matrices based on vines and extended onion method."

    Parameters
    ----------
    n : (int) Number of correlation matrices to generate
    d : (int) Dimension of the correlation matrices
    eta : (float) Concentration parameter for the LKJ distribution (default is 1; uniform sampling)
    seed: (int, None, or np.random.RandomState) Random seed for reproducibility. Can inherit from np.random.RandomState

    Returns
    ----------
    list: A list of generated correlation matrices
    """
    assert d >= 2, "Dimension must be at least 2."
    
    RS = seed if isinstance(seed, np.random.RandomState) else np.random.RandomState(seed)
    
    corr_mats = []
    for _ in range(n):
        ### Initialization
        beta = eta + (d - 2) / 2
        # Sample u ~ Beta(beta, beta) and transform to (-1, 1)
        # note: if d = 2 (beta(1, 1)), equivalent to U(-1, 1) on correlations
        u_sample = RS.beta(beta, beta)
        r12 = 2 * u_sample - 1
        R = get_bivar_corr(r12)

        # Recursively expand R to dimension d (onion method)
        # Loop: k is current dimension of R; new matrix will be of size (k+1) x (k+1)
        for k in range(2, d):
            beta -= 0.5  # update beta 
            
            # Sample y ~ Beta(n/2, beta) 
            y = np.random.beta(n / 2.0, beta)

            # Generate an n-dimensional vector uniformly on the unit sphere in R^k
            u_vec = RS.normal(size=k)
            u_vec /= np.linalg.norm(u_vec)

            # Scale by sqrt(y)
            w_vec = np.sqrt(y) * u_vec

            # Cholesky factor A of the current matrix R (i.e., A @ A.T = R)
            A = np.linalg.cholesky(R)
            # z = A * w: the correlations between the new variable and the existing ones.
            z = A @ w_vec

            # Expand R
            R_new = np.zeros((k + 1, k + 1))
            R_new[:k, :k] = R
            R_new[k, k] = 1.0
            R_new[k, :k] = z
            R_new[:k, k] = z

            R = R_new

        corr_mats.append(R)
    return corr_mats


def autoregrssive(d, rho):
    """
    Correlation matrix from the AR(1) model, A_ij = rho^{|i-j|}

    Parameters
    ----------
    d : (int) Dimension of the correlation matrix
    rho : (float) Correlation parameter, must be between -1 and 1

    Returns
    ----------
    np.ndarray: The generated correlation matrix
    """
    assert d >= 2, "Dimension must be at least 2."
    assert -1 < rho < 1, "Correlation parameter must be between -1 and 1."
    
    corr = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            corr[i, j] = rho ** abs(i - j)
    
    return corr


def autoreg_sample(n, d, seed=None):
    """
    Sample correlation matrices from the AR(1) model

    Parameters
    ----------
    n : (int) Number of correlation matrices to generate
    d : (int) Dimension of the correlation matrices
    seed: (int, None, or np.random.RandomState) Random seed for reproducibility. Can inherit from np.random.RandomState

    Returns
    ----------
    list: A list of generated correlation matrices
    """
    RS = seed if isinstance(seed, np.random.RandomState) else np.random.RandomState(seed)
    
    rho_values = RS.uniform(-0.9, 0.9, size=n)
    
    corr_mats = []
    for rho in rho_values:
        corr_mats.append(autoregrssive(d, rho))
    
    return corr_mats


########################################################
## Riemannian projected gradient descent for correlation barycenter


def BW_projection(A):
    """
    Project a psd matrix A to the nearest correlation matrix with respect to the Bures-Wasserstein metric.
    It is just the normalization.
    
    Parameters
    ----------
    A : np.ndarray
        A matrix to be projected.

    Returns
    ----------
    np.ndarray: The projected correlation matrix.
    """
    # Ensure A is symmetric (helps numerical stability)
    A = (A + A.T) / 2

    # Symmetric normalization
    std_diag = np.sqrt(np.diag(A))
    C = A / np.outer(std_diag, std_diag)

    # Ensure unit diagonal to avoid numerical issues
    np.fill_diagonal(C, 1.0)

    return C

def riemannian_corr_barycenter(samples, weights=None, step_size=1., max_iter=1000, tol=1e-6, 
                               record=True, verbose=True, proj=True):
    """
    Correlation barycenter matrix using the projected Riemannian gradient descent approach.
    Since the Riemannian gradient is the 2 times the Euclidean gradient, we use the same `grad_barycenter_functional` function.
    
    Parameters
    ----------
    samples : list of np.ndarrays or list of correlations (if 2-dimensional)
    weights : list of weights, summing to 1. If None, uniform weights are used.
    step_size : float
        Step size for the gradient descent.
    max_iter : int
        Maximum number of iterations for the gradient descent.
    tol : float
        Tolerance for convergence. If the change in the barycenter matrix is less than this value, the algorithm stops.
    record : bool
        Whether to record the functional values during the iterations.
    verbose : bool
        Whether to print the progress of the algorithm.
    proj : bool, default True
        Whether to project the gradient descent update to correlation matrices
        if false, BW barycenter covariance matrix is computed instead.

    Returns
    ----------
    np.ndarray: the correlation Barycenter matrix
    np.ndarray: the barycenter functional values during the iterations
    """
    if weights is None:
        weights = [1.0 / len(samples)] * len(samples)
    if len(samples) != len(weights):
        raise ValueError("Weights must match the number of samples.")
    if not np.isclose(sum(weights), 1):
        raise ValueError("Weights must sum to 1.")
    if isinstance(samples[0], float) or isinstance(samples[0], int):
        samples = [get_bivar_corr(sample) for sample in samples]

    # Initial guess: just the first sample
    A = samples[0]
    d = A.shape[0]

    # Use closed formula for bivariate correlation case
    if d == 2 and proj:
        if verbose:
            print("Using closed formula for bivariate correlation barycenter.")
        A = barycenter_bivar_corr(samples, weights)
        if record:
            vals = [barycenter_functional(A, samples, weights)]
            return A, vals
        else:
            return A

    # =========================================================================
    # FAST PATH: Use Numba-accelerated version for d >= 3, proj=True, no recording
    # =========================================================================
    if not record and not verbose and proj and d >= 3:
        samples_arr = np.asarray(samples)
        weights_arr = np.asarray(weights)
        return _riemannian_barycenter_numba(samples_arr, weights_arr, step_size, max_iter, tol)

    # =========================================================================
    # SLOW PATH: Full Python version with recording/verbose support
    # =========================================================================
    vals = []
    if record:
        vals.append(barycenter_functional(A, samples, weights))

    start = time.time()
    for i in range(max_iter):
        if verbose:
            if i % 50 == 0:
                print(f"Iteration {i}: Functional value = {vals[-1] if record else 'set record=True'}")

        # Riemannian gradient descent on the tangent space
        grad = grad_barycenter_functional(A, samples, weights)

        # Riemannian retraction on the BW manifold
        S = step_size * (np.eye(d) - grad)
        exp_A_neg_grad = S @ A @ S

        # BW projection to the set of correlation matrices
        A_new = BW_projection(exp_A_neg_grad) if proj else exp_A_neg_grad.copy()
        
        # Check convergence
        if np.linalg.norm(A_new - A) < tol:
            A = A_new
            end = time.time()
            if verbose:
                print(f"Converged at iteration {i} after {end - start:.2f} seconds.")
            break
        
        A = A_new
        if record:
            vals.append(barycenter_functional(A, samples, weights))
    else:
        if verbose:
            end = time.time()
            print(f"Max iteration {max_iter} achieved after {end - start:.2f} seconds.")
    if verbose:
        print(f"Final functional value: {vals[-1] if record else 'set record=True'}")

    if record:
        return A, vals
    
    return A
