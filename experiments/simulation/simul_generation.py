import sys
from pathlib import Path
import numpy as np
from scipy.stats import gamma, norm
from scipy.linalg import expm

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from functions import BW_projection


def marginal_conditional_param(j, beta_j, Z, sigma=3, c=1, fixed_marginal=False, random_state=None):
    """
    Generate the jth marginal parameter theta_j (scale parameter of Gamma distribution)
    
    Parameters:
    -----------
    j : int
        Dimension index (0-based).
    beta_j : array-like
        Coefficients for Z.
    Z : array-like, shape (n, 2)
        Predictor variables.
    sigma : float
        Parameter sigma.
    c : float
        Parameter c.
    fixed_marginal : bool
        If True, use fixed theta = 3.
    random_state : int, RandomState instance or None
    
    Returns:
    --------
    thetas : array-like, shape (n,)
        Scale parameters for Gamma distribution.
    sign : int
        Sign for the distribution (-1)^(j+1).
    """
    RS = random_state if isinstance(random_state, np.random.RandomState) else np.random.RandomState(random_state)
    
    n = Z.shape[0]
    # Paper: (-1)^j Y_j ~ Gamma(2, theta_j).
    # Assuming j is 0-based index from loop, corresponding to 1st, 2nd... dimension.
    # If j=0 (1st dim), sign is -1. If j=1 (2nd dim), sign is +1.
    sign = (-1)**(j + 1)

    if fixed_marginal:
        thetas = np.full(n, 3.0)
    else:
        # theta_j | Z ~ Gamma(shape, scale)
        # shape = (sigma + beta_j^T Z)^2 / c
        # scale = c / (sigma + beta_j^T Z)
        
        # This linear term is always positive over the support of Z
        linear_term = sigma + np.dot(Z, beta_j)
        
        shape_params = (linear_term**2) / c
        scale_params = c / linear_term
        
        thetas = RS.gamma(shape=shape_params, scale=scale_params)
        
    return thetas, sign

def correlation_generation(Z, d, type='linear', std_noise=0.1, fixed_corr=False, random_state=None):
    """
    Generate correlation matrices for n samples.
    """
    RS = random_state if isinstance(random_state, np.random.RandomState) else np.random.RandomState(random_state)
    n = Z.shape[0]
    
    Sigmas = np.zeros((n, d, d))
    
    if fixed_corr:
        for i in range(n):
            Sigmas[i] = np.eye(d)
        return Sigmas

    if d == 2:
        if type == 'linear':
            # rho = 0.3 * Z1 + epsilon
            eps = RS.normal(0, std_noise, size=n)
            rhos = 0.3 * Z[:, 0] + eps
        elif type == 'nonlinear':
            # rho = tanh(2 * Z2 + epsilon)
            eps = RS.normal(0, std_noise, size=n)
            rhos = np.tanh(2 * Z[:, 1] + eps)
        else:
            raise ValueError("type must be 'linear' or 'nonlinear' for d=2.")
            
        for i in range(n):
            rho = rhos[i]
            # Clip rho to be safe
            rho = np.clip(rho, -1., 1.)
            Sigmas[i] = np.array([[1.0, rho], [rho, 1.0]])
            
    elif d > 2: # will only use d==10
        # Use a fixed seed for universally fixed M1, M2 across calls
        RS_fixed = np.random.RandomState(42)
        
        def generate_M(rng):
            M = rng.uniform(-0.5, 0.5, size=(d, d))
            M = (M + M.T) / 2
            np.fill_diagonal(M, 0)
            return M
            
        M1 = generate_M(RS_fixed)
        M2 = generate_M(RS_fixed)

        for i in range(n):
            # E_i random symmetric noise
            E = RS.normal(0, std_noise, size=(d, d))
            E = (E + E.T) / 2
            np.fill_diagonal(E, 0)
            
            Log_Sigma = Z[i, 0] * M1 + Z[i, 1] * M2 + E
            Sigma_raw = expm(Log_Sigma)
            
            # Project to correlation
            Sigma_proj = BW_projection(Sigma_raw)
                
            Sigmas[i] = Sigma_proj
            
    return Sigmas


def nonparanormal_generation(N, thetas, signs, latent_cor, random_state=None):
    """
    Generate N empirical samples of a d-dimensional nonparanormal distribution
    from a given sequence of marginal parameters and a correlation matrix.
    """
    RS = random_state if isinstance(random_state, np.random.RandomState) else np.random.RandomState(random_state)
    d = len(thetas)
    
    # 1. Generate latent Gaussian
    Z_latent = RS.multivariate_normal(mean=np.zeros(d), cov=latent_cor, size=N)
    
    # 2. Transform to Uniform (pass through the normal CDF dimension-wise)
    U = norm.cdf(Z_latent)
    
    # 3. Apply marginal quantile functions
    X = np.zeros((N, d))
    for j in range(d):
        theta = thetas[j]
        sign = signs[j]
        
        # (-1)^j Y_j ~ Gamma(2, theta)
        # If sign is 1: Y_j ~ Gamma(2, theta) -> Y_j = gamma.ppf(U, ...)
        # If sign is -1: -Y_j ~ Gamma(2, theta) -> Y_j = -gamma.ppf(1-U, ...)
        
        if sign == 1:
            X[:, j] = gamma.ppf(U[:, j], a=2, scale=theta)
        else:
            X[:, j] = -gamma.ppf(1 - U[:, j], a=2, scale=theta)
            
    return X

def joint_signal_generation(n, d, N=None, N0=None, M=200, type='linear', random_state=None, test=False):
    """
    Generate N empirical samples of a d-dimensional joint distribution.
    If test=True, only return the joint distribution information.
    """
    if N is None:
        if N0 is None:
            N = 100 * d # Default behavior if neither is provided
        else:
            N = N0 * d
    
    RS = random_state if isinstance(random_state, np.random.RandomState) else np.random.RandomState(random_state)

    # Generate Z
    Z = RS.uniform(-1, 1, size=(n, 2))
    
    # Define betas
    betas = []
    betas.append(np.array([0.5, 0.0]))
    betas.append(np.array([0.4, -0.3]))
    
    if d > 2:
        # Generate additional betas universally fixed across calls
        RS_fixed = np.random.RandomState(42)
        for _ in range(d - 2):
            betas.append(RS_fixed.uniform(-0.5, 0.5, size=2))
            
    # Generate marginal parameters
    all_thetas = np.zeros((n, d))
    all_signs = np.zeros(d)
    
    for j in range(d):
        # Use correct beta index. For d=10, we have d betas.
        beta = betas[j] if j < len(betas) else np.zeros(2)
        thetas, sign = marginal_conditional_param(j, beta, Z, random_state=RS)
        all_thetas[:, j] = thetas
        all_signs[j] = sign
        
    # Generate correlations
    Sigmas = correlation_generation(Z, d, type=type, random_state=RS)
    joint_distribution = {
        'Predictor': Z,
        'Marginal_Params': all_thetas,
        'Latent_Corrs': Sigmas
    }

    if test:
        if isinstance(M, int):
            alphas = np.linspace(0, 1, M)
        else:
            alphas = M
        quantiles = np.zeros((n, d, len(alphas)))
        for i in range(n):
            for j in range(d):
                theta = all_thetas[i, j]
                sign = all_signs[j]
                if sign == 1:
                    quantiles[i, j, :] = gamma.ppf(alphas, a=2, scale=theta)
                else:
                    quantiles[i, j, :] = -gamma.ppf(1 - alphas, a=2, scale=theta)
        joint_distribution['Marginal_Quantiles'] = quantiles
        return joint_distribution

    # Generate data
    data = np.zeros((n, N, d))
    for i in range(n):
        data[i] = nonparanormal_generation(N, all_thetas[i], all_signs, Sigmas[i], random_state=RS)

    return data, joint_distribution


def corr_signal_generation(n, N0, M=200, type='linear', random_state=None, test=False):
    """
    Generate N = N0 * d empirical samples of a d-dimensional correlation signal distribution.
    d=2.
    """
    d = 2
    N = N0 * d
    RS = random_state if isinstance(random_state, np.random.RandomState) else np.random.RandomState(random_state)

    Z = RS.uniform(-1, 1, size=(n, 2))
    
    # Fixed marginals
    all_thetas = np.full((n, d), 3.0)
    all_signs = np.array([(-1)**(j+1) for j in range(d)])
    
    Sigmas = correlation_generation(Z, d, type=type, random_state=RS)
    
    
    joint_distriubtion = {
        'Predictor': Z,
        'Marginal_Params': all_thetas,
        'Latent_Corrs': Sigmas
    }

    if test:
        if isinstance(M, int):
            alphas = np.linspace(0, 1, M)
        else:
            alphas = M
        quantiles = np.zeros((n, d, len(alphas)))
        for i in range(n):
            for j in range(d):
                theta = all_thetas[i, j]
                sign = all_signs[j]
                if sign == 1:
                    quantiles[i, j, :] = gamma.ppf(alphas, a=2, scale=theta)
                else:
                    quantiles[i, j, :] = -gamma.ppf(1 - alphas, a=2, scale=theta)
        joint_distriubtion['Marginal_Quantiles'] = quantiles
        return joint_distriubtion

    data = np.zeros((n, N, d))
    for i in range(n):
        data[i] = nonparanormal_generation(N, all_thetas[i], all_signs, Sigmas[i], random_state=RS)
        
    return data, joint_distriubtion


def indep_marg_generation(n, N0, M=200, random_state=None, test=False):
    """
    Generate N = N0 * d empirical samples of a d-dimensional independent marginal distribution.
    d=2.
    """
    d = 2
    N = N0 * d
    RS = random_state if isinstance(random_state, np.random.RandomState) else np.random.RandomState(random_state)

    Z = RS.uniform(-1, 1, size=(n, 2))
    
    betas = [np.array([0.5, 0.0]), np.array([0.4, -0.3])]
    
    all_thetas = np.zeros((n, d))
    all_signs = np.zeros(d)
    
    for j in range(d):
        thetas, sign = marginal_conditional_param(j, betas[j], Z, random_state=RS)
        all_thetas[:, j] = thetas
        all_signs[j] = sign
        
    Sigmas = correlation_generation(Z, d, fixed_corr=True, random_state=RS)
    
    
    joint_distriubtion = {
        'Predictor': Z,
        'Marginal_Params': all_thetas,
        'Latent_Corrs': Sigmas
    }

    if test:
        if isinstance(M, int):
            alphas = np.linspace(0, 1, M)
        else:
            alphas = M
        quantiles = np.zeros((n, d, len(alphas)))
        for i in range(n):
            for j in range(d):
                theta = all_thetas[i, j]
                sign = all_signs[j]
                if sign == 1:
                    quantiles[i, j, :] = gamma.ppf(alphas, a=2, scale=theta)
                else:
                    quantiles[i, j, :] = -gamma.ppf(1 - alphas, a=2, scale=theta)
        joint_distriubtion['Marginal_Quantiles'] = quantiles
        return joint_distriubtion

    data = np.zeros((n, N, d))
    for i in range(n):
        data[i] = nonparanormal_generation(N, all_thetas[i], all_signs, Sigmas[i], random_state=RS)
        
    return data, joint_distriubtion
