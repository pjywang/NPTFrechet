import os
import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import ot
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from scipy.stats import norm

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Simulation Parameters (matched with Simulation.py)
n_values = [50, 100, 200]
d_values = [2, 10]
N_values = [100, 1000]
types = ['linear', 'nonlinear']
reps = 100

def _parse_int_list_env(var_name, default_list):
    """Parse comma-separated integers from environment variable.

    Example: N_VALUES="50, 100, 200"
    """
    raw = os.environ.get(var_name, "").strip()
    if raw == "":
        return default_list
    try:
        return [int(x.strip()) for x in raw.split(",") if x.strip() != ""]
    except Exception as e:
        raise ValueError(f"Failed to parse {var_name}={raw!r} as comma-separated ints") from e


def _parse_str_list_env(var_name, default_list):
    """Parse comma-separated strings from environment variable."""
    raw = os.environ.get(var_name, "").strip()
    if raw == "":
        return default_list
    return [x.strip() for x in raw.split(",") if x.strip() != ""]

def sample_from_npt(n_samples, marginal_quantiles, latent_corr, quantile_grid, random_state=None):
    """
    Generate empirical samples from NPT fitted model.
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    marginal_quantiles : numpy.ndarray
        Shape (d, M). Quantiles for each dimension.
    latent_corr : numpy.ndarray
        Shape (d, d). Latent correlation matrix.
    quantile_grid : numpy.ndarray
        Shape (M,). Grid points for quantiles (alphas).
    random_state : np.random.RandomState, optional
    
    Returns
    -------
    X : numpy.ndarray
        Shape (n_samples, d). Generated samples.
    """
    if random_state is None:
        random_state = np.random.RandomState()
        
    d = latent_corr.shape[0]
    
    # 1. Sample from latent Gaussian
    Z = random_state.multivariate_normal(mean=np.zeros(d), cov=latent_corr, size=n_samples)
    
    # 2. Transform to Uniform
    U = norm.cdf(Z)
    
    # 3. Transform to Marginals using inverse CDF (quantile function)
    X = np.zeros((n_samples, d))
    for j in range(d):
        # Interpolate quantile function
        # marginal_quantiles[j] corresponds to quantile_grid
        # We want to find values for U[:, j]
        X[:, j] = np.interp(U[:, j], quantile_grid, marginal_quantiles[j])
        
    return X

def sample_from_gaussian(n_samples, mean, cov, random_state=None):
    """
    Generate empirical samples from Gaussian fitted model.
    """
    if random_state is None:
        random_state = np.random.RandomState()
        
    return random_state.multivariate_normal(mean=mean, cov=cov, size=n_samples)

def process_single_file(file_path, n_samples=2000):
    """
    Process a single pickle file: calculate Wasserstein distance for NPT and Gaussian fits.
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        # In parallel evaluation, missing/corrupted files shouldn't crash the whole run.
        # Return an empty list so downstream flattening is safe.
        print(f"Error loading {file_path}: {e}")
        return []

    params = data['params']
    npt_fit = data['npt_fit']
    gaussian_fit = data['gaussian_fit']
    test_data = data['test_data']
    quantile_grid = data['quantile_grid']
    
    # Extract test ground truth
    true_quantiles = test_data['Marginal_Quantiles']
    true_corrs = test_data['Latent_Corrs']
    
    n_test = true_quantiles.shape[0]
    d = true_quantiles.shape[1]
    
    # Extract fitted models
    npt_marginals = npt_fit['Marginals'] # List of length d, each (n_test, M)
    npt_corrs = npt_fit['LatentCor'] # (n_test, d, d)
    
    # Gaussian: 'means' (n_test, d), 'covs' (n_test, d, d)
    gauss_means = gaussian_fit['means']
    gauss_covs = gaussian_fit['covs']
    
    # Calculate Wasserstein distances
    npt_wd_list = []
    marg_wd_list = []
    gauss_wd_list = []
    
    # Use a fixed seed for reproducibility of evaluation sampling
    eval_seed = (params['rep'] * 10000 + params['n']) % (2**32 - 1)
    RS = np.random.RandomState(eval_seed)
    
    for i in range(n_test):
        # 1. Generate Ground Truth Samples
        # True marginals are given as quantiles.
        # True correlation is given.
        # We can use sample_from_npt logic for ground truth as well since we have quantiles.
        # Note: true_quantiles[i] is (d, M)
        X_true = sample_from_npt(n_samples, true_quantiles[i], true_corrs[i], quantile_grid, random_state=RS)
        
        # 2. Generate NPT Fitted Samples
        # Construct (d, M) array for this test point
        curr_npt_marginals = np.array([npt_marginals[j][i] for j in range(d)])
        X_npt = sample_from_npt(n_samples, curr_npt_marginals, npt_corrs[i], quantile_grid, random_state=RS)
        
        # 3. Generate Marginal-FR Fitted Samples (Latent Corr = Identity)
        # Use same marginals as NPT, but identity correlation
        X_marg = sample_from_npt(n_samples, curr_npt_marginals, np.eye(d), quantile_grid, random_state=RS)

        # 4. Generate Gaussian Fitted Samples
        X_gauss = sample_from_gaussian(n_samples, gauss_means[i], gauss_covs[i], random_state=RS)
        
        # Compute Wasserstein Distance (Squared)
        # Uniform weights for ot.emd2
        a, b = np.ones((n_samples,)) / n_samples, np.ones((n_samples,)) / n_samples
        
        # Metric: Squared Euclidean
        M_npt = ot.dist(X_true, X_npt, metric='sqeuclidean')
        wd_npt = ot.emd2(a, b, M_npt)
        npt_wd_list.append(wd_npt)
        
        M_marg = ot.dist(X_true, X_marg, metric='sqeuclidean')
        wd_marg = ot.emd2(a, b, M_marg)
        marg_wd_list.append(wd_marg)

        M_gauss = ot.dist(X_true, X_gauss, metric='sqeuclidean')
        wd_gauss = ot.emd2(a, b, M_gauss)
        gauss_wd_list.append(wd_gauss)
        
    # Average over test points
    npt_mspe = np.mean(npt_wd_list)
    marg_mspe = np.mean(marg_wd_list)
    gauss_mspe = np.mean(gauss_wd_list)
    
    # Construct result rows
    # We want to match the structure of simulation_results.csv but with 'Wasserstein' metric
    actual_N = params.get('N', params.get('N0', 0) * params.get('d', 0))
    
    res_npt = {
        'rep': params['rep'], 'n': params['n'], 'd': params['d'], 'N': actual_N, 
        'type': params['type'], 'method': 'NPT-FR', 'metric': 'Wasserstein', 'value': npt_mspe
    }
    
    res_marg = {
        'rep': params['rep'], 'n': params['n'], 'd': params['d'], 'N': actual_N, 
        'type': params['type'], 'method': 'Marginal-FR', 'metric': 'Wasserstein', 'value': marg_mspe
    }

    res_gauss = {
        'rep': params['rep'], 'n': params['n'], 'd': params['d'], 'N': actual_N, 
        'type': params['type'], 'method': 'Gaussian-FR', 'metric': 'Wasserstein', 'value': gauss_mspe
    }
    
    return [res_npt, res_marg, res_gauss]

def plot_wasserstein_results(df):
    """
    Plot Wasserstein distance results.
    """
    df = df.copy()

    # Create a combined column for (n, N)
    df['n_N'] = df.apply(lambda x: f"n={x['n']}\nN={x['N']}", axis=1)
    
    # Sort to ensure correct plotting order (n major, N minor)
    df = df.sort_values(['n', 'N'])
    
    # Define settings (Columns)
    settings = [
        {'name': 'd=2, Linear', 'query': 'd == 2 and type == "linear"'},
        {'name': 'd=2, Nonlinear', 'query': 'd == 2 and type == "nonlinear"'},
        {'name': 'd=10', 'query': 'd == 10'}
    ]
    
    # Setup plot
    sns.set_theme(style="whitegrid")
    # 1 Row, 3 Columns
    fig, axes = plt.subplots(nrows=1, ncols=len(settings), 
                             figsize=(5 * len(settings), 5), 
                             sharex='col', sharey='row', squeeze=False)
    
    for j, setting in enumerate(settings):
        subset_setting = df.query(setting['query'])
        
        if subset_setting.empty:
            print(f"No data for setting: {setting['name']}")
            continue
            
        ax = axes[0, j]
        
        # Ensure sorting for this specific subset
        subset_setting = subset_setting.sort_values(['n', 'N'])

        sns.boxplot(
            data=subset_setting,
            x='n_N',
            y='value',
            hue='method',
            ax=ax,
            showfliers=False 
        )
        
        # Titles
        ax.set_title(setting['name'])
        
        # Remove x label
        ax.set_xlabel("")

        # Y Labels (only on first column)
        if j == 0:
            ax.set_ylabel("Wasserstein Distance (Squared)")
        else:
            ax.set_ylabel("")
            
        # Legend handling
        # Only show legend in the last column, or outside
        if j == len(settings) - 1:
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        else:
            if ax.get_legend():
                ax.get_legend().remove()

    plt.tight_layout()
    
    # Save
    output_dir = REPO_ROOT / 'results' / 'simulations'
    os.makedirs(output_dir, exist_ok=True)
    plot_path = output_dir / 'boxplot_wasserstein.png'
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"Saved Wasserstein plot to {plot_path}")

def main(n_samples=2000, n_jobs=10):
    fitted_dir = REPO_ROOT / 'results' / 'simulations' / 'fitted_objects'
    if not fitted_dir.exists():
        print(f"Directory not found: {fitted_dir}")
        return

    n_values_eff = _parse_int_list_env("N_VALUES", n_values)
    d_values_eff = _parse_int_list_env("D_VALUES", d_values)
    # Due to the shell environment, variable names are changed as
    # n -> N, N -> N0.
    N_values_eff = _parse_int_list_env("N0_VALUES", N_values)
    types_eff = _parse_str_list_env("TYPES", types)

    # Generate expected file paths based on simulation settings
    files = []
    for d in d_values_eff:
        for n in n_values_eff:
            for N in N_values_eff:
                for type_name in types_eff:
                    if d == 10 and type_name == 'nonlinear':
                        continue  # d=10 runs only one setting
                    for rep in range(reps):
                        filename = f'fit_n{n}_d{d}_N{N}_{type_name}_rep{rep}.pkl'
                        filepath = fitted_dir / filename
                        files.append(filepath)
    
    print(f"Starting Wasserstein evaluation on {len(files)} files with n_jobs={n_jobs}...")

    results_flat = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_single_file)(f, n_samples) for f in files
    )
    
    # Flatten list (be defensive in case any worker returns None)
    all_results = [item for sublist in results_flat if sublist for item in sublist]
    
    if not all_results:
        print("No results generated.")
        return

    df_new = pd.DataFrame(all_results)
    
    output_dir = REPO_ROOT / 'results' / 'simulations'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save separate file (include overrides in filename so multiple jobs can be merged later)
    suffix_parts = []
    if n_values_eff != n_values:
        suffix_parts.append("n" + "-".join(map(str, n_values_eff)))
    if d_values_eff != d_values:
        suffix_parts.append("d" + "-".join(map(str, d_values_eff)))
    if N_values_eff != N_values:
        suffix_parts.append("N" + "-".join(map(str, N_values_eff)))
    if types_eff != types:
        suffix_parts.append("t" + "-".join(types_eff))
    suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""
    new_csv_path = output_dir / f'simulation_results_wasserstein{suffix}.csv'
    df_new.to_csv(new_csv_path, index=False)
    print(f"Wasserstein results saved to {new_csv_path}")
    
    return df_new


if __name__ == "__main__":
    df = main(n_samples=2000, n_jobs=-1)
    
    # csv_path = REPO_ROOT / 'results' / 'simulations' / 'simulation_results_wasserstein.csv'
    # if os.path.exists(csv_path):
    #     print(f"Loading existing results from {csv_path}")
    #     df = pd.read_csv(csv_path)
    #     plot_wasserstein_results(df)
