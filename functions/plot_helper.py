import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde, norm
from scipy.interpolate import interp1d

# Standard normal CDF for convenience
standard_normal_cdf = norm.cdf


def plot_multivar_distributions(multivar_df, axes_names, all_features, 
                                distribution_info=None, n_cols=5, vmax=None, lcor=True,
                                cmap='bone_r', heatmap=True, x_range=None, y_range=None,
                                figsize=None, title_fontsize=12, label_fontsize=12,
                                grid_resolution=100, kde_bandwidth=None, 
                                colorbar_label=None, colorbar=True, vline=True,
                                suptitle=False, show=True, figax=None,
                                auto_sample=True, sample_n=3000, sample_seed=None):
    """   
    Plot bivariate distributions of selected two axes from the given `multivar_df` DataFrame.
    Gaussian KDE is used for density estimation, and the plot can be either a heatmap or contour plot.

    Parameters:
    ----------
    multivar_df (pd.DataFrame): DataFrame containing the column 'multivar_distribution', consisting of np.arrays of multivariate distributions.
    axes_names (list): List of two feature names to plot.
    all_features (list): List of all feature names in the multivariate distribution.
    distribution_info (list): List of names of the information on the plotted distributions
        Names must be contained in the columns of `mutlivar_df`.
        If None, defaults to the first column name in `multivar_df`.
    n_cols (int): Number of columns for the subplot layout. Default is 5.
    vmax (float): Maximum value for density plotting. If None, it is automatically determined from the data.
    lcor (bool): Whether display the latent correlation in the plot title. Default is True.
    cmap (str): Colormap for the plot. Default is 'bone_r'.
    heatmap (bool): If True, use heatmap for density plotting. If not, contour is plotted.
    x_range (tuple): Range for the x-axis. Default is None (automatically determined).
    y_range (tuple): Range for the y-axis. Default is None (automatically determined).
    figsize (tuple): Figure size. If None, automatically determined based on layout.
    title_fontsize (int): Font size for subplot titles. Default is 12.
    label_fontsize (int): Font size for axis labels. Default is 12.
    grid_resolution (int): Resolution of the density estimation grid. Default is 100.
    kde_bandwidth (float): Bandwidth for KDE. If None, uses scipy's default.
    colorbar_label (str): Label for the colorbar. If None, uses default.
    auto_sample (bool): If True, allow local sampling from `marginal_fits`/`latentcor_fits`
        when `multivar_distribution` is not already present.
    sample_n (int): Number of samples to generate when `auto_sample=True`.
    sample_seed (int or np.random.RandomState): Random seed used only for auto-sampling.

    Returns:
    --------
    fig: matplotlib.figure.Figure
        The matplotlib figure object
    """
    
    # Input validation
    _validate_inputs(multivar_df, axes_names, all_features, distribution_info, auto_sample)

    plot_df = _prepare_plot_dataframe(
        multivar_df,
        auto_sample=auto_sample,
        sample_n=sample_n,
        sample_seed=sample_seed,
    )

    # Extract bivariate distributions
    bivar_distributions = _extract_bivariate_data(plot_df, axes_names, all_features)
    
    # Determine axis ranges
    x_min, x_max, y_min, y_max = _get_axis_ranges(bivar_distributions, x_range, y_range)
    
    # Create density grid
    xx, yy, positions = _create_density_grid(x_min, x_max, y_min, y_max, grid_resolution)
    
    # Calculate all KDE densities efficiently
    zz_list = _calculate_kde_densities(bivar_distributions, positions, xx, kde_bandwidth)
    
    # Determine color scale
    vmin = 0.0
    if vmax is None:
        vmax = _determine_vmax(zz_list)
    
    # Setup figure and axes (if not None, must be generated from plt.subplots)
    if figax is None:
        fig, axes = _setup_figure_axes(len(bivar_distributions), n_cols, figsize)
    else:
        fig, axes = figax

    # Create plots
    _create_density_plots(axes, xx, yy, zz_list, axes_names, plot_df, 
                         distribution_info, all_features, heatmap, cmap, 
                         vmin, vmax, lcor, x_min, x_max, y_min, y_max,
                         title_fontsize, label_fontsize, n_cols, vline)
    
    # Add colorbar
    if len(bivar_distributions) > 0 and colorbar:
        _add_colorbar(fig, axes, zz_list[0] if zz_list else None, cmap, vmin, vmax, 
                     colorbar_label or r'Density $(\times 10^{-3})$')

    # Needs axes adjustments (later)    
    # if suptitle:
    #     fig.suptitle(f"Bivariate Distributions of {axes_names[0]} and {axes_names[1]}", fontsize=title_fontsize + 2)

    if show:
        fig.tight_layout(rect=[0, 0, 0.93, 1])
        fig.show()
    else:
        return fig


# Helper functions for the improved plot_multivar_distributions
def _validate_inputs(multivar_df, axes_names, all_features, distribution_info, auto_sample):
    """Validate function inputs."""
    if len(axes_names) != 2:
        raise ValueError("axes_names must contain exactly two elements.")
    if not all(name in all_features for name in axes_names):
        raise ValueError("Both axes_names must be in the all_features list.")
    has_distributions = 'multivar_distribution' in multivar_df.columns
    has_sampling_inputs = {'marginal_fits', 'latentcor_fits'}.issubset(multivar_df.columns)
    if not has_distributions and not has_sampling_inputs:
        raise ValueError(
            "multivar_df must contain 'multivar_distribution' or both "
            "'marginal_fits' and 'latentcor_fits' columns."
        )
    if not has_distributions and not auto_sample:
        raise ValueError(
            "'multivar_distribution' is required when auto_sample=False."
        )


def _prepare_plot_dataframe(multivar_df, auto_sample, sample_n, sample_seed):
    """Build a local plotting dataframe without mutating notebook state."""
    plot_df = multivar_df.copy()

    if 'multivar_distribution' not in plot_df.columns and auto_sample:
        plot_df['multivar_distribution'] = sample_predictions(
            plot_df,
            n_samples=sample_n,
            seed=sample_seed,
            inplace=False,
        )

    return plot_df



def _extract_bivariate_data(multivar_df, axes_names, all_features):
    """Extract bivariate distributions for specified axes."""
    bivar_distributions = multivar_df['multivar_distribution'].copy()
    indices = [all_features.index(name) for name in axes_names]
    
    if len(all_features) != 2:
        bivar_distributions = bivar_distributions.apply(
            lambda x: np.array(x)[:, indices] if len(x) > 0 else np.empty((0, 2))
        )
    
    return bivar_distributions


def _get_axis_ranges(bivar_distributions, x_range, y_range):
    """Determine axis ranges from data or use provided ranges."""
    if x_range is None or y_range is None:
        x_values, y_values = [], []
        for dist in bivar_distributions:
            if len(dist) > 0:
                x_values.extend(dist[:, 0])
                y_values.extend(dist[:, 1])
        
        x_min = np.min(x_values) if x_values and x_range is None else x_range[0]
        x_max = np.max(x_values) if x_values and x_range is None else x_range[1]
        y_min = np.min(y_values) if y_values and y_range is None else y_range[0]
        y_max = np.max(y_values) if y_values and y_range is None else y_range[1]
    else:
        x_min, x_max = x_range
        y_min, y_max = y_range
    
    return x_min, x_max, y_min, y_max


def _create_density_grid(x_min, x_max, y_min, y_max, resolution):
    """Create grid for density estimation."""
    x_grid = np.linspace(x_min, x_max, resolution)
    y_grid = np.linspace(y_min, y_max, resolution)
    xx, yy = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([xx.ravel(), yy.ravel()])
    return xx, yy, positions


def _calculate_kde_densities(bivar_distributions, positions, xx, bandwidth):
    """Calculate KDE densities for all distributions efficiently."""
    zz_list = []
    for dist in bivar_distributions:
        if len(dist) < 2:
            zz_list.append(np.zeros_like(xx))
            continue
        
        try:
            values = np.vstack([dist[:, 0], dist[:, 1]])
            if bandwidth is not None:
                kernel = gaussian_kde(values, bw_method=bandwidth)
            else:
                kernel = gaussian_kde(values)
            zz = np.reshape(kernel(positions).T, xx.shape) * 1e3
        except (np.linalg.LinAlgError, ValueError):
            zz = np.zeros_like(xx)
        
        zz_list.append(zz)
    
    return zz_list


def _determine_vmax(zz_list):
    """Determine maximum value for color scaling."""
    max_vals = [np.max(zz) for zz in zz_list if np.max(zz) > 0]
    return np.max(max_vals) if max_vals else 1.0


def _setup_figure_axes(n_distributions, n_cols, figsize):
    """Setup figure and axes layout."""
    if n_distributions <= n_cols:
        n_rows, n_cols_actual = 1, n_distributions
    else:
        n_rows = (n_distributions + n_cols - 1) // n_cols
        n_cols_actual = n_cols
    
    if figsize is None:
        figsize = (n_cols_actual * 4, n_rows * 4)
    
    fig, axes = plt.subplots(n_rows, n_cols_actual, figsize=figsize)
    
    # Normalize axes to always be a list
    if n_distributions == 1:
        axes = [axes]
    elif n_rows == 1 and n_distributions > 1:
        axes = list(axes) if hasattr(axes, '__len__') else [axes]
    elif n_rows > 1:
        axes = axes.flatten()
    
    return fig, axes


def _create_density_plots(axes, xx, yy, zz_list, axes_names, multivar_df, 
                         distribution_info, all_features, heatmap, cmap, 
                         vmin, vmax, lcor, x_min, x_max, y_min, y_max,
                         title_fontsize, label_fontsize, n_cols, vline):
    """Create individual density plots."""
    n_distributions = len(zz_list)
    
    # Setup distribution info
    if distribution_info is None:
        distribution_info = [multivar_df.columns[0]]
    elif isinstance(distribution_info, str):
        distribution_info = [distribution_info]
    
    for i in range(n_distributions):
        ax = axes[i] if n_distributions > 1 else axes[0]
        
        # Clip values to prevent density exceeding vmax (just for clipped visualization purpose)
        zz_clipped = np.clip(zz_list[i], vmin, vmax)

        # Create plot
        if heatmap:
            cf = ax.pcolormesh(xx, yy, zz_clipped, shading='auto', cmap=cmap, 
                             vmin=vmin, vmax=vmax)
        else:
            levels = np.linspace(vmin, vmax, 13)
            cf = ax.contourf(xx, yy, zz_clipped, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels)
        
        # Add special lines for glucose plots
        if axes_names[0] == 'Mean':
            if axes_names[1] == 'tir70-180' and vline:
                ax.axvline(x=70, color='k', linestyle='--', linewidth=0.5, alpha=0.7)
                ax.axvline(x=180, color='k', linestyle='--', linewidth=0.5, alpha=0.7)
        
        # Create title
        title_parts = []
        for info_name in distribution_info:
            if info_name in multivar_df.columns:
                value = multivar_df[info_name].iloc[i]
                if isinstance(value, (int, float)):
                    title_parts.append(f"{info_name}: {value:.2f}")
                else:
                    title_parts.append(f"{info_name}: {value}")
        
        title_str = ', '.join(title_parts)
        
        # Add latent correlation if requested
        if lcor:
            indices = [all_features.index(name) for name in axes_names]
            lcorr_name = 'latentcor' if 'latentcor' in multivar_df.columns else 'latentcor_fits'
            if lcorr_name in multivar_df.columns:
                lcor_val = multivar_df[lcorr_name].iloc[i][indices[0], indices[1]]
                title_str += f" (Lcor = {lcor_val:.3f})"
        
        ax.set_title(title_str, fontsize=title_fontsize)
        ax.set_xlabel(axes_names[0], fontsize=label_fontsize)
        if i % n_cols == 0:
            ax.set_ylabel(axes_names[1], fontsize=label_fontsize)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    
    # Hide unused subplots
    for i in range(n_distributions, len(axes)):
        axes[i].set_visible(False)


def _add_colorbar(fig, axes, sample_plot, cmap, vmin, vmax, label):
    """Add colorbar to the figure."""
    if sample_plot is not None:
        cbar_ax = fig.add_axes([0.937, 0.15, 0.012, 0.7])
        # Create a dummy mappable for the colorbar
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        
        sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label(label)



# Visualize how the latent correlation matrix changes with the predictor value
# Specialized to single predictor variable
def plot_correlation_trends(result, predictor_name, feature_names=None, show=True):
    """
    Plot how each pairwise correlation changes with the predictor value
    
    Parameters:
    ----------
    result: dict
        The output of the `multivariate_frechet_regression` function.
    predictor_name: str
        Name of the predictor variable for which the correlation trends are plotted.
    feature_names: list, optional
        List of feature names to be used in the plot.
        If None, it will use all of the feature names from the result.
    """
    # Extract latent correlation matrices
    cor_matrices = np.array(result['fits']['latentcor_fits'].to_list())  # Shape: (m, d, d)

    # Feature names for plotting correlations    
    all_features = result['feature_names']
    feature_names = all_features if feature_names is None else feature_names

    # Check if the feature names are valid
    if not all(name in all_features for name in feature_names):
        raise ValueError("Some feature names are not present in the result's feature names: {}".format(all_features))

    # Extract the feature indices for the specified feature names
    feature_indices = [all_features.index(name) for name in feature_names]
    # Extract the correlation matrices for the specified feature names
    cor_matrices = cor_matrices[:, feature_indices, :][:, :, feature_indices]

    # Check if the shape is correct
    if cor_matrices.shape[1] != len(feature_indices):
        raise ValueError(f"Expected shape (m, {len(feature_indices)}, {len(feature_indices)}) but got {cor_matrices.shape}")

    length = cor_matrices.shape[1]

    # Create pairs of indices for the upper triangular part of correlation matrices
    pairs = [(i, j) for i in range(length) for j in range(i+1, length)]
    pair_names = [f'{feature_names[i]} vs {feature_names[j]}' for i, j in pairs]
    
    # Extract correlation values for each pair across all predictor values
    cor_values = {pair_name: [cor_matrices[k, i, j] for k in range(len(result['fits'][predictor_name]))] 
                  for (i, j), pair_name in zip(pairs, pair_names)}
    
    # Create DataFrame for plotting
    plot_data = []
    for pair_name, values in cor_values.items():
        for idx, val in enumerate(values):
            plot_data.append({
                'Predictor': result['fits'][predictor_name][idx],
                'Correlation': val,
                'Pair': pair_name
            })
    plot_df = pd.DataFrame(plot_data)
    
    # Include r^2 value if available
    if 'R2_latentcor' in result:
        r2_value = r" ($R^2$ = {:.3f})".format(result['R2_latentcor'])
    else:
        r2_value = ""

    # If the quadratic feature is used, add it to the xlabel
    xlab = ""
    if f"{predictor_name}^2" in result['fits']:
        xlab += ' (x, x^2)-embedding'

    # Plot the correlation trends
    # plt.figure(figsize=(7, 5))
    sns.lineplot(data=plot_df, x='Predictor', y='Correlation', hue='Pair', style='Pair', markers=True, dashes=True, markersize=9)
    plt.title(r'Predicted Latent Correlation with {}'.format(predictor_name) + r2_value, fontsize=14)
    plt.xlabel(predictor_name + xlab, fontsize=12)
    plt.ylabel('Latent Correlation', fontsize=12)
    # plt.grid(True, alpha=0.3)
    plt.legend(title='Feature Pair')
    if show:
        plt.tight_layout()
        plt.show()
    
    return plot_df




##############################################################
# Sampling function from the fitted Frechet regression

def sample_predictions(frechet_fit, n_samples=3000, seed=None, inplace=False):
    """
    Monte Carlo sampling from the proxy Frechet regression fit distributions (visualization purpose).

    Parameters:
    ----------
    frechet_fit : pd.DataFrame
        The output of the frechet_proxy function. 
        Must contain 'marginal_fits' and 'latentcor_fits' columns
    n_samples : int
        Number of samples to generate.
    seed : int or np.random.RandomState
        Random seed for reproducibility.
    inplace : bool
        If True, attach the sampled arrays to `frechet_fit['multivar_distribution']`.
        
    Returns:
    --------
    List of numpy.ndarray
        List of Monte Carlo samples from the fitted Frechet regression model
    """

    RS = seed if isinstance(seed, np.random.RandomState) else np.random.RandomState(seed)

    # Extract the marginal predictions and latent correlation matrices
    # marginals = frechet_fit['Marginals']
    # latent_cors = frechet_fit['LatentCor']
    marginals = frechet_fit['marginal_fits']
    latent_cors = frechet_fit['latentcor_fits']

    # Sampe for each prediction point of Z
    samples = []
    for i in range(len(latent_cors)):
        # i-th sample
        fitted_quantiles_ith_sample = []
        for j in range(len(marginals[i])):
            # j-th coordinate of distribution
            q = interp1d(np.linspace(0, 1, len(marginals[i][j])), marginals[i][j], 
                         bounds_error=False, fill_value="extrapolate")
            fitted_quantiles_ith_sample.append(q)
        
        latent_gaussian = RS.multivariate_normal(mean=np.zeros(len(marginals[i])),
                                                  cov=latent_cors[i], size=n_samples)
        
        prediction_sampling_ith = np.zeros_like(latent_gaussian)
        for j in range(len(marginals[0])):
            # Apply the standard normal CDF to the latent samples
            prediction_sampling_ith[:, j] = fitted_quantiles_ith_sample[j](
                standard_normal_cdf(latent_gaussian[:, j]))

        samples.append(prediction_sampling_ith)

    if inplace:
        # This is for visualization purposes when the caller explicitly wants the column attached.
        frechet_fit['multivar_distribution'] = samples

    return samples
