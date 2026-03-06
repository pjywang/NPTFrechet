import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simul_generation import joint_signal_generation
from gaussian_frechet import gaussian_frechetreg, get_nonparanormal_mse
from functions.regression import npt_frechetreg

# Configuration
SEED = 20251225 # Use SEED ** 2 for bivariate further simulations
M = 200
quantile_grid = np.linspace(0, 1, M + 2)[1:-1]
TYPE_SEED_OFFSET = {
    'linear': 100000000,
    'nonlinear': 200000000,
}

# Simulation Parameters
n_values = [50, 100, 200]
d_values = [2, 10]
N_values = [100, 1000] # Total samples N
types = ['linear', 'nonlinear']
n_te = 500  # Test set size
reps = 100
n_jobs = 10


def single_monte_carlo_run(rep, n, d, N, type_name, n_te=500):
    # Unique seed for each run
    # note: N should not change the seed for same (rep, n, d, type) for fair comparison
    # under the same base distributions
    seed_val = SEED + TYPE_SEED_OFFSET[type_name] + rep + n + d * 1000
    RS = np.random.RandomState(seed_val)
    
    # Create a separate RNG for test data to ensure consistency across N
    # Train data generation consumes RNG states proportional to N.
    test_seed = (seed_val + 123456789) % (2**32 - 1)
    RS_test = np.random.RandomState(test_seed)

    # Generate train data
    data, joint_distribution = joint_signal_generation(n=n, d=d, N=N, type=type_name, random_state=RS)
    
    # Generate test data (fixed size for evaluation)
    test_data = joint_signal_generation(n=n_te, d=d, M=quantile_grid, type=type_name, random_state=RS_test, test=True)

    # 1. Nonparanormal Frechet Regression (NPT)
    result_npt = npt_frechetreg(
        X=joint_distribution['Predictor'], 
        Y=data,
        Z=test_data['Predictor'], 
        test_Y=(test_data['Marginal_Quantiles'], test_data['Latent_Corrs']),
        M=quantile_grid, 
        verbose=False,
        do_mse=True,
        do_cor_err=True
    )
    
    # Extract NPT metrics
    npt_marg_mse = np.mean(result_npt['marginal_mses'])
    npt_corr_mse = result_npt['latentcor_mse']

    marg_corr_mse = result_npt['latentcor_identity_mse']

    # 2. Gaussian Frechet Regression
    result_gaussian = gaussian_frechetreg(
        joint_distribution['Predictor'], 
        data,
        Z=test_data['Predictor'],
        verbose=False
    )
    
    # Calculate MSE for Gaussian method (using NPT metric)
    mse_gaussian = get_nonparanormal_mse(
        result_gaussian, 
        true_Y=(test_data['Marginal_Quantiles'], test_data['Latent_Corrs']), 
        quantile_grid=quantile_grid
    )
    
    gauss_marg_mse = np.mean(mse_gaussian['marginal_mses'])
    gauss_corr_mse = mse_gaussian['latentcor_mse']
    
    actual_N = N
    
    # Save fitted parameters and test ground truth for later analysis (e.g. Wasserstein distance)
    save_dict = {
        'params': {'rep': rep, 'n': n, 'd': d, 'N': N, 'type': type_name},
        'npt_fit': result_npt,
        'gaussian_fit': result_gaussian,
        'test_data': test_data, # Contains true Marginals and Corrs
        'quantile_grid': quantile_grid
    }
    
    save_dir = REPO_ROOT / 'results' / 'simulations' / 'fitted_objects'
    os.makedirs(save_dir, exist_ok=True)
    save_path = save_dir / f'fit_n{n}_d{d}_N{N}_{type_name}_rep{rep}.pkl'
    
    with open(save_path, 'wb') as f:
        pickle.dump(save_dict, f)

    return [
        {'rep': rep, 'n': n, 'd': d, 'N': actual_N, 'type': type_name, 'method': 'NPT-FR', 'metric': 'Marginal', 'value': npt_marg_mse},
        {'rep': rep, 'n': n, 'd': d, 'N': actual_N, 'type': type_name, 'method': 'NPT-FR', 'metric': 'Correlation', 'value': npt_corr_mse},
        {'rep': rep, 'n': n, 'd': d, 'N': actual_N, 'type': type_name, 'method': 'Marginal-FR', 'metric': 'Marginal', 'value': npt_marg_mse},
        {'rep': rep, 'n': n, 'd': d, 'N': actual_N, 'type': type_name, 'method': 'Marginal-FR', 'metric': 'Correlation', 'value': marg_corr_mse},
        {'rep': rep, 'n': n, 'd': d, 'N': actual_N, 'type': type_name, 'method': 'Gaussian-FR', 'metric': 'Marginal', 'value': gauss_marg_mse},
        {'rep': rep, 'n': n, 'd': d, 'N': actual_N, 'type': type_name, 'method': 'Gaussian-FR', 'metric': 'Correlation', 'value': gauss_corr_mse}
    ]

def run_simulations(reps=reps, n_jobs=n_jobs):
    tasks = []
    for d in d_values:
        for n in n_values:
            for N in N_values:
                for type_name in types:
                    if d == 10 and type_name == 'nonlinear':
                        continue  # d=10 runs only one setting
                    for rep in range(reps):
                        tasks.append((rep, n, d, N, type_name, n_te))
    
    print(f"Starting {len(tasks)} simulation runs with {n_jobs} jobs...")
    results_flat = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(single_monte_carlo_run)(*task) for task in tasks
    )
    
    # Flatten the list of lists
    all_results = [item for sublist in results_flat for item in sublist]
    
    df = pd.DataFrame(all_results)
    
    # Save results
    output_dir = REPO_ROOT / 'results' / 'simulations'
    os.makedirs(output_dir, exist_ok=True)
    csv_path = output_dir / 'simulation_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    return df

def plot_results(
    df,
    *,
    out_path=None,
    figsize=None,
    hue_order=('NPT-FR', 'Marginal-FR', 'Gaussian-FR'),
    palette=None,
    showfliers=False,
    box_width=0.65,
    linewidth=1.0,
    unify_whisker_cap_style=True,
    grid_alpha=0.55,
    x_label_rotation=90,
    x_label_fontsize=10,
    corr_pad_ratio=0.20,
    corr_n_ticks=9,
    corr_label_ticks=(2, 4, 6, 8),
):
    import matplotlib.ticker as ticker
    import math

    # Create a combined column for (n, N)
    df = df.copy()
    df['n_N'] = df.apply(lambda x: f"n={x['n']}\nN={x['N']}", axis=1)
    
    # Sort to ensure correct plotting order (n major, N minor)
    df = df.sort_values(['n', 'N'])
    
    # Define settings (Columns)
    settings = [
        {'name': 'd=2, Linear', 'query': 'd == 2 and type == "linear"'},
        {'name': 'd=2, Nonlinear', 'query': 'd == 2 and type == "nonlinear"'},
        {'name': 'd=10', 'query': 'd == 10'}
    ]
    
    # Define metrics (Rows)
    metrics = ['Marginal', 'Correlation']

    def _apply_common_aesthetics(ax):
        """Shared styling for all subplots."""
        ax.set_xlabel("")
        # Only horizontal grid lines, dashed and light.
        ax.grid(False)
        ax.yaxis.grid(True, which='major', linestyle='--', linewidth=0.8, alpha=grid_alpha)
        ax.xaxis.grid(False)
        # Keep spines subtle
        for side in ['top', 'right']:
            ax.spines[side].set_visible(False)

    def _unify_box_whisker_cap_colors(ax):
        """Post-process seaborn/matplotlib artists so whiskers/caps match box color.

        Works across seaborn versions by pairing each box patch with its related
        Line2D objects in the order matplotlib generates them.
        """
        import matplotlib.patches as mpatches

        # Collect boxes in draw order
        boxes = [p for p in ax.patches if isinstance(p, mpatches.PathPatch)]
        if not boxes:
            return

        # Matplotlib's bxp typically adds per box:
        # - 2 whiskers + 2 caps + 1 median (+ fliers if enabled)
        lines = list(ax.lines)

        # Heuristic: per box, recolor the next 4 Line2D objects whose x-data has
        # The first 4 lines in each group correspond to whiskers/caps.
        per_box = 5 if not showfliers else 6
        for bi, box in enumerate(boxes):
            color = box.get_facecolor()
            start = bi * per_box
            group = lines[start:start + per_box]
            if not group:
                continue

            # whisker(2) + cap(2) are usually first 4
            for ln in group[:4]:
                ln.set_color(color)
            # median is usually 5th (keep it black for readability)
            if len(group) >= 5:
                group[4].set_color('black')
                # Keep median thickness controlled by medianprops.

    def _set_log10_ylim_and_ticks(
        ax,
        values,
        *,
        pad_ratio=0.15,
        n_ticks=9,
        label_ticks=(2, 4, 6, 8),
    ):
        """Set log10 y-scale, multiplicative padding, and geomspace ticks."""
        v = np.asarray(values)
        v = np.asarray(np.ma.masked_invalid(v).compressed())
        v = v[v > 0]
        if v.size == 0:
            return

        vmin = v.min()
        vmax = v.max()
        if vmin == vmax:
            # If everything is identical, create a small multiplicative band.
            vmin = vmin / 1.2
            vmax = vmax * 1.2

        # multiplicative padding around min/max
        low = vmin / (1.0 + pad_ratio)
        high = vmax * (1.0 + pad_ratio)
        # Clamp to a tiny positive number to keep log-scale happy
        low = max(low, 1e-300)

        ax.set_yscale('log')
        ax.set_ylim(low, high)

        # Geometric ticks within y-limits
        if n_ticks is None or n_ticks < 3:
            n_ticks = 9
        ticks = np.geomspace(low, high, num=n_ticks)
        ax.set_yticks(ticks)

        # Label only some ticks (2,4,6,8 by default; 1-indexed)
        labels = []
        for idx1, t in enumerate(ticks, start=1):
            if idx1 in set(label_ticks):
                # exponent in base-10 with 1 decimal place
                exp = math.log10(float(t))
                labels.append(rf"$10^{{{exp:.1f}}}$")
            else:
                labels.append("")
        ax.set_yticklabels(labels)

        # Minor ticks help reading, but keep them subtle
        ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    
    # Setup plot
    sns.set_theme(style="white")
    # Transposed layout: Rows=Metrics, Cols=Settings
    # Removed sharey='row' to allow independent y-limits for each panel
    if figsize is None:
        figsize = (4.5 * len(settings), 3.5 * len(metrics))

    fig, axes = plt.subplots(nrows=len(metrics), ncols=len(settings), 
                             figsize=figsize, 
                             sharex='col', squeeze=False)
    
    # Palette: by default use matplotlib's active color cycle. Allow override.
    if palette is None:
        palette = plt.rcParams['axes.prop_cycle'].by_key().get('color', None)[:len(hue_order)]

    # Pre-compute global y-limits for Marginal (row 1) per setting (column)
    # so that within each column, methods share the same y-scale.
    marginal_ylim_by_setting = {}
    for setting in settings:
        subset_setting = df.query(setting['query'])
        sub = subset_setting[subset_setting['metric'] == 'Marginal']['value']
        sub = np.asarray(np.ma.masked_invalid(sub).compressed())
        if sub.size == 0:
            continue
        ymin = sub.min()
        ymax = sub.max()
        # Add a tiny additive pad for linear scale readability
        pad = 0.05 * (ymax - ymin) if ymax > ymin else 0.05 * max(1.0, abs(ymax))
        marginal_ylim_by_setting[setting['name']] = (ymin - pad, ymax + pad)

    for j, setting in enumerate(settings):
        subset_setting = df.query(setting['query'])
        
        if subset_setting.empty:
            print(f"No data for setting: {setting['name']}")
            continue
            
        for i, metric in enumerate(metrics):
            ax = axes[i, j]
            metric_data = subset_setting[subset_setting['metric'] == metric].copy()
            
            if metric_data.empty:
                continue
            
            # Ensure sorting for this specific subset
            metric_data = metric_data.sort_values(['n', 'N'])

            sns.boxplot(
                data=metric_data,
                x='n_N',
                y='value',
                hue='method',
                hue_order=list(hue_order) if hue_order is not None else None,
                ax=ax,
                showfliers=showfliers,
                width=box_width,
                linewidth=linewidth,
                saturation=1.0,
                boxprops={'alpha': 0.95, 'linewidth': linewidth},
                # Keep all line widths tied to `linewidth` so there's a single knob.
                # (Previously separate whisker/cap widths were often overridden by
                # the post-processing step and were hard to reason about.)
                whiskerprops={'linewidth': linewidth},
                capprops={'linewidth': linewidth},
                medianprops={'linewidth': linewidth-0.6, 'color': 'black'},
                palette=palette
            )

            _apply_common_aesthetics(ax)
            
            # Titles (only on top row) - Bold font
            if i == 0:
                ax.set_title(setting['name'], fontweight='bold', fontsize=13)
            
            # Y Labels (only on first column)
            if j == 0:
                ax.set_ylabel(f"{metric} MSPE\n", fontsize=12) # Averages across marginals could be reflected
            else:
                ax.set_ylabel("")

            # Unify Marginal y-limits within each column
            if metric == 'Marginal':
                ylim = marginal_ylim_by_setting.get(setting['name'])
                if ylim is not None:
                    ax.set_ylim(*ylim)

            # Correlation row: use log10 scale + clean ticks
            if metric == 'Correlation':
                _set_log10_ylim_and_ticks(
                    ax,
                    metric_data['value'].values,
                    pad_ratio=corr_pad_ratio,
                    n_ticks=corr_n_ticks,
                    label_ticks=corr_label_ticks,
                )
                if j == 0:
                    ax.set_ylabel(f"{metric} MSPE (log scale)", fontsize=12)

            # X tick labels adjustments (rotation) for readability (apply to bottom row)
            if i == len(metrics) - 1:
                ax.tick_params(axis='x', labelsize=x_label_fontsize, pad=6, bottom=True, length=3, width=1.7, colors='0.1', direction='in')
                for tick in ax.get_xticklabels():
                    tick.set_rotation(x_label_rotation)
                    # if x_label_rotation == 90:
                    #     tick.set_ha('center')
                    #     tick.set_va('top')
                    #     tick.set_rotation_mode('default')
                    # else:
                    #     tick.set_ha('right')
                    #     tick.set_rotation_mode('anchor')

            # If requested: make whiskers/caps same width+color as their box.
            if unify_whisker_cap_style:
                _unify_box_whisker_cap_colors(ax)
                
            # Remove legend from individual plots
            if ax.get_legend():
                ax.get_legend().remove()

    # Global Legend
    # Get handles and labels from the first plot (assuming it contains all methods)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.985, 0.5), title='Method', fontsize=11, title_fontsize=12,  handlelength=0.8, handleheight=1.3, # box size inside the legend
               labelspacing=0.4, frameon=True, framealpha=0.9)
    
    # Adjust layout to make room for legend on the right
    plt.tight_layout(rect=(0, 0, 0.89, 1))
    
    # Save
    if out_path is None:
        output_dir = REPO_ROOT / 'results' / 'simulations'
        os.makedirs(output_dir, exist_ok=True)
        out_path = output_dir / 'boxplot_combined.pdf'
    else:
        os.makedirs(os.path.dirname(os.fspath(out_path)), exist_ok=True)

    plt.savefig(out_path, format='pdf', bbox_inches='tight', dpi=500)
    print(f"Saved combined plot to {out_path}")

if __name__ == "__main__":
    df = run_simulations(reps=100, n_jobs=-2)
    
    # Load existing results
    csv_path = REPO_ROOT / 'results' / 'simulations' / 'simulation_results.csv'
    if csv_path.exists():
        print(f"Loading results from {csv_path}")
        df = pd.read_csv(csv_path)
        plot_results(df)
    else:
        print("No results file found. Running simulations...")
        df = run_simulations(reps=5)
        plot_results(df)
    
