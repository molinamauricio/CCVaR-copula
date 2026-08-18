# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 12:32:19 2025

@author: mauro
"""
if __package__:
    from .acopula import acopula
    from .ccvar import *
    from .ac_utils import vectorize_grid
else:  # Allow running modules directly from this folder.
    from acopula import acopula
    from ccvar import *
    from ac_utils import vectorize_grid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import scipy.stats as st
from statsmodels.distributions.empirical_distribution import ECDF
import itertools



def scatterplot_matrix_emp(data, tailtest_n, names=[], **kwargs):
    """
    Plots a scatterplot matrix with scatter plots above the diagonal,
    empirical CDFs on the diagonal, and tail test results below the diagonal.

    Upper triangle:
        Pairwise scatter plots.

    Diagonal:
        Empirical CDF and uniform CDF benchmark.

    Lower triangle:
        Tail test statistic, p-value, and Kendall's tau.

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        2D array of transformed residuals.

    tailtest_n : pd.DataFrame
        DataFrame containing tail dependence test statistics and p-values.
        The upper triangular part is assumed to contain test statistics,
        and the lower triangular part is assumed to contain p-values.

    names : list of str, optional
        List of variable names to be used as labels.

    **kwargs :
        Additional keyword arguments passed to plt.scatter().
    """

    # Allow pandas DataFrame input
    if hasattr(data, "values"):
        if not names:
            names = list(data.columns)
        data = data.values

    data = np.asarray(data)
    numdata, numvars = data.shape

    fig, axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=(15, 15))
    fig.subplots_adjust(hspace=0.0, wspace=0.0)

    for ax in axes.flat:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        row_index, col_index = np.where(axes == ax)

        is_first_col = col_index == 0
        is_last_col = col_index == axes.shape[1] - 1
        is_first_row = row_index == 0
        is_last_row = row_index == axes.shape[0] - 1

        if is_first_col:
            ax.yaxis.set_ticks_position("left")
        if is_last_col:
            ax.yaxis.set_ticks_position("right")
        if is_first_row:
            ax.xaxis.set_ticks_position("top")
        if is_last_row:
            ax.xaxis.set_ticks_position("bottom")

    # Upper triangle: scatter plots only
    for i, j in zip(*np.triu_indices_from(axes, k=1)):
        axes[i, j].scatter(data[:, j], data[:, i], **kwargs)
        axes[i, j].set_xlim(0, 1)
        axes[i, j].set_ylim(0, 1)

    # Diagonal: empirical CDF and uniform benchmark
    for i in range(numvars):
        ax = axes[i, i]

        ecdf = ECDF(data[:, i])
        x_vals = np.linspace(0, 1, numdata)

        ax.plot(x_vals, ecdf(x_vals), color="b")
        ax.plot(x_vals, x_vals, color="r", linestyle="--")

        if i == 0:
            ax.plot([], [], color="b", label="Empirical CDF")
            ax.plot([], [], color="r", linestyle="--", label="Uniform CDF")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # Lower triangle: tail statistic, p-value, and Kendall's tau
    for i, j in zip(*np.tril_indices_from(axes, k=-1)):
        ax = axes[i, j]

        tau_ij, _ = st.kendalltau(data[:, i], data[:, j], nan_policy="omit")

        try:
            stat = float(tailtest_n.iloc[j, i])
            p_value = float(tailtest_n.iloc[i, j])

            if p_value < 0.001:
                p_text = r"$p<0.001$"
            else:
                p_text = rf"$p={p_value:.4f}$"

            text = (
                rf"${stat:.2f}$" "\n"
                rf"({p_text})" "\n"
                rf"$\hat{{\tau}}_{{{i+1},{j+1}}}={tau_ij:.3f}$"
            )

        except Exception:
            text = (
                f"{tailtest_n.iloc[j, i]}\n"
                f"({tailtest_n.iloc[i, j]})\n"
                rf"$\hat{{\tau}}_{{{i+1},{j+1}}}={tau_ij:.3f}$"
            )

        ax.text(
            0.5,
            0.5,
            text,
            ha="center",
            va="center",
            fontsize=8
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # Add names to diagonal
    if not names:
        names = [f"Var{i+1}" for i in range(numvars)]

    for i, label in enumerate(names):
        axes[i, i].annotate(
            label,
            (0.2, 0.9),
            xycoords="axes fraction",
            fontsize=8,
            ha="center",
            va="center"
        )

    # Show ticks on outer edges
    for i, j in zip(range(numvars), itertools.cycle((-1, 0))):
        axes[j, i].xaxis.set_visible(True)
        axes[i, j].yaxis.set_visible(True)

    # Legend
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), fontsize=8)

    # If numvars is odd, fix bottom-right axes limits
    if numvars % 2:
        axes[-1, -1].set_xlim([0, 1])
        axes[-1, -1].set_ylim([0, 1])

    return fig


def plot_var_cvar_ccvar(var_res, list_distributions, list_copulas):
    var_res['Date'] = pd.to_datetime(var_res['Date'])

    def plot_for_beta_distribution(ax, beta_level, beta_str, distribution):
        ax.plot(var_res['Date'], var_res['Act_ret'], label='Actual Returns', color='blue')

        # Define line styles and a set of distinct colors
        line_styles = {'VaR': ':', 'CVaR': '-.', 'CCVaR': '-'}
        cmap = plt.get_cmap('jet')
        colors = cmap(np.linspace(0, 1, len(list_copulas)))  # Evenly spaced colors

        # Iterate over copulas with corresponding colors
        for i, c in enumerate(list_copulas):
            color = colors[i]
            var_col = f"{c}_{distribution}_VaR_{beta_str}"
            cvar_col = f"{c}_{distribution}_CVaR_{beta_str}"
            ccvar_col = f"{c}_{distribution}_CCVaR_{beta_str}"

            if var_col in var_res.columns:
                ax.plot(var_res['Date'], var_res[var_col], label=f'VaR {beta_str}% {c}', linestyle=line_styles['VaR'], color=color, linewidth=1.5)
            if cvar_col in var_res.columns:
                ax.plot(var_res['Date'], var_res[cvar_col], label=f'CVaR {beta_str}% {c}', linestyle=line_styles['CVaR'], color=color, linewidth=1.5)
            if ccvar_col in var_res.columns:
                ax.plot(var_res['Date'], var_res[ccvar_col], label=f'CCVaR {beta_str}% {c}', linestyle=line_styles['CCVaR'], color=color, linewidth=1.5)

        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Values', fontsize=14)
        ax.grid(False)
        ax.tick_params(axis='x', rotation=45, labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

        # Set individual legend for each subplot
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fancybox=True, shadow=True, fontsize=12)

    # Iterate over distributions and plot for both 95% and 99% confidence levels in the same figure
    for distribution in list_distributions:
        print(f"Plotting for distribution: {distribution}")

        fig, axes = plt.subplots(2, 1, figsize=(11.7, 8.3))  # A4 landscape size in inches

        plot_for_beta_distribution(axes[0], 0.95, '95', distribution)  # Plot for 95% confidence level
        plot_for_beta_distribution(axes[1], 0.99, '99', distribution)  # Plot for 99% confidence level

        # Set titles
        axes[0].set_title(r'VaR, CVaR, and CCVaR for ' + distribution + r' Distribution at 95% Confidence Level', fontsize=16)
        axes[1].set_title(r'VaR, CVaR, and CCVaR for ' + distribution + r' Distribution at 99% Confidence Level', fontsize=16)

        plt.subplots_adjust(right=0.85, hspace=0.4)  # Add space between subplots and adjust space for the legends
        plt.show()


def plot_CCVaR(cop_name, theta_i, lamb_i, dist_i):

    """


    Parameters
    ----------
    cop_name : string
        name for contruct an Archimedian type copula
    theta_i : np.array
        Parameters for contruction of the copulas
    lamb_i : TYPE
        Weights for the CCVaR.
    dist_i : list of rv_continuous from scipy.stats
            distribution function for each X_i..

    Returns
    -------
    Plot of the CCVaR with respect the risk level beta for diferent theta_i

    """
    n = len(theta_i)
    d = len(lamb_i)
    if cop_name == 'Indep':
        C_i = acopula(cop_name, dim=d)
    else:
        C_i = [acopula(cop_name, dim=d, theta=theta_i[i]) for i in range(n)]

    beta = np.linspace(0.01, 0.99, 100)
    plt.figure(figsize=(10,10))
    plt.rcParams['text.usetex'] = True

    if cop_name == 'Indep':
        CCVaR_val = np.zeros((1,len(beta)))
        for j in range(len(beta)):
            CCVaR_val[0,j] = CCVaR(C_i, lamb_i, dist_i, beta[j])
        plt.plot(beta, CCVaR_val[0,:], label = r'Indep. Copula')
        plt.title(r'CCVaR with %s Copula $(d= %d)$ ' % (cop_name, d), fontsize=14)
        plt.xlabel(r'$\beta$')
        plt.ylabel(r'$\mathrm{CCVaR}_{\beta}$')
    else:
        CCVaR_val = np.zeros((n,len(beta)))
        for i in range(n):
            for j in range(len(beta)):
                CCVaR_val[i,j] = CCVaR(C_i[i], lamb_i, dist_i, beta[j])
            plt.plot(beta, CCVaR_val[i,:], label = r'$\theta = %.2f$'%(theta_i[i]))
        plt.title(r'CCVaR with %s Copula $(d= %d)$ for $\theta$' % (cop_name, d), fontsize=14)
        plt.xlabel(r'$\beta$')
        plt.ylabel(r'$\mathrm{CCVaR}_{\beta}$')

        plt.legend()
        plt.show()

def plot_CCVaR_theta(cop_name, beta_k, lamb_i, dist_i):

    """


    Parameters
    ----------
    cop_name : string
        name for contruct an Archimedian type copula
    beta_k : np.array
        Array for diferent beta's for risk level
    theta_i : np.array
        Parameters for contruction of the copulas
    lamb_i : TYPE
        Weights for the CCVaR.
    dist_i : list of rv_continuous from scipy.stats
            distribution function for each X_i..

    Returns
    -------
    Plot of the CCVaR with respect the theta for fixed beta

    """

    d = len(lamb_i)

    if cop_name == 'Clayton':
        theta = np.linspace(-0.999, 250, 1000)
        C_i = [acopula(cop_name, dim=d, theta=theta[i]) for i in range(len(theta))]
    elif cop_name == 'Gumbel':
        theta = np.linspace(1, 3, 500)
        C_i = [acopula(cop_name, dim=d, theta=theta[i]) for i in range(len(theta))]
    elif cop_name == 'Joe':
        theta = np.linspace(1, 3, 200)
        C_i = [acopula(cop_name, dim=d, theta=theta[i]) for i in range(len(theta))]
    elif cop_name == 'AMH':
        theta = np.linspace(-1, 0.99, 100)
        C_i = [acopula(cop_name, dim=d, theta=theta[i]) for i in range(len(theta))]
    elif cop_name == 'Frank':
        theta = np.linspace(0, 10, 100)
        C_i = [acopula(cop_name, dim=d, theta=theta[i]) for i in range(len(theta))]

    plt.figure(figsize=(10,10))
    plt.rcParams['text.usetex'] = True

    CCVaR_val = np.zeros((len(beta_k),len(theta)))

    for k in range(len(beta_k)):
        for i in range(len(theta)):
            CCVaR_val[k,i] = CCVaR(C_i[i], lamb_i, dist_i, beta_k[k])

        plt.plot(theta, CCVaR_val[k,:], label=r'$\mathrm{CCVaR}_{\beta}(\beta = %.2f)$' %(beta_k[k]))

    plt.title(r'CCVaR with %s Copula $(d= %d)$' % (cop_name, d), fontsize=14)
    plt.xlabel(r'$\theta$')
    plt.ylabel(r' $\mathrm{CCVaR}_{\beta}$')

    plt.legend()
    plt.show()

def plot_lvl_curve(cop_name, theta_i, beta):
    """


    Parameters
    ----------
    cop_name : string
        Name of archimedean copula.
    theta_i : np.array
        Array with the parameters for the acopula
    beta : float
        Risk level.

    Returns
    -------
    Graph with the level curves C(u1,u2)=beta

    """
    if cop_name == 'Indep':
        theta_i = np.zeros(1)
        C_i = [acopula(cop_name, dim=2) for i in range(len(theta_i))]
    elif cop_name == 'Clayton':
        C_i = [acopula(cop_name, dim=2, theta=theta_i[i]) for i in range(len(theta_i))]
    elif cop_name == 'Gumbel':
        C_i = [acopula(cop_name, dim=2, theta=theta_i[i]) for i in range(len(theta_i))]
    elif cop_name == 'Joe':
        C_i = [acopula(cop_name, dim=2, theta=theta_i[i]) for i in range(len(theta_i))]
    elif cop_name == 'AMH':
        C_i = [acopula(cop_name, dim=2, theta=theta_i[i]) for i in range(len(theta_i))]
    elif cop_name == 'Frank':
        C_i = [acopula(cop_name, dim=2, theta=theta_i[i]) for i in range(len(theta_i))]

    t = np.linspace(beta, 1, 100)
    for i in range(len(theta_i)):
        u2 = np.zeros(len(t))
        u2 = C_i[i].iPhi(C_i[i].Phi(beta)-C_i[i].Phi(t))
        plt.plot(t, u2, label=r'($\theta = %.2f)$' %(theta_i[i]))

    plt.title(r'Level curve for Copula %s at level $\beta$ = %.2f' % (cop_name, beta), fontsize=14)
    plt.xlabel(r'$u_1$')
    plt.ylabel(r' $u_2$')

    plt.legend()
    plt.show()

def plot_K_theta(cop_name, dim, theta_i, beta):
    """
    Parameters
    ----------
    cop_name : string
        Name of archimedean copula.
    dim : int
        Dimension of the copula
    theta_i : np.array
        Array with the parameters for the acopula
    beta : float
        Risk level.

    Returns
    -------
    Graph with the level curves C(u1,u2)=beta

    """
    if cop_name == 'Indep':
        theta_i = np.zeros(1)
        C_i = [acopula(cop_name, dim=dim) for i in range(len(theta_i))]
    elif cop_name == 'Clayton':
        C_i = [acopula(cop_name, dim=dim, theta=theta_i[i]) for i in range(len(theta_i))]
    elif cop_name == 'Gumbel':
        C_i = [acopula(cop_name, dim=dim, theta=theta_i[i]) for i in range(len(theta_i))]
    elif cop_name == 'Joe':
        C_i = [acopula(cop_name, dim=dim, theta=theta_i[i]) for i in range(len(theta_i))]
    elif cop_name == 'AMH':
        C_i = [acopula(cop_name, dim=dim, theta=theta_i[i]) for i in range(len(theta_i))]
    elif cop_name == 'Frank':
        C_i = [acopula(cop_name, dim=dim, theta=theta_i[i]) for i in range(len(theta_i))]

    t = np.linspace(beta, 1, 100)
    for i in range(len(theta_i)):
        K = np.zeros(len(t))
        K = C_i[i].K(t)
        plt.plot(t, 1-K, label=r'($\theta = %.2f)$' %(theta_i[i]))

    plt.title(r'Kendall distribution for Copula %s' % (cop_name), fontsize=14)
    plt.xlabel(r'$t$')
    plt.ylabel(r' $K(t)$')

    plt.legend()
    plt.show()

def plot_ubeta_2d(cop_name, theta_i, beta):
    """
    Plots level curves for a given copula along with boundary lines,
    a shaded region, and an additional "infinite" level curve.

    Parameters
    ----------
    cop_name : str
        Name of the copula to plot the level curve sets.
    theta_i : numpy.ndarray
        Array with the parameters for which to plot the level curves.
    beta : float
        Risk level.

    Returns
    -------
    None
    """
    plt.figure(figsize=(4, 4))
    dim = 2
    # Create the acopula objects based on the copula name and provided theta values.
    if cop_name == 'Indep':
        theta_i = np.zeros(1)
        C_i = [acopula(cop_name, dim=dim) for _ in range(len(theta_i))]
    elif cop_name in ['Clayton', 'Gumbel', 'Joe', 'AMH', 'Frank']:
        C_i = [acopula(cop_name, dim=dim, theta=theta_i[i]) for i in range(len(theta_i))]
    else:
        raise ValueError("Unsupported copula name.")

    # Define u-values from beta to 1.
    t = np.linspace(beta, 1, 100)
    V_last = None  # Will store the last curve's v-values for shading.

    # Plot the level curves for each theta value.
    for i in range(len(theta_i)):
        # Compute v-values from the copula functions.
        V = C_i[i].iPhi(C_i[i].Phi(beta) - C_i[i].Phi(t))
        plt.plot(t, V, label=r'($\theta = %.2f)$' % theta_i[i])
        # Save the last computed curve for later shading.
        if i == len(theta_i) - 1:
            V_last = V.copy()

    # Draw boundary lines (dashed) from the edges:
    # From (beta, 1) to (1, 1)
    plt.plot([beta, 1], [1, 1], color='black', linestyle='--')
    # From (1, beta) to (1, 1)
    plt.plot([1, 1], [beta, 1], color='black', linestyle='--')

    # Shade the region above the last computed level curve (up to v=1).
    if V_last is not None:
        plt.fill_between(t, V_last, 1, color='lightgray', alpha=0.5)
        plt.text(0.98, 0.98, r'$\mathcal{U}_\beta$', ha='right', va='top',
                 fontsize=14, color='black')

    # Add the "infinite theta" level curve as two straight line segments:
    # Segment 1: vertical from (beta, beta) to (beta, 1)
    plt.plot([beta, beta], [beta, 1], color='black', linestyle='-', label=r'($\theta=\infty$)')
    # Segment 2: horizontal from (beta, beta) to (1, beta)
    plt.plot([beta, 1], [beta, beta], color='black', linestyle='-')

    # Set the display limits of the axes from (0.8, 0.8) to (1,1)
    plt.xlim(0.925, 1)
    plt.ylim(0.925, 1)

    plt.title(r'Set $\mathcal{U}_\beta$ Copula %s and $\beta$ = %.2f' % (cop_name, beta), fontsize=8)
    plt.xlabel(r'$u$')
    plt.ylabel(r'$v$')
    plt.legend(loc='lower left', fontsize=6)
    plt.tight_layout()
    plt.show()

def plot_acopula_by_thetas(cop_name, theta_array, n_grid, sam_size):
    """
    For a given copula family and an array of theta's, creates a plot with one row per theta.
    In each row the left panel shows the density (pdf) of the copula (via a 3D surface plot)
    and the right panel shows a scatter plot of a random sample from the copula.

    Parameters
    ----------
    cop_name : str
        Name of the copula/family. (This will be passed to the acopula constructor.)
    theta_array : array_like
        Array (or list) of theta values.
    n_grid : int
        Number of grid points for the density plot.
    sam_size : int
        Sample size for the scatter plot.

    Returns
    -------
    None
        (The function creates a figure and calls plt.show().)
    """
    n_rows = len(theta_array)
    # We fix the figure size so that each subplot is about 3.5 inches square.
    # Two columns ⇒ width = 7 inches; height = 3.5 * (number of rows)
    fig = plt.figure(figsize=(7, 3.5 * n_rows))
    gs = GridSpec(n_rows, 2, figure=fig)

    for i, theta in enumerate(theta_array):
        # Create the acopula instance for the given theta.
        # (This assumes your acopula constructor accepts cop_name, dim, and theta.)
        current_ac = acopula(cop_name, dim=2, theta=theta)

        # ---------------------
        # Left subplot: Density plot (3D surface of the pdf)
        # ---------------------
        ax_density = fig.add_subplot(gs[i, 0], projection='3d')
        x = np.linspace(0.01, 0.99, n_grid)
        y = np.linspace(0.01, 0.99, n_grid)
        X, Y = np.meshgrid(x, y)
        U = vectorize_grid(X, Y)
        # Compute the pdf (density) of the copula.
        pdf = current_ac.c(U).reshape((n_grid, n_grid))
        surf = ax_density.plot_surface(X, Y, pdf, cmap='cool', alpha=0.8)
        ax_density.set_title(r'Density: $\theta=%.2f$' % theta, fontsize=10)
        ax_density.set_xlabel('u')
        ax_density.set_ylabel('v')
        ax_density.set_zlabel(r'$c(u,v)$')
        # Optionally, you could add a color bar if desired:
        # fig.colorbar(surf, ax=ax_density, shrink=0.5, aspect=10)

        # ---------------------
        # Right subplot: Scatter plot of a random sample.
        # ---------------------
        ax_scatter = fig.add_subplot(gs[i, 1])
        data = current_ac.random_sample(sam_size)
        ax_scatter.scatter(data[:, 0], data[:, 1], marker='.', color='b')
        ax_scatter.set_title(r'Scatter: $\theta=%.2f$' % theta, fontsize=10)
        ax_scatter.set_xlim(0, 1)
        ax_scatter.set_ylim(0, 1)
        ax_scatter.set_xlabel('u')
        ax_scatter.set_ylabel('v')

    fig.suptitle('Density and Scatter plots for copula %s' % cop_name, fontsize=14)
    plt.tight_layout()
    plt.show()

def upper_tail_dependence(Ui, Uj, q=0.99):
    """
    Estimate the upper tail dependence coefficient between two variables.

    Parameters
    ----------
    Ui, Uj : array_like
        Data arrays for the two variables (assumed to lie in [0, 1]).
    q : float, optional
        Threshold to define the upper tail (default is 0.95).

    Returns
    -------
    lam : float
        The estimated upper tail dependence coefficient.
    """
    count_Ui = np.sum(Ui > q)
    count_Uj = np.sum(Uj > q)
    count_joint = np.sum((Ui > q) & (Uj > q))
    lam_i = count_joint / count_Ui if count_Ui > 0 else 0
    lam_j = count_joint / count_Uj if count_Uj > 0 else 0
    return (lam_i + lam_j) / 2

def plot_kendall_and_lambda(Uij, list_symbols, innovation):
    """
    Creates a heatmap in which:
      - The upper triangle (including the diagonal) is filled with Kendall's τ values,
      - The lower triangle is filled with the estimated upper tail dependence coefficients λ₍ᵤ₎,
      - Each cell is annotated with a formatted string showing τ (for i<=j) or λ₍ᵤ₎ (for i>j),
      - The background colors of the cells come from a common colormap.

    Parameters
    ----------
    Uij : ndarray of shape (n, d)
        Data matrix with n observations and d variables (values in [0,1]).
    list_symbols : list of str
        Labels for the d variables (used as row and column names).
    innovation : str
        A label for the innovation (included in the title).

    Returns
    -------
    None
        Displays the heatmap.
    """
    n, d = Uij.shape

    # Initialize matrices for Kendall's tau and for lambda_u.
    tau_matrix = np.zeros((d, d))
    lambda_matrix = np.zeros((d, d))

    # Compute tau for every pair and lambda_u for i > j.
    for i in range(d):
        for j in range(d):
            if i == j:
                tau_matrix[i, j] = 1.0
                # (Diagonal: you might leave lambda undefined or set it to a dummy value.)
                lambda_matrix[i, j] = np.nan
            else:
                # Compute Kendall's tau (symmetric)
                tau, _ = kendalltau(Uij[:, i], Uij[:, j])
                tau_matrix[i, j] = tau
                # Compute lambda_u for the lower triangle only.
                lam = upper_tail_dependence(Uij[:, i], Uij[:, j], q=0.95)
                lambda_matrix[i, j] = lam

    combined_matrix = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if i <= j:
                combined_matrix[i, j] = tau_matrix[i, j]
            else:
                combined_matrix[i, j] = lambda_matrix[i, j]

    annot_matrix = np.empty((d, d), dtype=object)
    for i in range(d):
        for j in range(d):
            if i <= j:
                annot_matrix[i, j] = f"$\\tau={tau_matrix[i, j]:.2f}$"
            else:
                annot_matrix[i, j] = f"$\\lambda_u={lambda_matrix[i, j]:.2f}$"

    # Create the heatmap using imshow.
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(combined_matrix, cmap="coolwarm", vmin=-1, vmax=1)

    # Set tick marks and labels.
    ax.set_xticks(np.arange(d))
    ax.set_yticks(np.arange(d))
    ax.set_xticklabels(list_symbols)
    ax.set_yticklabels(list_symbols)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(d):
        for j in range(d):
            ax.text(j, i, annot_matrix[i, j],
                    ha="center", va="center", color="black", fontsize=10)

    ax.set_title(f"Kendall's $\\tau$ (upper) and $\\lambda_u$ (lower) for residual with {innovation} innovations ")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()
