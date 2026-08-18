# -*- coding: utf-8 -*-
"""
Created on Fri May 31 12:03:34 2024

@author: mauro
"""

import numpy as np
import scipy.special as sp
from scipy.integrate import quad
import itertools
import math as mt



#%matplotlib qt

def plot_2d_acopula(acopula, n_grid):
    """


    Parameters
    ----------
    acopula : copula object of the acopula
        acopula must be archimedean of dim =2.
    n_grid : integer
    Number of partions for the unit interval

    Returns
    -------
    Graph of the pdf and cdf of the copula.

    """
    import matplotlib.pyplot as plt

    if acopula.dim != 2:
        print("Input copula must be of dimension 2")
        raise ValueError

    x = np.linspace(0.01, 0.99, n_grid)
    y = np.linspace(0.01, 0.99, n_grid)
    X, Y = np.meshgrid(x,y)
    U = vectorize_grid(X, Y)
    c = acopula.c(U)
    C = acopula.C(U)
    pdf = c.reshape((n_grid, n_grid))
    cdf = C.reshape((n_grid, n_grid))

    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, cdf, cmap='cool', alpha=0.8)
    ax1.set_title('Distribution func. of Copula')
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, Y, pdf, cmap='cool', alpha=0.8)
    ax2.set_title('Density of Copula')


def vectorize_grid(X, Y):
    """


    Parameters
    ----------
    X, Y : array of float
        This is the output of the np.meshgrid


    Returns
    -------
    Vector U of size n_grid ** 2

    """
    n_size = X.size
    U_0 = X.reshape((n_size,1))
    U_1 = Y.reshape((n_size,1))
    return np.concatenate((U_0, U_1), axis = 1)

def scatterplot_matrix(data, names=[], **kwargs):
    """
    Plots a scatterplot matrix of subplots.
    """
    import matplotlib.pyplot as plt

    numdata, numvars = data.shape
    fig, axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=(15,15))
    fig.subplots_adjust(hspace=0.0, wspace=0.0)

    for ax in axes.flat:
        # Hide all ticks and labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        # Get the grid position of the Axes object
        row_index, col_index = np.where(axes == ax)

        # Check if it's in the first or last column
        is_first_col = col_index == 0
        is_last_col = col_index == axes.shape[1] - 1
        is_first_row = row_index == 0
        is_last_row = col_index == axes.shape[0] - 1

        # Set up ticks only on one side for the "edge" subplots...
        if is_first_col:
            ax.yaxis.set_ticks_position('left')
        if is_last_col:
            ax.yaxis.set_ticks_position('right')
        if is_first_row:
            ax.xaxis.set_ticks_position('top')
        if is_last_row:
            ax.xaxis.set_ticks_position('bottom')

    # Plot the data.
    for i, j in zip(*np.triu_indices_from(axes, k=1)):
        for x, y in [(i,j), (j,i)]:
            # FIX #1: this needed to be changed from ...(data[x], data[y],...)
            axes[x,y].scatter(data[:,x], data[:,y], **kwargs)

    # Label the diagonal subplots...
    if not names:
        names = ['x'+str(i) for i in range(numvars)]

    for i, label in enumerate(names):
        axes[i,i].annotate(label, (0.5, 0.5), xycoords='axes fraction',
                ha='center', va='center')

    # Turn on the proper x or y axes ticks.
    for i, j in zip(range(numvars), itertools.cycle((-1, 0))):
        axes[j,i].xaxis.set_visible(True)
        axes[i,j].yaxis.set_visible(True)

    # FIX #2: if numvars is odd, the bottom right corner plot doesn't have the
    # correct axes limits, so we pull them from other axes
    if numvars%2:
        axes[-1,-1].set_xlim([0,1])
        axes[-1,-1].set_ylim([0,1])

    return fig

def plot_scatter(acopula, sam_size):
    """


    Parameters
    ----------
    acopula : acopula
        Object of the class copula
    sam_size : int
        size of the sample

    Returns
    -------
    A scatter plot matrix

    """
    import matplotlib.pyplot as plt

    dim = acopula.dim
    data = acopula.random_sample(sam_size)
    names = ['U_%d'%(i+1) for i in range(dim)]
    fig = scatterplot_matrix(data, names, marker='.', color='b')
    fig.suptitle(r'Scatterplot Matrix for Copula of family %s with $\theta = %.4f$' % (acopula.family, acopula.theta))
    plt.show()


def P_gum(x, d, theta):
    """
    See Hofert(2011)b. Function to compute the Gumbel derivative

    Parameters
    ----------
    x : float
        argument of the function.
    d : int
        order of the derivative.
    theta : float
        parameter of the copula.


    Returns
    -------
    P_gum.

    """
    def a_gum(chi, e, k):
        a_sum = 0
        for j in range(k):
            a_sum += (-1)**(e-(j+1))*sp.binom(k,j+1)*sp.binom((j+1)/chi,e)
        return (mt.factorial(e)/mt.factorial(k))*a_sum
    p_sum = 0
    for k in range(d):
        p_sum += a_gum(theta, d, k+1)*np.power(x,k+1)
    return p_sum

def P_joe(x, d, theta):
    """
    See Hofert(2011)b. Function to compute the Joe derivative

    Parameters
    ----------
    x : float
        argument of the function.
    d : int
        order of the derivative.
    theta : float
        parameter of the copula.


    Returns
    -------
    P_joe.

    """
    def a_joe(chi, e, f):
        return sp.stirling2(e,f)*sp.gamma(f-1/chi)/sp.gamma(1-1/chi)

    p_sum = 0
    for k in range(d):
        p_sum += a_joe(theta, d, k+1)*np.power(x,k)
    return p_sum

"""

  Sample V from a Sibuya(alpha) distribution with cdf F(n) = 1-1/(n*B(n,1-alpha)),
  n in IN, with Laplace-Stieltjes transform 1-(1-exp(-t))^alpha via the
  algorithm of Hofert (2011).

  @param alpha parameter theta0/theta1 in (0,1]
  @param gamma_1_a Gamma(1-alpha)
  @return a random variate from F
  @From Marius Hofert, Martin Maechler
 */
"""
def rSibuya(alpha, gamma_1_a):

    U = np.random.uniform()
    if U <= alpha:
        return 1
    else:
        Ginv = ((1-U)*gamma_1_a)**(-1/alpha)
        fGinv = np.floor(Ginv)
        if 1-U < 1/(fGinv*sp.beta(fGinv, 1-alpha)):
            return np.ceil(Ginv)
        else:
            return fGinv

def rSibuya_sum(alpha, n):
    gamma_1_a = sp.gamma(1. - alpha)
    n_sum_V = 0
    for _ in range(n):
        n_sum_V += rSibuya(alpha, gamma_1_a)
    return n_sum_V

def rSibuya_vec(alpha, n):
    V = np.zeros(n)
    if n >= 1:
        gamma_1_a = sp.gamma(1. - alpha)
        for i in range(n):
            V[i] = rSibuya(alpha, gamma_1_a)
    return V




def debye1(theta):
    """
    Debye function D1(theta) = (1/theta) * ∫_0^theta t/(exp(t) - 1) dt
    Approximation using Taylor for small |theta| and asymptotic for large |theta|.
    """
    theta = np.asarray(theta, dtype=float)

    def D1_scalar(x):
        if x == 0.0:
            return 1.0  # limit θ→0

        ax = abs(x)

        # Small |θ|: Taylor expansion around 0
        # D1(x) = 1 - x/4 + x^2/36 - x^4/3600 + O(x^6)
        if ax < 1e-2:
            x2 = x * x
            return 1.0 - x/4.0 + x2/36.0 - (x2 * x2)/3600.0

        # Moderate |θ|: use the same series (it converges for |x| < 2π)
        if ax < 2*np.pi:
            x2 = x * x
            s = 1.0 - x/4.0 + x2/36.0 - (x2 * x2)/3600.0
            return s

        # Large |θ|: asymptotic D1(x) ~ π²/(6x)
        return (np.pi**2) / (6.0 * x)

    if theta.ndim == 0:
        return float(D1_scalar(float(theta)))
    return np.vectorize(D1_scalar, otypes=[float])(theta)
