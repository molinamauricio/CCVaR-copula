# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 12:27:12 2024
Functions for estimation and hypotesis test for archimedian copulas
@author: mauro
"""
if __package__:
    from .acopula import acopula
else:  # Allow running modules directly from this folder.
    from acopula import acopula
import numpy as np
import scipy.stats as st
from scipy import optimize
from tqdm import tqdm
import warnings


def pobs(Xij):
    """


    Parameters
    ----------
    Xij : numpy array
         Observed realizations

    Returns
    -------
    out : numpy array
        Pseudo observations.

    """
    n, d = Xij.shape
    ranks = np.apply_along_axis(st.rankdata, 0, Xij)
    out = ranks / (n + 1)
    return out

def LnL(theta_hat, copula, Uij):
    """
    Log likelihood function for the copula

    Parameters
    ----------
    copula : Object of class acopula
        Archimedian copula to estimate
    theta_hat : float
        Parameter of the Archimedian copula
    Uij : np.array of size (n,d)
        Onservation of the copula of size n

    Returns
    -------
    Value of the (-1)*log-likelihood

    """
     # 1) Ensure theta_hat is a scalar (not array([x]))
    theta_hat = float(np.atleast_1d(theta_hat)[0])

    d = copula.dim
    family = copula.family

    # 2) Build copula with scalar theta
    cop = acopula(family=family, dim=d, theta=theta_hat)

    # 3) Get density values and force them to float
    cvals = cop.c(Uij)
    cvals = np.asarray(cvals, dtype=float)      # convert mpf -> float
    cvals = np.clip(cvals, 1e-300, np.inf)      # avoid log(0)

    out = np.sum(np.log(cvals))
    return -out

def pse_LnL(theta_hat, copula, Xij):
    """
    Pseudo-log-likelihood function for the copula

    Parameters
    ----------
    copula : Object of class acopula
        Archimedian copula to estimate
    theta_hat : float
        Parameter of the Archimedian copula
    Uij : np.array of size (n,d)
        Observation of the copula of size n

    Returns
    -------
    Value of the (-1)*pseudo-log-likelihood

    """
    n, d = Xij.shape
    theta_hat = float(np.atleast_1d(theta_hat)[0])

    # Transform Uij to empirical ranks
    Uij_empirical = pobs(Xij)
    family = copula.family
    cop = acopula(family, dim=d, theta=theta_hat)
    out = np.sum(np.log(np.asarray(cop.c(Uij_empirical), dtype=float)))
    return -out

def estim_theta(cop_name, Uij, method='mpl'):
    """
    Estimation of the parameter theta for the copula with observations Uij
    using maximum likelihood estimation 'ml' or maximum pseudo-likelihood estimation
    Parameters
    ----------
    cop_name : str
        Name of the archimedian copula
    Uij : numpy array of size (n, d)
        Observations of size n, and # columns equal to d
    method : string
        Describes the method to use.
        'ml' for maximum likelihood
        'mpl' for maximum pseudo-likelihood

    Returns
    -------
    estimated theta, value of the LnL at that theta, and Stand.Error

    """
    n, d = Uij.shape

    #Defining bounds for each copula
    if cop_name == 'Clayton':
        bounds = optimize.Bounds(lb = 1e-10)
        theta0 = 0.1
    elif cop_name == 'Frank':
        bounds = optimize.Bounds(lb = 1e-10)
        theta0 = 2
    elif cop_name == 'Gumbel':
        bounds = optimize.Bounds(lb = 1)
        theta0 = 1+1e-7
    elif cop_name == 'Joe':
        bounds = optimize.Bounds(lb = 1)
        theta0 = 1+1e-7
    elif cop_name == 'AMH':
        bounds = optimize.Bounds(lb = 0, ub = 1-1e-10)
        theta0 = 0.5
    elif cop_name in ['C12', 'C14']:
        bounds = optimize.Bounds(lb = 1+1e-10, ub = np.inf)
        theta0 = 1.5
    elif cop_name in ['C19', 'C20']:
        bounds = optimize.Bounds(lb = 1e-10, ub = np.inf)
        theta0 = 1.0

    copula = acopula(family=cop_name, dim=d, theta = theta0)
    if method == 'mpl':
        res = optimize.minimize(pse_LnL, theta0, method='L-BFGS-B', jac="2-point",
                                args=(copula, Uij), options={'disp': False},
                                bounds=bounds)
        new_cop = acopula(family=cop_name, dim=d, theta=res.x[0])
        f_hat = pse_LnL(res.x, new_cop, Uij)
    else:
        res = optimize.minimize(LnL, theta0, method='L-BFGS-B', jac="2-point",
                                args=(copula, Uij), options={'disp': False},
                                bounds=bounds)
        new_cop = acopula(family=cop_name, dim=d, theta=res.x[0])
        f_hat = LnL(res.x, new_cop, Uij)

    std_err = np.sqrt(res.hess_inv.todense()[0,0])
    return res.x[0], f_hat, std_err

def emp_cop(u, U):
    """
    Calculates the empirical copula in the sense of Genest (2009).

    Args:
        u (np.ndarray): A 1D array of length d representing the point at which to
                       evaluate the empirical copula.
        U (np.ndarray): A 2D array of shape (n, d) representing the sample data, where
                       n is the number of observations and d is the number of dimensions.

    Returns:
        float: The value of the empirical copula at point u.
    """

    n, d = U.shape

    # Ensure u is a numpy array and has the correct dimensions
    u = np.asarray(u).reshape(1, -1)

    # Input validation
    if u.shape[1] != d:
        raise ValueError("The dimensions of u and U must match.")
    if not np.all((u >= 0) & (u <= 1)):
        raise ValueError("All elements of u must be between 0 and 1.")
    if not np.all((U >= 0) & (U <= 1)):
        raise ValueError("All elements of U must be between 0 and 1.")

    # Efficiently calculate the empirical copula
    indicators = np.all(U <= u, axis=1)  # Element-wise comparison and row-wise 'all' check
    sum_indicators = np.sum(indicators)

    return sum_indicators / n

def sn_statistic(Uij, copula):
    """
    Calculate the statistic Sn for the data Uij with the copula

    Parameters
    ----------
    Uij : np.array of dimension (n, d)
        Pseudo-observations of the copula .
    copula : acopula
        Object of class acopula

    Returns
    -------
    Value of Sn

    """
    n, d = Uij.shape
    Sn = np.zeros(n)
    for i in range(n):
        emp_cop_val = emp_cop(Uij[i, :], Uij)
        copula_c_val = copula.C(Uij[i, :])[0]
        Sn[i] = (emp_cop_val - copula_c_val)**2
    return (1/n)*np.sum(Sn)



def dist_emp_cop(Uij, copula, gamma):
    """
    Calculate the distance between the empirical copula and the estimated copula
    in the region [lower_bound, 1]^d.

    Parameters
    ----------
    Uij : np.array of dimension (n, d)
        Pseudo-observations of the copula.
    copula : acopula
        Object of class acopula (estimated copula).
    gamma : float
        Lower bound of the region (e.g., 0.95 or 0.99).

    Returns
    -------
    float
        The average squared distance between the empirical copula and the estimated copula in the region.
    """
    n, d = Uij.shape
    Sn = 0
    count = 0

    # Select points in the region [gamma, 1]^d
    for i in range(n):
        if np.all(Uij[i, :] >= gamma):
            # Calculate empirical copula at the point Uij[i, :]
            emp_cop_val = emp_cop(Uij[i, :], Uij)

            # Calculate copula CDF at the point Uij[i, :]
            copula_c_val = copula.C(Uij[i, :])[0]

            # Sum of squared differences between empirical and estimated copula
            Sn += (emp_cop_val - copula_c_val) ** 2
            count += 1

    # Return the normalized Sn statistic or NaN if no points in the region
    return Sn / count if count > 0 else np.nan

def gofCopula(cop_name, x, N=100, method='mpl'):
    """
    Goodness of fit Test for Copulas using the Sn statistic given in Genest(2009)

    Parameters
    ----------
    cop_name : string
        Name of the copula
    x : numpy array
        Array of observations to test
    N : int, optional
        Number of simulations required to do the bootstrap method
        DESCRIPTION. The default is 1000.
    method : string, optional
        specifying the goodness-of-fit test statistic to be used.. The default is 'mpl'.
        It can be also 'ml'

    Returns
    -------
    statistic
    parameter of the copula
    p-value.

    """
    #Step 1
    n, d = x.shape
    Uij=pobs(x)
    theta_0,CMLE,std_err = estim_theta(cop_name, Uij, method=method)
    simulated_stats = []
    cop_theta = acopula(family=cop_name, dim=d, theta= theta_0)

    for _ in tqdm(range(N), desc="Simulations"):
        sim_data = cop_theta.random_sample(n)
        sim_stat = sn_statistic(sim_data, cop_theta)
        simulated_stats.append(sim_stat)

    obs_stat = n*sn_statistic(Uij, cop_theta)
    p_value = (0.5+np.sum(np.array(simulated_stats) > obs_stat))/(N+1)

    # Step 5: Compute distances in the regions [0.80, 1]^d and [0.90, 1]^d
    d_80 = dist_emp_cop(x, cop_theta, gamma=0.80)
    d_90 = dist_emp_cop(x, cop_theta, gamma=0.90)

    print(f" Sn statistic= {obs_stat} parameter = {theta_0} P_value: {p_value} ")

    return obs_stat, p_value, theta_0, CMLE, std_err, d_80, d_90

def T_statistic(Uij, copula, B=100):
    """
    Calculate the statistic T (Savu & Trede 2008) for the data Uij with the copula

    Parameters
    ----------
    Uij : np.array of dimension (n, d)
        Pseudo-observations of the copula .
    copula : acopula
        Object of class acopula
    B : int
        Number of partitions of the interval [0,1]
    Returns
    -------
    Value of T

    """
    n, d = Uij.shape
    a = np.linspace(0, 1, B+1)
    V = copula.C(Uij)

    T=0
    for k in range(1, B+1):
        obs_count = np.sum((a[k-1] < V) & (V <= a[k]))
        exp_count = n*(copula.K(a[k])-copula.K(a[k-1]))
        #print(f'for k={k} the obs_count = {obs_count} and exp_count = {exp_count}')
        # Adding a safeguard to avoid division by zero and check the stability of exp_count
        if exp_count > 0:
            T += (obs_count - exp_count) ** 2 / exp_count
        else:
            print(f"Warning: exp_count is zero or negative at interval {k}")
    return T

def gofCopula_T(cop_name, x, N=100, method='mpl'):
    """
    Goodness of fit Test for Copulas using the T statistic given in Savu&Trede(2008)

    Parameters
    ----------
    cop_name : string
        Name of the copula
    x : numpy array
        Array of observations to test
    N : int, optional
        Number of simulations required to do the bootstrap method
        DESCRIPTION. The default is 1000.
    method : string, optional
        specifying the goodness-of-fit test statistic to be used.. The default is 'mpl'.
        It can be also 'ml'

    Returns
    -------
    statistic
    parameter of the copula
    p-value.

    """
    #Step 1
    n, d = x.shape
    Uij=pobs(x)
    theta_0 = estim_theta(cop_name, Uij, method=method)[0]
    cop_theta = acopula(family=cop_name, dim=d, theta= theta_0)
    obs_stat = T_statistic(Uij, cop_theta, B=100)

    simulated_stats = []


    for _ in tqdm(range(N), desc="Simulations"):
        sim_data = cop_theta.random_sample(n)
        theta_star = estim_theta(cop_name, sim_data, method='ml')[0]
        cop_theta_star = acopula(family=cop_name, dim=d, theta=theta_star)
        sim_stat = T_statistic(pobs(sim_data), cop_theta_star)
        simulated_stats.append(sim_stat)


    p_value = np.mean(np.array(simulated_stats) > obs_stat)



    print(f" T statistic= {obs_stat} parameter = {theta_0} P_value: {p_value} ")

    return obs_stat, theta_0, p_value

def emp_cop_two(U):
    n, d = U.shape
    list_emp_cop = [[None for _ in range(d)] for _ in range(d)]
    for i in range(d):
        for j in range(i + 1, d):
            list_emp_cop_ij = np.zeros(n)
            for ii in range(n):
                list_emp_cop_ij[ii] = np.sum((U[:, i] <= U[ii, i]) & (U[:, j] <= U[ii, j]))
            list_emp_cop[i][j] = list_emp_cop_ij / n
    return list_emp_cop


def gof2Sn(U, family, theta, gofTestName, emp_copula):
    """
    Computes a goodness-of-fit statistic for a bivariate Archimedean copula (2-AC).

    Parameters:
        U: numpy array of bivariate observations with shape (n, 2).
        family: string indicating the copula family.
        theta: the parameter of the copula.
        gofTestName: a string in {'E', 'K', 'R'} indicating the test to use:
                     'E' uses the empirical copula,
                     'K' uses the Kendall transformation, and
                     'R' uses the Rosenblatt transformation.
        emp_copula: For tests 'E' and 'K', the precomputed empirical copula (for example, from computeallemp2copulas(U)).

    Returns:
        Sn: the computed goodness-of-fit statistic (a scalar).

    If a NaN or Inf is generated, it is replaced using nanapprox. If more than
    5% of the data are replaced, a warning is issued.
    """
    NAN_ACCEPT_RATIO = 0.05
    if gofTestName == 'K':
        Sn = gof2SnK(U, family, theta, emp_copula, NAN_ACCEPT_RATIO)
    elif gofTestName == 'R':
        Sn = gof2SnR(U, family, theta, NAN_ACCEPT_RATIO)
    elif gofTestName == 'E':
        Sn = gof2SnE(U, family, theta, emp_copula, NAN_ACCEPT_RATIO)
    else:
        raise ValueError("gof2Sn: unsupported gof test.")
    return Sn

def gof2SnE(data, family, parameter, empCopula, NAN_ACCEPT_RATIO):
    """
    Computes the Cramer-von Mises statistic based on the empirical copula.

    Parameters:
        data: numpy array of bivariate observations (n x 2).
        family: copula family (string).
        parameter: copula parameter (theta).
        empCopula: precomputed empirical copula values (numpy array).
        NAN_ACCEPT_RATIO: maximum acceptable ratio of approximated NaNs.

    Returns:
        Sn: the sum of squared differences between theoretical and empirical copula values.
    """
    acop = acopula(dim=2, family=family, theta=parameter)
    psiinv = lambda t: acop.Phi(t)
    psi = lambda t: acop.iPhi(t)
    # Compute theoretical copula values: psi( psiinv(u1) + psiinv(u2) )
    yTheo = psi(psiinv(data[:, 0]) + psiinv(data[:, 1]))
    yTheo, nNaNs = nanapprox(yTheo, data)
    if nNaNs / data.shape[0] > NAN_ACCEPT_RATIO:
        warnings.warn(f"gof2SnE: {nNaNs} NaNs detected and replaced by approximations.")
    yEmp = empCopula
    Sn = np.sum((yTheo - yEmp)**2)
    return Sn

def gof2SnK(data, family, parameter, W, NAN_ACCEPT_RATIO):
    """
    Computes the Cramer-von Mises statistic for the Kendall transformation.
    (For bivariate Archimedean copulas only.)

    Parameters:
        data: numpy array of bivariate observations (n x 2).
        family: copula family (string).
        parameter: copula parameter (theta).
        W: empirical copula values computed via computeallemp2copulas.
        NAN_ACCEPT_RATIO: maximum acceptable ratio of NaN replacements.

    Returns:
        Sn: the computed goodness-of-fit statistic.

    Note:
        This function uses Kend_trans to compute the theoretical Kendall transform.
    """
    n = data.shape[0]
    Kn = np.zeros(n - 1)
    for i in range(n - 1):
        Kn[i] = np.sum(W <= ((i + 1) / n))
    Kn = Kn / n

    # vals: a vector from 1/n to 1 in n steps.
    vals = np.linspace(1 / n, 1, n)
    K_theta_n = Kend_trans(family, parameter, vals)

    # Replace any NaNs in K_theta_n: use 0 for indices < n/2, 1 for indices >= n/2.
    nans = np.isnan(K_theta_n)
    nNans = np.sum(nans)
    if nNans > 0:
        for i in np.where(nans)[0]:
            K_theta_n[i] = 0 if i < n / 2 else 1
        if nNans / n > NAN_ACCEPT_RATIO:
            warnings.warn(f"gof2SnK: {nNans} NaNs detected and replaced by approximations.")

    K_theta_n_lin = 0
    K_theta_n_sqr = 0
    for j in range(n - 1):
        K_theta_n_lin += (Kn[j]**2) * (K_theta_n[j + 1] - K_theta_n[j])
        K_theta_n_sqr += Kn[j] * (K_theta_n[j + 1]**2 - K_theta_n[j]**2)
    Sn = n / 3 + n * K_theta_n_lin - n * K_theta_n_sqr
    return Sn

def gof2SnR(data, family, parameter, NAN_ACCEPT_RATIO):
    """
    Computes the Cramer-von Mises statistic based on the Rosenblatt transformation.

    Parameters:
        data: numpy array of bivariate observations (n x 2).
        family: copula family (string).
        parameter: copula parameter (theta).
        NAN_ACCEPT_RATIO: maximum acceptable ratio of NaN replacements.

    Returns:
        Sn: the computed goodness-of-fit statistic.

    If the Rosenblatt transformation (via cond_cdf) returns only NaNs,
    the parameter is repeatedly divided by 10 until at least one real value is obtained.
    """
    u1 = data[:, 0]
    u2 = data[:, 1]
    theta = parameter
    isAtLeastOneNotNaN = False
    while not isAtLeastOneNotNaN:
        # cond_cdf must be implemented elsewhere.
        e2 = cond_cdf(family, theta, u1, u2)
        if np.sum(~np.isnan(e2)) == 0:
            theta = theta / 10.0
            # getfamilytaurange and tau2theta must be defined elsewhere.
            famRange = tau2theta(family, getfamilytaurange(family))
            if theta < famRange[0]:
                raise ValueError("gof2SnR: Unable to find a parameter value such that the Rosenblatt transformation returns real values.")
            else:
                warnings.warn(f"gof2SnR: Rosenblatt transformation returned only NaNs for the original parameter {parameter}. Trying parameter {theta} instead.")
        else:
            isAtLeastOneNotNaN = True
    e2, nNaNs = nanapprox(e2, data)
    if nNaNs / data.shape[0] > NAN_ACCEPT_RATIO:
        warnings.warn(f"gof2SnR: {nNaNs} NaNs detected and replaced by approximations.")
    e1 = u1
    C_pi = e1 * e2
    D = emp_cop_two(np.column_stack((e1, e2)))
    D = D[0][1]
    if D is None:
        raise ValueError("emp_cop_two returned None.")
    Sn = np.sum((D - C_pi) ** 2)
    return Sn

def Kend_trans(family, theta, t):
    """
    Evaluate the Kendall transformation for a bivariate Archimedean copula.

    Parameters:
        family (str): One of 'A', 'C', 'F', 'G', 'J', '12', '14', '19', or '20'.
        theta (float): Copula parameter.
        t (array-like): A numpy array (or scalar) with values in [0,1].

    Returns:
        kendTrans (numpy array): The Kendall transformation evaluated at t.

    References:
        Genest and Favre (2007).
    """
    t = np.asarray(t)
    if family == 'AMH':
        tTh = t * theta - theta + 1
        # Use elementwise operations; note that division is elementwise.
        kendTrans = t - (t * np.log(tTh / t) * tTh) / (theta - 1)
    elif family == 'Clayton':
        kendTrans = t - (t * (t ** theta - 1)) / theta
    elif family == 'Frank':
        em1T = np.expm1(-theta * t)
        kendTrans = t + (np.exp(theta * t) * np.log(em1T / np.expm1(-theta)) * em1T) / theta
    elif family == 'Gumbel':
        kendTrans = t - (t * np.log(t)) / theta
    elif family == 'Joe':
        tT = (1 - t) ** theta
        kendTrans = t + (np.log(1 - tT) * (tT - 1) * (1 - t) ** (1 - theta)) / theta
    elif family == 'C12':
        kendTrans = t - (t * (t - 1)) / theta
    elif family == 'C14':
        kendTrans = 2 * t - t ** (1 / theta + 1)
    elif family == 'C19':
        kendTrans = t - (t ** 2 * np.expm1((theta * (t - 1)) / t)) / theta
    elif family == 'C20':
        tT = t ** (theta + 1)
        kendTrans = t - ((np.exp(1) * tT * np.exp(-1 / (t ** theta))) - tT) / theta
    else:
        raise ValueError("ACkendtrans: Unsupported family.")
    return kendTrans

def BB1(u1, u2, theta, delta):
    """
    Evaluate the BB1 function used in the conditional CDF for bivariate Archimedean copulas.

    Parameters:
        u1, u2 (array-like): Arrays (or scalars) with values in [0,1].
        theta (float): Parameter.
        delta (float): Secondary parameter.

    Returns:
        cCdf (numpy array): The value of BB1.
    """
    u1 = np.asarray(u1)
    u2 = np.asarray(u2)
    x = (u1 ** (-theta) - 1) ** delta
    y = (u2 ** (-theta) - 1) ** delta
    return (1 + (x + y) ** (1 / delta)) ** (-1 / theta - 1) * (x + y) ** (1 / delta - 1) * \
           x ** (1 - 1 / delta) * u1 ** (-theta - 1)

def BB2(u1, u2, theta, delta):
    """
    Evaluate the BB2 function used in the conditional CDF for bivariate Archimedean copulas.

    Parameters:
        u1, u2 (array-like): Arrays (or scalars) with values in [0,1].
        theta (float): Parameter.
        delta (float): Secondary parameter.

    Returns:
        cCdf (numpy array): The value of BB2.
    """
    u1 = np.asarray(u1)
    u2 = np.asarray(u2)
    x = np.expm1(delta * (u1 ** (-theta) - 1))
    y = np.expm1(delta * (u2 ** (-theta) - 1))
    return (1 + 1 / delta * np.log(x + y + 1)) ** (-1 / theta - 1) / (x + y + 1) * (x + 1) * u1 ** (-theta - 1)

def BB10(u1, u2, theta, delta):
    """
    Evaluate the BB10 function used in the conditional CDF for bivariate Archimedean copulas.

    Parameters:
        u1, u2 (array-like): Arrays (or scalars) with values in [0,1].
        theta (float): Parameter.
        delta (float): Secondary parameter.

    Returns:
        cCdf (numpy array): The value of BB10.
    """
    u1 = np.asarray(u1)
    u2 = np.asarray(u2)
    return (1 - delta * (1 - u1 ** theta) * (1 - u2 ** theta)) ** (-1 / theta - 1) * u2 * \
           (1 - delta * (1 - u2 ** theta))

def cond_cdf(family, theta, u1, u2):
    """
    Evaluate the conditional CDF C(u2|u1) of a bivariate Archimedean copula.

    Parameters:
        family (str): One of 'A', 'C', 'F', 'G', 'J', '12', '14', '19', or '20'.
        theta (float): Copula parameter.
        u1, u2 (array-like): Arrays (or scalars) with values in [0,1]. They must have the same shape.

    Returns:
        cCdf (numpy array): The conditional CDF evaluated at (u1, u2).

    References:
        Joe (2014).
    """
    u1 = np.asarray(u1)
    u2 = np.asarray(u2)
    if family == 'AMH':
        # For family 'A', use BB10 with parameters delta = theta and theta fixed at 1.
        cCdf = BB10(u1, u2, 1, theta)
    elif family == 'Clayton':
        cCdf = (u1 ** (-theta - 1)) * (u1**(-theta) + u2**(-theta) - 1)**(-(1+theta)/theta)
    elif family == 'Frank':
        cCdf = np.exp(-theta * u1) / (np.expm1(-theta) / np.expm1(-theta * u2) + np.expm1(-theta * u1))
    elif family == 'Gumbel':
        x = -np.log(u1)
        y = -np.log(u2)
        cCdf = np.exp(- (x ** theta + y ** theta) ** (1 / theta)) * (1 + (y / x) ** theta) ** (1 / theta - 1) / u1
    elif family == 'Joe':
        cCdf = (1 + ((1 - u2) / (1 - u1)) ** theta - (1 - u2) ** theta) ** (-1 + 1 / theta) * (1 - (1 - u2) ** theta)
    elif family == 'C12':
        cCdf = BB1(u1, u2, 1, theta)
    elif family == 'C14':
        cCdf = BB1(u1, u2, 1 / theta, theta)
    elif family == 'C19':
        cCdf = BB2(u1, u2, 1, theta)
    elif family == 'C20':
        cCdf = BB2(u1, u2, theta, 1)
    else:
        raise ValueError("ACcondcdf: Unsupported family.")
    return cCdf

def gof_SnE(U, copula_theta_hat) -> float:
    """Calculates the GoF statistic S_n^(E) based on the empirical copula."""
    n, d = U.shape
    if d != 2:
        raise ValueError("S_n^(E) is defined for bivariate data only.")

    C_theta_U = copula_theta_hat.C(U)
    C_n_U = np.array([np.mean(np.all(U <= U[i], axis=1)) for i in range(n)])

    return np.sum((C_n_U - C_theta_U.flatten())**2)


def gof_SnK(U, copula_theta_hat) -> float:
    """Calculates the GoF statistic S_n^(K) based on the Kendall process."""
    n, d = U.shape
    if d != 2:
        raise ValueError("S_n^(K) is defined for bivariate data only.")

    V = copula_theta_hat.C(U).flatten()
    V_sorted = np.sort(V)
    K_n_V = np.arange(1, n + 1) / n

    try:
        # Calculate the components for K(v) = v - φ(v) / φ'(v)

        # phi_v = φ(v)
        phi_v = copula_theta_hat.iPhi(V_sorted)

        # dphi_v = φ'(v)
        dphi_v = copula_theta_hat.diPhi(V_sorted)

        # Calculate K_theta, avoiding division by zero
        K_theta_V = V_sorted - np.divide(phi_v, dphi_v,
                                          out=np.full_like(V_sorted, np.inf),
                                          where=dphi_v != 0)

    except (AttributeError, TypeError):
        raise NotImplementedError("acopula class must have .iPhi and .diPhi methods for S_n^(K).")

    return np.sum((K_n_V - K_theta_V)**2)


def gof_SnR(U, copula_theta_hat) -> float:
    """Calculates the GoF statistic S_n^(R) based on Rosenblatt's transform."""
    n, d = U.shape
    if d != 2:
        raise ValueError("S_n^(R) is defined for bivariate data only.")

    try:
        E2 = cond_cdf(copula_theta_hat.family, copula_theta_hat.theta, U[:,0], U[:,1])

    except AttributeError:
         raise NotImplementedError("acopula class must have a .h(U) method.")

    E2_sorted = np.sort(E2)
    i = np.arange(1, n + 1)

    return np.sum((E2_sorted - (2 * i - 1) / (2 * n))**2) + 1 / (12 * n)

