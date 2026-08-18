# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 12:33:29 2025

@author: mauro
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 11:36:58 2024
Calculation of the CCVaR
@author: mauro
"""
import numpy as np
import math
from scipy.integrate import quad
import scipy.special as sc
import mpmath as mp
if __package__:
    from .acopula import acopula
    from .mp_func import *
else:  # Allow running modules directly from this folder.
    from acopula import acopula
    from mp_func import *


def h_dminus1(t, beta, acopula):
    """


    Parameters
    ----------
    t: numpy array like
    beta: float
        beta required for the CCVaR_beta
    i: int
        index for the function h
    acopula : object of the archimedian copula
        archimedian copula.

    Returns
    -------
    Value of the function h_{d-1} at t with d = copula dimension

    """
    t = np.asarray(t)
    d = acopula.dim

    sum_aux = 0
    for j in range(1,d-1):
        sum_aux += (acopula.aux_fi(beta, j)/math.factorial(j))*(acopula.Phi(t)-acopula.Phi(beta))**j

    return acopula.aux_fi(t, 0)-acopula.aux_fi(beta, 0)-sum_aux


def mph_dminus1(t, beta, acopula):
    """


    Parameters
    ----------
    t: numpy array like
    beta: float
        beta required for the CCVaR_beta
    i: int
        index for the function h
    acopula : object of the archimedian copula
        archimedian copula.

    Returns
    -------
    Value of the function h_{d-1} at t with d = copula dimension

    """

    d = acopula.dim
    sum_aux = 0
    for j in range(1,d-1):
        sum_aux += (acopula.aux_fi(beta, j)/mp.factorial(j))*mp.power(acopula.mpPhi(t)-acopula.mpPhi(beta),j)

    return acopula.mpaux_fi(t, 0)-acopula.mpaux_fi(beta, 0)-sum_aux


def CCVaR(acopula, lamb_i, distr_i, beta, **kwargs):
    """


    Parameters
    ----------
    acopula : Archimedian copula of class acopula
        Copula for the vector X.
    lamb_i : np.array
        Weights of the portfolio.
    distr_i : list of rv_continuous from scipy.stats
        distribution function for each X_i.
    beta : float
        Level of confidence.
    loc: localization parameter, if not passed will be equal to 0
    shape: shape parameter for the i-th factor. If not passed, will be equal to 1

    Returns
    -------
    CCVaR at level beta, of the vector X=(X_1, X_2, ..., X_d) with copula C
    and weights lamb_i

    """


    loc = kwargs.get('loc',np.zeros(acopula.dim))
    scale = kwargs.get('scale',np.ones(acopula.dim))

    def integrand(t ,beta_aux, weight, list_dist, cop_aux, loc, scale):
        d = cop_aux.dim
        sum_aux = 0
        for j in range(d):
            sum_aux += weight[j]*(scale[j] * list_dist[j].ppf(t) + loc[j])
        return sum_aux*cop_aux.dPhi(t)*(h_dminus1(t, beta_aux, cop_aux))

    def mpintegrand(t ,beta_aux, weight, list_dist, cop_aux, loc, scale):
        d = cop_aux.dim
        sum_aux = 0

        for j in range(d):
            name_dist = get_distribution_info(list_dist[j])
            if name_dist == 'norm':
                sum_aux += weight[j]*(scale[j] * mp_ndtri(t) + loc[j])
            elif name_dist == 't':
                sum_aux += weight[j]*(scale[j] *  mp_stdtrit(t, list_dist[j].args[0]) + loc[j])
            elif name_dist == 'skew-t':
                sum_aux += weight[j]*(scale[j] *  mp_skestdtrit(t, xi=list_dist[j].xi, nu=list_dist[j].nu) + loc[j])

        return sum_aux*cop_aux.mpdPhi(t)*(mph_dminus1(t, beta_aux, cop_aux))

    numer, _ = quad(integrand, beta, 1, args=(beta, lamb_i, distr_i, acopula, loc, scale))
    denom = 1-acopula.K(beta)
    out = numer / denom

    if np.isnan(numer) or np.isinf(numer) or denom <= np.finfo(np.float64).tiny:
        mp.mp.dps = 50
        print('Values of numer is NaN or inf and/or denom is too close to zero. \n Using mpmath to get higher precision')
        numer = mp.quad(lambda t: mpintegrand(t, beta, lamb_i, distr_i, acopula, loc, scale), [beta, 1], method='gauss-legendre', verbose=True)
        denom = 1-acopula.mpK(beta)
        out = numer / denom
        out = np.asanyarray(out, dtype=float)
        mp.mp.dps = 15
    return out

def copVaR(acopula, lamb_i, distr_i, beta, n1=100000, **kwargs):
    """
    Estimate copula based VaR and CVaR with copula and margins given by distr_i
    using MonteCarlo approach

    Parameters
    ----------
    acopula : acopula
        Archimedian copula object.
    lamb_i : numpy array
        Weights of the portfolio.
    distr_i : list
        list of distribution for the margins.
    beta : float
        Confidence level for the VaR. Number between 0 and 1
     n1 : int
         number of points for estimation of VaR
    **kwargs : scale and shape parameters for each distribution in distr_i



    Returns
    -------
    VaR and CVaR at level beta, of the vector X=(X_1, X_2, ..., X_d) with copula C
    and weights lamb_i

    """
    loc = kwargs.get('loc',np.zeros(acopula.dim))
    scale = kwargs.get('scale',np.ones(acopula.dim))

    d = acopula.dim
    #Generating samples form the copula
    C_sam = acopula.random_sample(n1)

    q = np.zeros([n1, d])

    for i in range(d):
        q[:,i] = lamb_i[i]*(loc[i] + scale[i] * distr_i[i].ppf(C_sam[:,i]))

    rp_sam = np.sum(q, axis=1)
    var = np.quantile(rp_sam, beta)
    cvar = np.mean(rp_sam[rp_sam>=var])
    return var, cvar

