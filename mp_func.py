# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 14:15:13 2024

@author: mauro
"""
import mpmath as mp
import scipy.special as sc
import numpy as np
import time
from timeit import default_timer as timer


def get_distribution_info(dist):
    try:
        if dist.dist.name == 'norm':
            return 'norm'
        elif dist.dist.name == 't':
            return 't'
    except:
        return 'skew-t'


def mp_ndtri(p):
    z = sc.ndtri(np.array(p, dtype=float))
    return z



def mp_stdtrit(p, df):
    def t_cdf(t, df):
        x = df / (df + t**2)
        beta_cdf = 0.5 * mp.betainc(df / 2, 0.5, x2=x, regularized=True)
        return 1 - beta_cdf if t > 0 else beta_cdf

    t_value = sc.stdtrit(df, np.array(p, dtype=float))
    return t_value






def mp_skestdtrit(q, xi, nu, mu=0, sigma=1):
    """Percent point function (inverse CDF) of the Skew Student-t distribution."""
    mp.dps = 100
    m1 = 2.0 * mp.sqrt(nu - 2.0) / (nu - 1.0) / mp.beta(0.5, 0.5 * nu)
    mu1 = m1 * (xi - 1.0 / xi)
    sigma1 = mp.sqrt((1 - m1**2) * (xi**2 + 1.0 / xi**2) + 2.0 * m1**2 - 1.0)
    g = 2.0 / (xi + 1.0 / xi)
    z = q - (1.0 / (1.0 + xi**2))
    xi_eff = mp.power(xi, mp.sign(z))
    tmp = (mp_heaviside(z, 0) - mp.sign(z) * q) / (g * xi_eff)
    quantiles = (-mp.sign(z) * (sc.stdtrit(nu, np.array(tmp, dtype=float))/(mp.sqrt(nu/(nu-2.0)))) * xi_eff - mu1) / sigma1
    return quantiles * sigma + mu

def mp_heaviside(x1, x2):
    """
    Heaviside step function for mpmath objects.

    Args:
        x: The input value (mpmath mpf or float).
        zero_value (optional): The value to return when x is zero (default 0.5).

    Returns:
        0 if x < 0, zero_value if x == 0, and 1 if x > 0.
    """
    if mp.almosteq(x1, 0):
        return x2
    elif x1 < 0:
        return mp.mpf(0)
    else:
        return mp.mpf(1)