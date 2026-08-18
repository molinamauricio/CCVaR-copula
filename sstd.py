# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 12:04:14 2024

@author: mauro
"""
import numpy as np
import scipy.stats as st
import scipy.special as sp
from scipy.integrate import quad

class sstd_gen(st.rv_continuous):
    """
    Create a rv_continuous object representing the skewed student distribution
    (Fernandez & Steel)
    xi: skewness parameter
    nu: degrees of freedom
    """
    def __init__(self, xi=1.0, nu=3.0 ,*args, **kwargs):
        """
        Create a rv_continuous object representing the skewed student distribution
        (Fernandez & Steel)

        Parameters
        ----------
        xi : float
            Skewness parameter. Must be strictly positive.
        nu : float
            Degrees of freedom. Must be strictly greater than 2.
        """
        super().__init__(*args, **kwargs)
        self.xi = xi
        self.nu = nu
        self.dist_name = 'skew-t'

    def _argcheck(self):
        valid_xi = (self.xi > 0)
        valid_nu = (self.nu > 2)

        # Check for variance >= 0
        m1 = 2 * np.sqrt(self.nu - 2) / ((self.nu - 1) * sp.beta(0.5, self.nu / 2))
        valid_variance = ((1 - m1**2) * (self.xi**2 + 1 / self.xi**2) + 2 * m1**2 - 1) >= 0

        return valid_xi & valid_nu & valid_variance

    def _stats(self):
        """
        Calculate the mean and variance of the Skew Student-t distribution.

        Returns:
            A tuple containing (mean, variance, skewness, kurtosis).
        """
        xi = self.xi
        nu = self.nu

        m1 = 2 * np.sqrt(nu - 2) / ((nu - 1) * sp.beta(0.5, nu / 2))
        mean = m1 * (xi - 1 / xi)
        variance = (1 - m1**2) * (xi**2 + 1 / xi**2) + 2 * m1**2 - 1
        sigma = np.sqrt(variance)
        skewness = quad(lambda x: ((x - mean) ** 3) * self._pdf(x), -np.inf, np.inf)[0] / sigma**3
        kurtosis = (quad(lambda x: ((x - mean) ** 4) * self._pdf(x), -np.inf, np.inf)[0] / sigma**4) - 3

        return mean, variance, skewness, kurtosis

    def _rvs(self, size=None, random_state=None):
        if size is None:
            size = 1
        # Ensure compatibility with older and newer NumPy versions
        rng = self._random_state

        xi = self.xi
        nu = self.nu

        # Generar muestras de t de Student estándar
        t_samples = st.t.rvs(df=nu, size=size, random_state=rng)

        # Calcular pesos para sesgo
        weight = xi / (xi + 1.0 / xi)
        u = rng.uniform(size=size)  # Números aleatorios uniformes
        xx = np.where(u < weight, 1.0 / xi, xi)
        m1 = 2.0 * np.sqrt(nu - 2.0) / (nu - 1.0) / sp.beta(0.5, 0.5 * nu)
        mu = m1 * (xi - 1.0 / xi)
        sigma = np.sqrt((1.0 - m1**2) * (xi**2 + 1.0 / xi**2) + 2.0 * m1**2 - 1.0)

        # Transformar a Skew Student-t
        rr = -np.abs(t_samples) / xx * np.sign(u - weight)
        return (rr - mu) / sigma


    def _pdf(self, x):
        xi = self.xi
        nu = self.nu
        a = 0.5
        b = nu / 2.0
        beta_val = (sp.gamma(a) / sp.gamma(a + b)) * sp.gamma(b)
        m1 = 2.0 * np.sqrt(nu - 2.0) / (nu - 1.0) / beta_val
        mu = m1 * (xi - 1.0 / xi)
        sigma = np.sqrt((1.0 - m1**2) * (xi**2 + 1.0 / xi**2) + 2.0 * m1**2 - 1.0)
        z = x * sigma + mu
        xxi = np.where(z == 0, 1.0, xi)
        xxi = np.where(z < 0, 1.0 / xi, xxi)
        g = 2.0 / (xi + 1.0 / xi)

        def dstdstd(x, nu):
            s = np.sqrt(nu/(nu-2.0))
            xdt = (sp.gamma((nu+1.0)/2.0)/np.sqrt(np.pi*nu)) / (sp.gamma(nu/2.0)*np.power((1.0+(x*x)/nu),((nu+1.0)/2.0)))
            return s*xdt

        pdf = g * dstdstd(z  / xxi, nu) * sigma
        return pdf

    def _cdf(self, x):
        """
        Cumulative distribution function (CDF) of the Skew Student-t distribution.

        Args:
            x: The value(s) at which to evaluate the CDF. Can be a scalar or a NumPy array.

        Returns:
            The CDF value(s) at x.
        """
        xi = self.xi
        nu = self.nu
        mean, variance, _, _ = self._stats()
        sigma = np.sqrt(variance)
        z = (x * sigma) + mean
        xi_eff = np.where(z < 0, 1.0 / xi, xi)
        g = 2.0 / (xi + 1.0 / xi)
        s = np.sqrt(nu/(nu-2.0))
        p = np.heaviside(z, 0) - np.sign(z) * g * xi_eff * st.t.cdf(-np.abs(z) * s / xi_eff, df=nu)
        return p

    def _ppf(self, q, mu=0, sigma=1):
        """Percent point function (inverse CDF) of the Skew Student-t distribution."""
        xi = self.xi
        nu = self.nu
        m1 = 2.0 * np.sqrt(nu - 2.0) / (nu - 1.0) / sp.beta(0.5, 0.5 * nu)
        mu1 = m1 * (xi - 1.0 / xi)
        sigma1 = np.sqrt((1 - m1**2) * (xi**2 + 1.0 / xi**2) + 2.0 * m1**2 - 1.0)
        g = 2.0 / (xi + 1.0 / xi)
        z = q - (1.0 / (1.0 + xi**2))
        xi_eff = np.power(xi, np.sign(z))
        tmp = (np.heaviside(z, 0) - np.sign(z) * q) / (g * xi_eff)
        quantiles = (-np.sign(z) * (st.t.ppf(tmp, df=nu)/(np.sqrt(nu/(nu-2.0)))) * xi_eff - mu1) / sigma1
        return quantiles * sigma + mu


sstd = sstd_gen()
