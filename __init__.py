"""
mcopula — Archimedean copulas and copula-based Conditional Value at Risk (CCVaR).

Reference implementation for:
    Molina Barreto, A. M. (2026). "On a Multivariate Extension for Copula-Based
    Conditional Value at Risk." Journal of Statistical Theory and Applications,
    25, Article 21. https://doi.org/10.1007/s44199-026-00174-x
"""

from .acopula import acopula
from .ccvar import CCVaR, copVaR, h_dminus1, mph_dminus1
from .sstd import *
from .ac_utils import *
from .ac_estimation import *

__version__ = "1.0.0"
__author__ = "Andres Mauricio Molina Barreto"
