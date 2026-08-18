# mcopula

Reference implementation of **copula-based Conditional Value at Risk (CCVaR)** for
Archimedean copulas in arbitrary dimension.

This is the code accompanying:

> Molina Barreto, A. M. (2026). *On a Multivariate Extension for Copula-Based
> Conditional Value at Risk.* **Journal of Statistical Theory and Applications**,
> 25, Article 21. https://doi.org/10.1007/s44199-026-00174-x
> Preprint: [arXiv:2508.16132](https://arxiv.org/abs/2508.16132)

## What it does

The package computes CCVaR for a portfolio `Z = Σ λ_i X_i` whose dependence follows an
Archimedean copula, using the almost closed-form representation derived in the paper —
reducing the defining conditional expectation to a one-dimensional integral involving the
multivariate Kendall distribution function. It also provides a Monte Carlo estimator of
copula-based VaR and CVaR for comparison.

## Modules

| Module | Contents |
|---|---|
| `acopula.py` | Archimedean copula class: generators, Kendall function `K`, sampling, arbitrary dimension |
| `ccvar.py` | `CCVaR(...)` — semi-analytic CCVaR; `copVaR(...)` — Monte Carlo VaR/CVaR |
| `ac_estimation.py` | Copula parameter estimation |
| `ac_utils.py` | Generator helpers (Joe, Gumbel, Sibuya sampling, Debye function) |
| `sstd.py` | Skew Student-t marginals |
| `mp_func.py` | Arbitrary-precision fallbacks (mpmath) for numerically hard regions |
| `plots.py` | Figure generation |

## Installation

```bash
git clone https://github.com/<your-username>/mcopula.git
cd mcopula
pip install -r requirements.txt
```

## Quick start

```python
import numpy as np
from scipy.stats import norm
from acopula import acopula
from ccvar import CCVaR, copVaR

d = 3
cop = acopula(family='clayton', theta=2.0, dim=d)   # check acopula.py for exact signature
w = np.repeat(1/d, d)                               # equally weighted portfolio
margins = [norm() for _ in range(d)]

print(CCVaR(cop, w, margins, beta=0.95))            # semi-analytic
print(copVaR(cop, w, margins, beta=0.95))           # Monte Carlo VaR, CVaR
```

Run scripts from the project root so that `from acopula import acopula` resolves.

## Citation

See `CITATION.cff`, or cite the paper above.

## License

MIT — see `LICENSE`.
