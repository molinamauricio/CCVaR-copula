

# -*- coding: utf-8 -*-
"""
Created on Tue May 28 12:02:27 2024
All the copula are taken from the Nelsen book, except for AMH that was taken
from Barbe (1996) paper
@author: mauro
"""

import numpy as np
import math
if __package__:
    from .ac_utils import P_joe, P_gum, rSibuya_vec, debye1
else:  # Allow running modules directly from this folder.
    from ac_utils import P_joe, P_gum, rSibuya_vec, debye1
import scipy.integrate as integrate
import scipy.stats as st
import scipy.optimize as opt
import mpmath as mp


class acopula():
    """
    # Create an Archimedean copula object


    Attributes
    --------------

    Methods
    -------------

    """



    def __init__(self, family, dim, theta = None, tau = None):
        """


        Parameters
        ----------
        family : str
            The name of the copula
        dim : int
            Integer representing the dimension of the copula d>=2

        Returns
        -------
        acopula object

        Raises
        -------
        ValueError
            If the copula is not supported yet

        """
        super().__init__()
        self.family = family
        self.type = "acopula"
        self.dim = dim


        if not isinstance(dim, int) or dim < 2:
            raise ValueError("dim must be an integer >= 2")

        # If both theta and tau are provided, check consistency.
        if theta is not None and tau is not None:
            computed_tau = self._compute_tau(theta)
            if not np.isclose(computed_tau, tau, atol=1e-6):
                raise ValueError(
                    f"Inconsistent parameters: provided tau={tau} but computed tau from theta={computed_tau}"
                )
            self.theta = theta
            self.tau = tau

        elif theta is not None:
            self.theta = theta
            self.tau = self._compute_tau(theta)

        elif tau is not None:
            self.theta = self._invert_tau(tau)
            self.tau = tau

        elif self.family == 'Indep':
            self.tau = 0.0

        else:
            raise ValueError("Either theta or tau must be provided")

        # Validate the parameter ranges according to the family.
        if self.family == 'Indep':
            self.theta = None
            self.tau = 0.0

        elif self.family == 'Clayton':
            if self.theta <= 0:
                raise ValueError("Clayton copula: theta must be > 0")

        elif self.family == 'Gumbel':
            if self.theta <= 1:
                raise ValueError("Gumbel copula: theta must be > 1")

        elif self.family == 'Frank':
            if self.theta <= 0:
                raise ValueError("Frank copula: theta must be > 0")

        elif self.family == 'Joe':
            if self.theta <= 1:
                raise ValueError("Joe copula: theta must be > 1")

        elif self.family == 'AMH':
            if not (0 < self.theta < 1):
                raise ValueError("AMH copula: theta must be in (0,1)")

        # For families C12, C14, C19, C20 we assume theta > 0
        elif self.family in ['C12', 'C14', 'C19', 'C20']:
            if self.theta <= 0:
                raise ValueError(f"{self.family} copula: theta must be > 0")

        elif self.family == '?':
            pass

        else:
            raise ValueError(f"Family {self.family} not supported.")

    def _compute_tau(self, theta):
        """Compute Kendall's tau from theta for the given family."""
        if self.family == 'Indep':
            return 0.0

        elif self.family == 'Clayton':
            return theta / (theta + 2)

        elif self.family == 'Gumbel':
            return 1 - 1 / theta

        elif self.family == 'Frank':
            # tau = 1 - 4/theta*(1 - D1(theta))
            return 1 - 4 / theta * (1 - debye1(theta))

        elif self.family == 'Joe':
            s = 0.0
            PRECISION = 10000
            for k in range(1, PRECISION + 1):
                term = 1.0 / (k * (theta * k + 2) * (theta * (k - 1) + 2))
                s += term
                if term < 1e-10:
                    break
            return max(1 - 4 * s, 0)

        elif self.family == 'AMH':
            return 1 - 2 * (theta + (1 - theta) ** 2 * math.log(1 - theta)) / (3 * theta ** 2)

        elif self.family == 'C12':
            return 1 - 2 / (3 * theta)

        elif self.family == 'C14':
            return 1 - 2 / (1 + 2 * theta)

        elif self.family == 'C19':
            # For theta in [LOWER_BOUND, UPPER_BOUND], use numerical integration
            LOWER_BOUND = 1.0e-14
            UPPER_BOUND = 91.0
            if theta >= UPPER_BOUND:
                # Here, we do linear extrapolation.
                DELTA = 1.0e-6
                # For illustration purposes only; you might use a pre‐computed value.
                tau_low = 1 - 2 / (3 * (UPPER_BOUND - DELTA))
                tau_high = 1 - 2 / (3 * UPPER_BOUND)
                # Linear interpolation:
                return min(1 - np.finfo(float).eps, tau_low + (theta - (UPPER_BOUND - DELTA)) * (tau_high - tau_low) / DELTA)
            elif theta >= LOWER_BOUND:
                RIGHT_BOUND = 100.0
                integ, _ = integrate.quad(lambda x: math.exp(-x) / x, min(theta, RIGHT_BOUND), RIGHT_BOUND)
                return 1.0 / 3 + 2 * theta * (1 - theta * math.exp(theta) * integ) / 3
            else:
                # Linear interpolation from 0
                tau_at_lower = 1 - 2 / (3 * LOWER_BOUND)
                return (tau_at_lower - 1/3) * theta / LOWER_BOUND + 1/3

        elif self.family == 'C20':
            LOWER_BOUND = 1.0e-8
            if theta >= LOWER_BOUND:
                # Using the formula from Gorecki et al. (2017):
                integ, _ = integrate.quad(lambda t: np.exp((theta+1)*np.log(t) - np.power(t, -theta)), 1e-4, 1-1e-4)
                return 1 - (4 / theta) * (1 / (theta + 2) - np.exp(1) * integ)
            else:
                # Linear interpolation near zero:
                tau_at_lower = 1 - (4 / LOWER_BOUND) * (1 / (LOWER_BOUND + 2) - math.exp(1) * 0)  # approximate
                return tau_at_lower * theta / LOWER_BOUND

        elif self.family == '?':
            return theta

        else:
            raise NotImplementedError(f"tau computation not implemented for family {self.family}")




    def _invert_tau(self, tau_target, tol=1e-6):
        """Compute theta from tau_target for the given family."""
        if self.family == 'Indep':
            return 0.0

        elif self.family == 'Clayton':
            # Closed form: tau = theta/(theta+2)  => theta = 2*tau/(1-tau)
            return 2*tau_target/(1-tau_target)

        elif self.family == 'Gumbel':
            # tau = 1 - 1/theta  => theta = 1/(1-tau)
            return 1/(1-tau_target)

        elif self.family == 'Frank':
            # No closed form; using root finding:
            f = lambda t: self._compute_tau(t) - tau_target
            sol = opt.root_scalar(f, bracket=[1e-6, 50], method='bisect', xtol=tol)
            if sol.converged:
                return sol.root
            else:
                raise ValueError("Frank inversion did not converge")

        elif self.family == 'Joe':
            f = lambda t: self._compute_tau(t) - tau_target
            sol = opt.root_scalar(f, bracket=[1e-6, 1e6], method='bisect', xtol=tol)
            if sol.converged:
                return sol.root
            else:
                raise ValueError("Joe inversion did not converge")

        elif self.family == 'AMH':
            f = lambda t: self._compute_tau(t) - tau_target
            sol = opt.root_scalar(f, bracket=[1e-6, 1-1e-6], method='bisect', xtol=tol)
            if sol.converged:
                return sol.root
            else:
                raise ValueError("AMH inversion did not converge")
        elif self.family == 'C12':
            # According to MATLAB, theta = 2/(3*(1-tau)) for family C12.
            return 2/(3*(1-tau_target))
        elif self.family == 'C14':
            # MATLAB: theta = 1/(1-tau) - 0.5
            return 1/(1-tau_target) - 0.5
        elif self.family == 'C19':
            UPPER_BOUND = 1 - 1e-2
            if tau_target <= 1/3:
                raise ValueError("tau too low for C19")
            elif tau_target > UPPER_BOUND:
                # linear extrapolation (simplified)
                DELTA = 1e-6
                # Compute approximate theta at two points
                theta1 = self._invert_tau(UPPER_BOUND - DELTA, tol)
                theta2 = self._invert_tau(UPPER_BOUND, tol)
                # Linear interpolation
                slope = (theta2 - theta1) / DELTA
                return theta2 + slope * (tau_target - UPPER_BOUND)
            else:
                f = lambda t: self._compute_tau(t) - tau_target
                sol = opt.root_scalar(f, bracket=[1e-6, 91], method='bisect', xtol=tol)
                if sol.converged:
                    return sol.root
                else:
                    raise ValueError("C19 inversion did not converge")
        elif self.family == 'C20':
            f = lambda t: self._compute_tau(t) - tau_target
            sol = opt.root_scalar(f, bracket=[1e-6, 50], method='bisect', xtol=tol)
            if sol.converged:
                return sol.root
            else:
                raise ValueError("C20 inversion did not converge")
        elif self.family == '?':
            return tau_target
        else:
            raise ValueError(f"Unknown family: {self.family}")

    @property
    def tau_value(self):
        return self.tau


    def Phi(self, t):
        """
        Generator function Phi for the Archimedean copula

        Parameters
        ----------
        t : real number between 0 and 1

        Returns
        Value of the generator Phi at t

        """
        epsilon = 1e-10
        t = np.clip(t, epsilon, 1 - epsilon)
        if self.family == 'Indep':
            return -np.log(t)

        elif self.family == 'Clayton':
            theta = self.theta
            return np.power(t,-theta)-1

        elif self.family == 'Frank':
            theta = self.theta
            return -np.log(np.expm1(-theta*t) / np.expm1(-theta))

        elif self.family == 'Gumbel':
            theta = self.theta
            return np.power(-np.log(t),theta)

        elif self.family == 'Joe':
            theta = self.theta
            return -np.log(1-np.power(1-t,theta))

        elif self.family == 'AMH':
            theta = self.theta
            return np.log((1-theta*(1-t))/t)

        elif self.family == 'C12':
            theta = self.theta
            return np.power(1/t -1, theta)

        elif self.family == 'C14':
            theta = self.theta
            return np.power(np.power(t, -1/theta) -1,theta)

        elif self.family == 'C19':
            theta = self.theta
            return np.exp(theta/t)-np.exp(theta)

        elif self.family == 'C20':
            theta = self.theta
            return np.exp(np.power(t, -theta))-np.exp(1)

    def dPhi(self, t):
        """
        First derivative of Generator function Phi of copula wrt t

        Parameters
        ----------
        t : float

        Returns
        Value of the generator Phi at t

        """

        if self.family == 'Indep':
            return -(1/t)

        elif self.family == 'Clayton':
            theta = self.theta
            return -theta*np.power(t,-(theta+1))

        elif self.family == 'Frank':
            theta = self.theta
            return theta*np.exp(-theta*t)/(np.exp(-theta*t)-1)

        elif self.family == 'Gumbel':
            theta = self.theta
            return -(theta/t) * np.power(-np.log(t),theta-1)

        elif self.family == 'Joe':
            theta = self.theta
            return -(theta*np.power(1-t,theta-1))/(1-np.power(1-t,theta))

        elif self.family == 'AMH':
            theta = self.theta
            return (theta-1)/(t*(1-theta*(1-t)))

        elif self.family == 'C12':
            theta = self.theta
            return -theta*np.power(-1+ 1/t, theta-1)/(t**2)

        elif self.family == 'C14':
            theta = self.theta
            return np.power(-1+np.power(t, -1/theta), theta)/(t*(-1+np.power(t, 1/theta)))

        elif self.family == 'C19':
            theta = self.theta
            return -theta*np.exp(theta/t)/(t**2)

        elif self.family == 'C20':
            theta = self.theta
            return -theta*np.power(t, -1-theta)*np.exp(np.power(t, -theta))

    def iPhi(self, t):
        """


        Parameters
        ----------
        t : double
        Input for the inverse.
        theta : double
        parameter of the generator

        Returns
        -------
        Value of the inverse of the generator at t

        """
        if self.family == 'Indep':
            return np.exp(-t)

        elif self.family == 'Clayton':
            theta = self.theta
            return np.power(1+t,-1/theta)

        elif self.family == 'Frank':
            theta = self.theta
            return -np.log(1-(-np.expm1(-theta))*np.exp(-t))/theta

        elif self.family == 'Gumbel':
            theta = self.theta
            return np.exp(-np.power(t, 1/theta))

        elif self.family == 'Joe':
            theta = self.theta
            return 1-np.power(1-np.exp(-t), 1/theta)

        elif self.family == 'AMH':
            theta = self.theta
            return (1-theta)/(np.exp(t)-theta)

        elif self.family == 'C12':
            theta = self.theta
            return 1/(1+np.power(t, 1/theta))

        elif self.family == 'C14':
            theta = self.theta
            return np.power(1+ np.power(t, 1/theta), -theta)

        elif self.family == 'C19':
            theta = self.theta
            return theta/np.log(np.exp(theta)+t)

        elif self.family == 'C20':
            theta = self.theta
            return np.power(np.log(np.exp(1)+t),-1/theta)


    def diPhi(self, t):
        """


        Parameters
        ----------
        t : double
        Input for the inverse.
        theta : double
        parameter of the generator

        Returns
        -------
        Value of the first derivative of the inverse of generator wrt t

        """
        if self.family == 'Indep':
            return np.exp(-t)

        elif self.family == 'Clayton':
            theta = self.theta
            return -(1/theta)*np.power(1+t,-(1/theta+1))

        elif self.family == 'Frank':
            theta = self.theta
            return (-1/theta)* (1-np.exp(-theta))*np.exp(-t)/(1-(1-np.exp(-theta))*np.exp(-t))

        elif self.family == 'Gumbel':
            theta = self.theta
            return (-1/theta)*np.power(t,1/theta-1)*np.exp(-np.power(t, 1/theta))

        elif self.family == 'Joe':
            theta = self.theta
            return -(np.exp(-t)/theta)*np.power(1-np.exp(-t),1/theta-1)

        elif self.family == 'AMH':
            theta = self.theta
            return -(1-theta)*np.exp(t)/((np.exp(t)-theta)**2)

        elif self.family == 'C12':
            theta = self.theta
            return np.power(t, -1 + 1/theta) / (theta * np.power(1 + np.power(t, 1/theta), 2))

        elif self.family == 'C14':
            theta = self.theta
            return -np.power(t, -1 + 1/theta) * np.power(1 + np.power(t, 1/theta), -1 - theta)

        elif self.family == 'C19':
            theta = self.theta
            return -theta / ((np.exp(theta) + t) * np.power(np.log(np.exp(theta) + t), 2))

        elif self.family == 'C20':
            theta = self.theta
            return -np.power(np.log(np.e + t), -1 - 1/theta) / (theta * (np.e + t))

    def aux_fi(self, t, i):
        """
        Parameters
        ----------
        t : float
            input of the function.
        i : int
            index of the auxiliar function

        Returns
        -------
        Value of the auxiliar function representing the (i+1)th derivative of the inverse generator
        evaluated at s=Phi(t)

        """
        if self.family == 'Indep':
            return ((-1)**(i+1))*t

        elif self.family == 'Clayton':
            theta = self.theta
            s = self.Phi(t)
            fall_fact = mp.gamma(i+1+1/theta)/mp.gamma(1/theta)
            return ((-1)**(i+1))*fall_fact*np.power(1+s,-(i+1+1/theta))

        elif self.family == 'Frank':
            theta = self.theta
            s = self.Phi(t)
            polylog_array = np.frompyfunc(mp.polylog, 2, 1)
            out = ((-1)**(i+1))*(1/theta)*polylog_array(-i,(1-np.exp(-theta))*np.exp(-s))
            np_out = np.array(out, dtype=float)
            return np_out

        elif self.family == 'Gumbel':
            theta = self.theta
            s = self.Phi(t)
            return ((-1)**(i+1))*(t/(s**(i+1)))*P_gum(s**(1/theta), i+1, theta)

        elif self.family == 'Joe':
            theta = self.theta
            s = self.Phi(t)
            arg = np.exp(-s)/(1-np.exp(-s))
            Pj = P_joe(arg, i+1, theta)
            return ((-1)**(i+1))*Pj*np.exp(-s)/(theta*np.power(1-np.exp(-s),1-1/theta))

        elif self.family == 'AMH':
            theta = self.theta
            s = self.Phi(t)
            polylog_array = np.frompyfunc(mp.polylog, 2, 1)
            out = ((-1)**(i+1))*((1-theta)/theta)*polylog_array(-(i+1),theta*np.exp(-s))
            np_out = np.array(out, dtype=float)
            return np_out

    def Psi_d(self, t, i):
        """
        Derivative of Psi (inverse of the generator function) of Hofert.
        Supports i = 1..4 for families C12, C14, C19, C20.
        """

        # ----- small helper for high-order central differences on scalars -----
        def richardson_diff(fun, x, order=1, h0=1e-3):
            """
            4th-order accurate central differences for 1st/2nd/3rd derivatives.
            fun: scalar -> scalar, smooth
            x:   scalar
            """
            x = float(np.clip(x, 1e-12, np.inf))
            h = h0 * max(1e-6, x)

            if order == 1:
                # 4th-order first derivative: ( -f(x+2h)+8f(x+h)-8f(x-h)+f(x-2h) )/(12h)
                return (-fun(x + 2*h) + 8.0*fun(x + h) - 8.0*fun(x - h) + fun(x - 2*h)) / (12.0*h)

            elif order == 2:
                # 4th-order second derivative: ( -f(x+2h)+16f(x+h)-30f(x)+16f(x-h)-f(x-2h) )/(12h^2)
                return (-fun(x + 2*h) + 16.0*fun(x + h) - 30.0*fun(x) + 16.0*fun(x - h) - fun(x - 2*h)) / (12.0*h*h)

            elif order == 3:
                # 4th-order third derivative: ( f(x+3h)-8f(x+2h)+13f(x+h)-13f(x-h)+8f(x-2h)-f(x-3h) )/(8h^3)
                return (fun(x + 3*h) - 8.0*fun(x + 2*h) + 13.0*fun(x + h)
                        - 13.0*fun(x - h) + 8.0*fun(x - 2*h) - fun(x - 3*h)) / (8.0*h**3)

            else:
                raise ValueError("order must be 1, 2, or 3")

        # ---------------------------------------------------------------------

        if self.family == 'Indep':
            return ((-1)**(i+1)) * t

        elif self.family == 'Clayton':
            theta = self.theta
            fall_fact = mp.gamma(i + 1/theta) / mp.gamma(1/theta)
            return ((-1)**i) * fall_fact * np.power(1 + t, -(i + 1/theta))

        elif self.family == 'Frank':
            theta = self.theta
            polylog_array = np.frompyfunc(mp.polylog, 2, 1)
            out = ((-1)**i) * (1/theta) * polylog_array(-(i-1), (1 - np.exp(-theta)) * np.exp(-t))
            return np.array(out, dtype=float)

        elif self.family == 'Gumbel':
            theta = self.theta
            return ((-1)**i) * (self.iPhi(t) / (np.power(t, i))) * P_gum(np.power(t, 1/theta), i, theta)

        elif self.family == 'Joe':
            theta = self.theta
            arg = np.exp(-t) / (1 - np.exp(-t))
            Pj = P_joe(arg, i, theta)
            return ((-1)**i) * Pj * np.exp(-t) / (theta * np.power(1 - np.exp(-t), 1 - 1/theta))

        elif self.family == 'AMH':
            theta = self.theta
            polylog_array = np.frompyfunc(mp.polylog, 2, 1)
            out = ((-1)**i) * ((1 - theta)/theta) * polylog_array(-i, theta * np.exp(-t))
            return np.array(out, dtype=float)

        elif self.family == 'C12':
            # iPhi(t) = 1 / (1 + t^(1/theta))
            theta = self.theta
            a = 1.0 / theta
            t_arr = np.asarray(t, dtype=float)

            def Bfun(x):
                x = np.clip(x, 1e-12, np.inf)
                xa = x**a
                return a * x**(a - 1.0) / (1.0 + xa)

            def Psi_d_scalar(x, k):
                x = float(np.clip(x, 1e-12, np.inf))
                fx = 1.0 / (1.0 + x**a)
                B  = Bfun(x)
                if k == 1:
                    return -fx * B
                B1 = richardson_diff(Bfun, x, 1)
                if k == 2:
                    return fx * (B*B - B1)
                B2 = richardson_diff(Bfun, x, 2)
                if k == 3:
                    return -fx * (B**3 - 3*B*B1 + B2)
                B3 = richardson_diff(Bfun, x, 3)
                if k == 4:
                    return fx * (B**4 - 6*B*B*B1 + 3*B1*B1 + 4*B*B2 - B3)
                raise NotImplementedError(f"i={k} not implemented for C12")

            if t_arr.ndim == 0:
                return Psi_d_scalar(float(t_arr), i)
            return np.vectorize(lambda x: Psi_d_scalar(x, i))(t_arr)

        elif self.family == 'C14':
            # iPhi(t) = (1 + t^(1/theta))^(-theta)
            theta = self.theta
            a = 1.0 / theta
            t_arr = np.asarray(t, dtype=float)

            def Bfun(x):
                x = np.clip(x, 1e-12, np.inf)
                xa = x**a
                return x**(a - 1.0) / (1.0 + xa)

            def Psi_d_scalar(x, k):
                x = float(np.clip(x, 1e-12, np.inf))
                fx = (1.0 + x**a)**(-theta)
                B  = Bfun(x)
                if k == 1:
                    return -fx * B
                B1 = richardson_diff(Bfun, x, 1)
                if k == 2:
                    return fx * (B*B - B1)
                B2 = richardson_diff(Bfun, x, 2)
                if k == 3:
                    return -fx * (B**3 - 3*B*B1 + B2)
                B3 = richardson_diff(Bfun, x, 3)
                if k == 4:
                    return fx * (B**4 - 6*B*B*B1 + 3*B1*B1 + 4*B*B2 - B3)
                raise NotImplementedError(f"i={k} not implemented for C14")

            if t_arr.ndim == 0:
                return Psi_d_scalar(float(t_arr), i)
            return np.vectorize(lambda x: Psi_d_scalar(x, i))(t_arr)

        elif self.family == 'C19':
            # iPhi(t) = theta / log(exp(theta) + t)
            theta = self.theta
            b = np.exp(theta)
            t_arr = np.asarray(t, dtype=float)
            t_arr = np.clip(t_arr, 1e-12, np.inf)

            A = b + t_arr
            L = np.log(A)
            f = theta / L                          # iPhi(t)
            B = 1.0 / (A * L)                      # -d/dt log f

            if i == 1:
                return -f * B

            # S and its first two derivatives (closed-form):
            S  = (L + 1.0) / (A * L)
            S1 = - (L*L + L + 1.0) / (A*A * L*L)
            S2 = (2*L**4 + 2*L**3 + 3*L**2 + 2*L) / (A**3 * L**4)

            B1 = -B * S
            if i == 2:
                return f * (B*B - B1)

            B2 = B * (S*S - S1)
            if i == 3:
                return -f * (B**3 - 3*B*B1 + B2)

            B3 = B * (-S**3 + 3.0*S*S1 - S2)
            if i == 4:
                return f * (B**4 - 6*B*B*B1 + 3*B1*B1 + 4*B*B2 - B3)

            raise NotImplementedError(f"i={i} not implemented for C19")

        elif self.family == 'C20':
            # iPhi(t) = [log(e + t)]^(-1/theta)
            theta = self.theta
            a = 1.0 / theta
            ee = np.exp(1.0)
            t_arr = np.asarray(t, dtype=float)
            t_arr = np.clip(t_arr, 1e-12, np.inf)

            A = ee + t_arr
            L = np.log(A)
            f = L**(-a)                            # iPhi(t)
            B = a / (A * L)                        # -d/dt log f

            if i == 1:
                return -f * B

            S  = (L + 1.0) / (A * L)
            S1 = - (L*L + L + 1.0) / (A*A * L*L)
            S2 = (2*L**4 + 2*L**3 + 3*L**2 + 2*L) / (A**3 * L**4)

            B1 = -B * S
            if i == 2:
                return f * (B*B - B1)

            B2 = B * (S*S - S1)
            if i == 3:
                return -f * (B**3 - 3*B*B1 + B2)

            B3 = B * (-S**3 + 3.0*S*S1 - S2)
            if i == 4:
                return f * (B**4 - 6*B*B*B1 + 3*B1*B1 + 4*B*B2 - B3)

            raise NotImplementedError(f"i={i} not implemented for C20")

        else:
            raise NotImplementedError(f"Psi_d not implemented for family {self.family}")


    def K(self,t):
        """


        Parameters
        ----------
        t : float
            input of the function.

        Returns
        Value of the Kendall distribution associated to the copula at t

        """
        d = self.dim
        Phi = lambda x: self.Phi(x)
        fi = lambda x,k: self.aux_fi(x, k)
        epsilon = 1e-10
        t = np.clip(t, epsilon, 1 - epsilon)
        K_cum = 0;
        for i in range(1,d):
            K_cum += ((-1)**i)*((Phi(t)**i)/math.factorial(i))*fi(t, i-1)
            #print(f'i={i} K_cum={t+K_cum}')
        return t+K_cum
        #Add the remaining here

    def C(self, U):
        """


        Parameters
        ----------
        U : np.array float of shape (n, dim)
            Input matrix of column size equal to the dimension of the copula
            Each row is an observation for which the cdf is to be calculated
        Returns
        -------
        np.array of dim 2 and shape (n,1)
        Value of the Copula at point U, C(U) or cdf evaluated at U
        None.

        """
        dim = self.dim
        #Check if the input is a np.array
        if U.ndim == 2:
            n,d = U.shape
        elif U.ndim == 1:
            d = U.shape[0]
            U = U.reshape(1,-1)
        else:
            print('Input argument must be np.array')
            raise ValueError

        # Check for dimension of the input
        if dim != d:
            print("Input argument size is not equal to copula dimension")
            raise ValueError

        if self.family == 'Indep':
            return np.prod(U, axis=1).reshape(-1,1)

        else:
            t = np.sum(self.Phi(U),axis=1)
            return self.iPhi(t)


        #Add the remaining here

    def c(self, U):
        """


         Parameters
         ----------
         U : np.array float of shape (n, d)
             Input matrix of column size equal to the dimension of the copula
             Each row is an observation for which the cdf is to be calculated
         Returns
         -------
         np.array of dim 2 and shape (n,1)
         Value of the density at point U, c(U) or pdf evaluated at U
         None.

        """
        dim = self.dim
        #Check if the input is a np.array
        if U.ndim == 2:
            n,d = U.shape
        elif U.ndim == 1:
            n = 1
            d = U.shape[0]
            U = U.reshape(1,-1)
        else:
            print('Input argument must be np.array')
            raise ValueError

        # Check for dimension of the input
        if dim != d:
            print("Input argument size is not equal to copula dimension")
            raise ValueError

        if self.family == 'Indep':
            return np.ones(n).reshape(-1,1)

        else:
            t = np.sum(self.Phi(U),axis=1)
            num = ((-1)**d)*self.Psi_d(t, d)
            den = np.prod(-self.diPhi(self.Phi(U)), axis=1)
            return num/den


    def random_sample(self, sam_size):
        """
        Algorithm 1 (Marshal, Olkin)
        described in Hofert(2008)

        Parameters
        ----------
        sam_size : int
            size of the sample.

        Returns
        -------
        np.array of size (sam_size, dim)
        Random sample drawn from the copula

        """
        dim = self.dim
        family = self.family

        if family == 'Indep':
            Xi = st.uniform.rvs(size=(sam_size,dim))
            return Xi

        elif family == 'Clayton':
            theta = self.theta
            V = st.gamma.rvs(1/theta, size=sam_size)
            U = np.zeros((sam_size,dim))
            Psi = lambda x: np.power(1+x,-1/theta)
            for i in range(dim):
                Ri = st.expon.rvs(size=sam_size)
                U[:,i] = Psi(Ri/V)
            return U

        elif family == 'Frank':
           theta = self.theta
           p = 1-np.exp(-theta)
           V = st.logser.rvs(p, size=sam_size)
           U = np.zeros((sam_size,dim))
           Psi = lambda x: -np.log(1-(1-np.exp(-theta))*np.exp(-x))/theta
           for i in range(dim):
               Ri = st.expon.rvs(size=sam_size)
               U[:,i] = Psi(Ri/V)
           return U

        elif family == 'Gumbel':
            theta = self.theta
            V = st.levy_stable.rvs(alpha= 1/theta, beta=1, scale=np.power(np.cos(math.pi/(2*theta)),theta), size=sam_size)
            U = np.zeros((sam_size,dim))
            for i in range(dim):
                Ri = st.expon.rvs(size=sam_size)
                U[:,i] = self.iPhi(Ri/V)
            return U
        elif family == 'Joe':
            theta = self.theta
            V = rSibuya_vec(1/theta, sam_size)
            U = np.zeros((sam_size,dim))
            for i in range(dim):
                Ri = st.expon.rvs(size=sam_size)
                U[:,i] = self.iPhi(Ri/V)
            return U
        elif family == 'AMH':
           theta = self.theta
           p = 1-theta
           V = st.geom.rvs(p, size=sam_size)
           U = np.zeros((sam_size,dim))
           Psi = lambda x: (1-theta)/(np.exp(x)-theta)
           for i in range(dim):
               Ri = st.expon.rvs(size=sam_size)
               U[:,i] = Psi(Ri/V)
           return U

    def mpK(self,t):
        """
        Kendall distribution with mpmath for high precision

        Parameters
        ----------
        t : float
            input of the function.

        Returns
        Value of the Kendall distribution associated to the copula at t

        """

        d = self.dim
        mpPhi = lambda x: self.mpPhi(x)
        mpfi = lambda x,k: self.mpaux_fi(x, k)

        t = mp.mpmathify(t)
        K_cum = mp.mpf('0')
        c = mp.mpf('0')
        phi_t = mpPhi(t)

        for i in range(1,d):
            y = ((-1)**i)*((mp.power(phi_t,i))/mp.factorial(i))*mpfi(t, i-1) - c
            temp = K_cum + y
            c = (temp - K_cum) - y
            K_cum = temp

        return t+K_cum

    def mpPhi(self, t):
        """
        Generator function Phi for the Archimedean copula with mpmath for high precision

        Parameters
        ----------
        t : real number between 0 and 1

        Returns
        Value of the generator Phi at t

        """

        if self.family == 'Indep':
            return -mp.log(t)

        elif self.family == 'Clayton':
            theta = self.theta
            return mp.power(t,-theta)-mp.mpf('1.0')

        elif self.family == 'Frank':
            theta = self.theta
            return -mp.log((mp.exp(-theta*t)-1)/(mp.exp(-theta)-1))

        elif self.family == 'Gumbel':
            theta = self.theta
            return mp.power(-mp.log(t),theta)

        elif self.family == 'Joe':
            theta = self.theta
            return -mp.log(1-mp.power(1-t,theta))

        elif self.family == 'AMH':
            theta = self.theta
            return mp.log((1-theta*(1-t))/t)

    def mpaux_fi(self, t, i):
        """
        Returns the auxiliar function f_i using high precision library mpmath

        Parameters
        ----------
        t : float
            input of the function.
        i : int
            index of the auxiliar function

        Returns
        -------
        Value of the auxiliar function representing the (i+1)th derivative of the inverse generator
        evaluated at s=Phi(t)

        """

        if self.family == 'Indep':
            return ((-1)**(i+1))*t

        elif self.family == 'Clayton':
            theta = self.theta
            s = self.mpPhi(t)
            fall_fact = mp.gamma(i+1+1/theta)/mp.gamma(1/theta)
            return ((-1)**(i+1))*fall_fact*mp.power(1+s,-(i+1+1/theta))

        elif self.family == 'Frank':
            theta = self.theta
            s = self.mpPhi(t)
            out = ((-1)**(i+1))*(1/theta)*mp.polylog(-i,(1-mp.exp(-theta))*mp.exp(-s))
            return out

        elif self.family == 'Gumbel':
            theta = self.theta
            s = self.mpPhi(t)
            iPhi = lambda x: mp.exp(-mp.power(x, 1/theta))
            return ((-1)**(i+1))*(iPhi(s)/(mp.power(s,(i+1))))*P_gum(mp.power(s,(1/theta)), i+1, theta)

        elif self.family == 'Joe':
            theta = self.theta
            s = self.mpPhi(t)
            arg = mp.exp(-s)/(1-mp.exp(-s))
            Pj = P_joe(arg, i+1, theta)
            return ((-1)**(i+1))*Pj*mp.exp(-s)/(theta*mp.power(1-mp.exp(-s),1-1/theta))

        elif self.family == 'AMH':
            theta = self.theta
            s = self.mpPhi(t)
            out = ((-1)**(i+1))*((1-theta)/theta)*mp.polylog(-(i+1),theta*mp.exp(-s))
            return out

    def mpdPhi(self, t):
        """
        First derivative of Generator function Phi of copula
        with mpmath for high precision

        Parameters
        ----------
        t : real number between 0 and 1

        Returns
        Value of the generator Phi at t

        """

        if self.family == 'Indep':
            return -(1/t)

        elif self.family == 'Clayton':
            theta = self.theta
            return -theta*mp.power(t,-(theta+1))

        elif self.family == 'Frank':
            theta = self.theta
            return theta*mp.exp(-theta*t)/(mp.exp(-theta*t)-1)

        elif self.family == 'Gumbel':
            theta = self.theta
            return -(theta/t) * mp.power(-mp.log(t),theta-1)

        elif self.family == 'Joe':
            theta = self.theta
            return -(theta*mp.power(1-t,theta-1))/(1-mp.power(1-t,theta))

        elif self.family == 'AMH':
            theta = self.theta
            t = mp.mpmathify(t)
            return mp.mpmathify((theta-1)/(t*(1-theta*(1-t))))

    def __repr__(self):
        return f"ArchimedeanCopula(dim={self.dim}, family='{self.family}', theta={self.theta}, tau={self.tau})"

    def tail_dependence(self, tail_type):
        """
        Compute the upper- or lower-tail dependence coefficient according to Nelsen (2006).

        Parameters
        ----------
        family : str
            A family of Archimedean generator. Expected values are:
            'AMH', 'Clayton', 'Frank', 'Gumbel', 'Joe', 'C12', 'C14', 'C19', 'C20'
        theta : float
            The parameter of the generator.
        tail_type : str
            Either 'lower' or 'upper' indicating the tail.

        Returns
        -------
        coefficient : float
            The computed tail dependence coefficient.

        Raises
        ------
        ValueError
            If the family is unknown or tail_type is not supported.
        """

        family = self.family
        theta = self.theta

        if tail_type == 'lower':
            if family in ['AMH', 'Frank', 'Gumbel', 'Joe']:
                return 0
            elif family in ['Clayton', 'C12']:
                return 2 ** (-1 / theta)
            elif family == 'C14':
                return 0.5
            elif family in ['C19', 'C20']:
                return 1
            else:
                raise ValueError("gettaildependence: Unknown family.")
        elif tail_type == 'upper':
            if family in ['AMH', 'Clayton', 'Frank', 'C19', 'C20']:
                return 0
            elif family in ['Gumbel', 'Joe', 'C12', 'C14']:
                return 2 - 2 ** (1 / theta)
            else:
                raise ValueError("gettaildependence: Unknown family.")
        else:
            raise ValueError("gettaildependence: Unsupported tail type. Choose 'lower' or 'upper'.")








