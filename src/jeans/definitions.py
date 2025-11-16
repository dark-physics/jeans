"""
definitions.py
--------------
Purpose:   Physical constants, default functions, and core integration utilities for the nonspherical SIDM Jeans modeling package.
Authors:   Sean Tulin, Adam Smith-Orlik
Contact:   stulin@yorku.ca, asorlik@yorku.ca
Status:    Stable Version
Last Edit: 2025-09-16

This file contains physical constants (e.g., $G_N$), default functions (e.g., no_baryons), and robust integration utilities used throughout the package.
"""

######################################################################
############################## IMPORTS ###############################
######################################################################
import numpy as np
import matplotlib.pyplot as plt

# These warnings raise an error if quad raises an integration warning
# integrate() will use solve_ivp instead to evaluate the integral
import warnings

warnings.filterwarnings(action="error", message="The maximum")
warnings.filterwarnings(action="error", message="The occurrence")

from scipy.special import lpmv
from scipy.integrate import quad, solve_ivp

# Newton's constant
GN = 4.302e-6  # km^2/s^2*kpc/Msol


######################################################################
######################## FUNCTION DEFINITIONS ########################
######################################################################
# default function for zero baryon density
def no_baryons(r):
    if np.ndim(r) == 0:
        return 0
    else:
        return np.zeros_like(r, "f")


# Numerical derivative
def central_derivative(f, x, dx):
    r"""
    Compute the numerical derivative of a function $f$ at point $x$ using the central difference formula.

    Parameters
    ----------
    f : callable
        Function to differentiate.
    x : float
        Point at which to evaluate the derivative.
    dx : float
        Step size for the finite difference.

    Returns
    -------
    float
        Approximate value of $f'(x)$ using central difference.

    Notes
    -----
    Uses the formula:

        f'(x) \approx [f(x + dx) - f(x - dx)] / (2 dx)
    """
    return (f(x + dx) - f(x - dx)) / (2 * dx)


# Use custom integration function throughout
def integrate(func, xmin, xmax, atol=1e-8, rtol=1e-8, args=()):

    try:
        output = quad(func, xmin, xmax, args=tuple(args), limit=50, epsabs=atol, epsrel=rtol)[0]

    except:

        # print('quad raised an error, using solve_ivp instead (a bit slower).')
        RHS = lambda t, y: func(t, *args)
        sol = solve_ivp(RHS, [xmin, xmax], [0], atol=1e-12, rtol=1e-12)
        output = sol.y[0][-1]

    return output


# Tesseral harmonics
def Z(L, M, theta, phi):

    Condon_Shortley_phase = (-1) ** M
    x = np.cos(theta)

    if M == 0:
        return Condon_Shortley_phase * np.sqrt((2 * L + 1) / (4 * np.pi)) * lpmv(0, L, x)
    elif M > 0:
        fact = lambda num: factorial(num, exact=True)
        return (
            Condon_Shortley_phase
            * np.sqrt(2 * (2 * L + 1) / (4 * np.pi))
            * np.sqrt(fact(L - M) / fact(L + M))
            * lpmv(M, L, x)
            * np.cos(M * phi)
        )
    elif M < 0:
        fact = lambda num: factorial(num, exact=True)
        return (
            (-1)
            * Condon_Shortley_phase
            * np.sqrt(2 * (2 * L + 1) / (4 * np.pi))
            * np.sqrt(fact(L + M) / fact(L - M))
            * lpmv(-M, L, x)
            * np.sin(M * phi)
        )
    else:
        return 0
