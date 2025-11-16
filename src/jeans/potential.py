"""
potential.py
------------
Purpose:   Gravitational potential calculation utilities for the nonspherical SIDM Jeans modeling package.
Authors:   Sean Tulin, Adam Smith-Orlik
Contact:   stulin@yorku.ca, asorlik@yorku.ca
Status:    Stable Version
Last Edit: 2025-09-16

This file contains functions for computing the gravitational potential from density multipoles and related utilities.
"""

######################################################################
############################## IMPORTS ###############################
######################################################################
import numpy as np
from inspect import signature
from scipy.integrate import solve_ivp
from .definitions import GN, Z, integrate


######################################################################
######################## FUNCTION DEFINITIONS ########################
######################################################################
def Phi_LM(rho_LM, r, L, M=0):
    r"""
    Compute the gravitational potential $\Phi$ from the $(L, M)$ multipole of the density.

    Parameters
    ----------
    rho_LM : callable
        Function returning the $(L, M)$ multipole of the density at radius $r$.
    r : float or array-like
        Radius or array of radii at which to compute the potential.
    L : int
        Spherical harmonic degree $L$.
    M : int, optional
        Spherical harmonic order $M$ (default: 0). Only $M=0$ is implemented.

    Returns
    -------
    float or ndarray
        The $(L, M)$ multipole contribution to the potential at each radius $r$.

    Notes
    -----
    Computes the solution to Poisson's equation for the $L$-th multipole of the density.
    Only $M=0$ (axisymmetric) is currently supported.
    """

    if M != 0:
        raise Exception("M != 0 not implemented.")

    # Handle case where r is just a number
    # Code below assumes r is array or list
    if np.ndim(r) == 0:
        return Phi_LM(rho_LM, [r], L, M=M)[0]
    elif np.ndim(r) > 1:
        raise Exception("r must be number or 1D array/list.")
    else:
        pass

    # Make sure r is ordered
    order = np.argsort(r)
    r_arr = np.array(r, dtype="float")[order]
    r_eval = r_arr[r_arr > 0]

    # Prefactor
    prefactor = -4 * np.pi * GN / (2 * L + 1)

    # Initialize output
    Phi_out = np.zeros_like(r_arr)

    if max(r) > 0:

        # First term:
        # G(r) = int_0^r dx x^(L+2) rho_LM(x)

        # End points for integration
        rmin, rmax = 0, max(r)

        # integrand = lambda r,y: r**(L+2) * self.rho_L(r,L)
        def integrand(r, y):
            if r > 0:
                return r ** (L + 2) * rho_LM(r)
            else:
                return 0

        # Calculate integrals
        solution = solve_ivp(integrand, [rmin, rmax], [0], rtol=1e-6, atol=1e-6, t_eval=r_eval)
        G_vals = solution.y[0]

        Phi_out[r_arr > 0] += prefactor / r_eval ** (L + 1) * G_vals

        # Second term:
        # H(r) = int_r^inf x^(1-L) rho_LM(x) = - F(r) + H0
        # where F(r) = int_rmin^r x^(1-L) rho_LM(x)
        # and H0 = int_rmin^inf x^(1-L) rho_LM(x)

        integrand = lambda r, y: r ** (1 - L) * rho_LM(r)
        r_eval = r_arr[r_arr > 0]
        rmin, rmax = r_eval[0], r_eval[-1]

        H_vals = np.zeros_like(r_eval)
        integral_list = []
        for i in range(len(r_eval) - 1):
            integral = integrate(
                lambda r: r ** (1 - L) * rho_LM(r),
                r_eval[i],
                r_eval[i + 1],
                rtol=1e-6,
                atol=1e-6,
            )
            integral_list.append(integral)

        for i in range(len(H_vals)):
            H_vals[i] = np.sum(integral_list[i:])

        Phi_out[r_arr > 0] += prefactor * r_eval**L * H_vals

        # Finally add extra piece int_rmax^infty dx x^(1-L) rho_LM(x)
        max_iter = 100
        for i in range(max_iter):
            rmin, rmax = rmax, 2 * rmax
            solution = solve_ivp(integrand, [rmin, rmax], [0], rtol=1e-6, atol=1e-6)
            extra = solution.y[0][-1]

            Phi_out[r_arr > 0] += prefactor * r_eval**L * extra

            if np.allclose(Phi_out[r_arr > 0], Phi_out[r_arr > 0] - extra):
                break

    # Unorder
    unorder = np.argsort(order)

    return Phi_out[unorder]
