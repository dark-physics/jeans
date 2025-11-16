"""
rotcurve.py
-----------
Purpose:   Rotation curve calculation utilities for the nonspherical SIDM Jeans modeling package.
Authors:   Sean Tulin, Adam Smith Orlik
Contact:   stulin@yorku.ca, asorlik@yorku.ca
Status:    Stable Version
Last Edit: 2025-09-16

This file contains functions for computing baryonic and halo contributions to the circular velocity, including multipole and numerical derivative utilities.
"""

######################################################################
############################## IMPORTS ###############################
######################################################################
import numpy as np
from inspect import signature
from scipy.integrate import solve_ivp

from .definitions import GN, Z, integrate, central_derivative


######################################################################
######################## FUNCTION DEFINITIONS ########################
######################################################################
# Baryon contributions to rotation curve
def Vsq_baryon(Phi_b, r):
    r"""
    Compute the baryonic contribution $v^2$ to the circular velocity from the baryon potential $\Phi_b$.

    Parameters
    ----------
    Phi_b : callable
        Baryon potential function. Should be a function of $r$ or $(r, \theta)$.
    r : float or array-like
        Radius or array of radii at which to compute the baryonic $v^2$.

    Returns
    -------
    float or ndarray
        The baryonic contribution $v^2$ at each radius $r$.

    Notes
    -----
    Computes $v^2 = r \, d\Phi_b/dr$ using a central finite difference for the derivative.
    If $\Phi_b$ is a function of $(r, \theta)$, evaluates at $\theta = \pi/2$ (the midplane).
    """

    # Baryon potential
    num_Phi_b_variables = len(signature(Phi_b).parameters)
    dPhi_b_dr = np.zeros_like(r)

    r_arr = np.array(r)
    pos = r_arr > 0

    # if num_Phi_b_variables == 1:
    #     dPhi_b_dr[pos] = np.array([derivative(Phi_b, ri, dx=1e-2*ri) for ri in r_arr[pos]])

    # elif num_Phi_b_variables == 2:
    #     theta = np.pi/2
    #     dPhi_b_dr[pos] = np.array([derivative(lambda r: Phi_b(r,theta), ri, dx=1e-2*ri) for ri in r_arr[pos]])

    if num_Phi_b_variables == 1:
        dPhi_b_dr[pos] = np.array([central_derivative(Phi_b, ri, dx=1e-2 * ri) for ri in r_arr[pos]])

    elif num_Phi_b_variables == 2:
        theta = np.pi / 2
        dPhi_b_dr[pos] = np.array(
            [central_derivative(lambda r: Phi_b(r, theta), ri, dx=1e-2 * ri) for ri in r_arr[pos]]
        )
    else:
        raise Exception("Case with %d Phi_b arguments not supported." % num_Phi_b_variables)

    Vsq_out = r_arr * dPhi_b_dr

    if np.ndim(r) == 0:
        return float(Vsq_out)
    else:
        return Vsq_out


# Halo contribution to rotation curve for given L,M mode
def Vsq_LM(rho_LM, r, L, M=0):
    r"""
    Compute the contribution $v^2$ to the circular velocity from the $(L, M)$ multipole of the halo density.

    Parameters
    ----------
    rho_LM : callable
        Function returning the $(L, M)$ multipole of the density at radius $r$.
    r : float or array-like
        Radius or array of radii at which to compute the contribution.
    L : int
        Spherical harmonic degree $L$.
    M : int, optional
        Spherical harmonic order $M$ (default: 0). Only $M=0$ is implemented.

    Returns
    -------
    float or ndarray
        The $(L, M)$ multipole contribution $v^2$ at each radius $r$.

    Notes
    -----
    Computes the contribution using the solution to Poisson's equation for the $L$-th multipole.
    Only $M=0$ (axisymmetric) is currently supported.
    """

    if M != 0:
        raise Exception("M != 0 not implemented.")

    # Handle case where r is just a number
    # Code below assumes r is array or list
    if np.ndim(r) == 0:
        return Vsq_LM(rho_LM, [r], L, M=M)[0]

    elif np.ndim(r) > 1:
        raise Exception("r must be number or 1D array/list.")
    else:
        pass

    # Make sure r is ordered
    order = np.argsort(r)
    r_arr = np.array(r, dtype="float")[order]
    r_eval = r_arr[r_arr > 0]
    if r_eval.size == 0:
        print(f"[Vsq_LM] Warning: r_eval is empty for L={L}. Skipping integration and returning zeros.")
        # Return zeros in the same shape as r_arr, reordered to match input
        Vsq_out = np.zeros_like(r_arr)
        unorder = np.argsort(order)
        return Vsq_out[unorder]

    # Prefactor
    prefactor = 4 * np.pi * GN / (2 * L + 1)

    # Initialize output
    Vsq_out = np.zeros_like(r_arr)

    if max(r) > 0:
        # First term:
        # G(r) = int_0^r dx x^(L+2) rho_LM(x)

        # End points for integration: match r_eval exactly
        rmin, rmax = r_eval[0], r_eval[-1]

        # integrand = lambda r,y: r**(L+2) * rho_LM(r,L)
        def integrand(r, y):
            if r > 0:
                return r ** (L + 2) * rho_LM(r)
            else:
                return 0

        # # Diagnostics before solve_ivp
        # print("[Vsq_LM] r_eval:", r_eval)
        # print("[Vsq_LM] rmin:", rmin, "rmax:", rmax)
        # print("[Vsq_LM] L:", L)
        # try:
        #     test_rho = [rho_LM(val) for val in r_eval[:3]]
        #     # print("[Vsq_LM] rho_LM(r_eval[:3]):", test_rho)
        # except Exception as e:
        #     print("[Vsq_LM] Error evaluating rho_LM on r_eval[:3]:", e)

        # Calculate integrals
        solution = solve_ivp(integrand, [rmin, rmax], [0], rtol=1e-6, atol=1e-6, t_eval=r_eval)
        # print("[Vsq_LM] solution.t:", solution.t)
        # print("[Vsq_LM] solution.y shape:", solution.y.shape)
        if solution.y.shape[0] == 0 or solution.y.shape[1] == 0:
            raise RuntimeError(
                f"[Vsq_LM] solve_ivp returned empty solution.y: shape {solution.y.shape}. Check input arrays and integrand."
            )
        # Align ODE output with r_eval, filling missing points with np.nan
        G_vals = np.full_like(r_eval, np.nan, dtype=float)
        # solution.t may be missing the first point if the solver can't reach it
        for i, rval in enumerate(r_eval):
            idx = np.where(np.isclose(solution.t, rval, atol=1e-10))[0]
            if idx.size > 0:
                G_vals[i] = solution.y[0][idx[0]]
            else:
                print(f"[Vsq_LM] WARNING: ODE solver did not return value for r={rval}. Filling with np.nan.")
        # print("[Vsq_LM] G_vals (aligned):", G_vals)

        mask = r_arr > 0
        Vsq_out[mask] += prefactor * (L + 1) / r_eval ** (L + 1) * G_vals

        # Second term: only needed if L > 0
        # H(r) = int_r^inf x^(1-L) rho_LM(x) = - F(r) + H0
        # where F(r) = int_rmin^r x^(1-L) rho_LM(x)
        # and H0 = int_rmin^inf x^(1-L) rho_LM(x)
        if L > 0:

            integrand = lambda r, y: r ** (1 - L) * rho_LM(r)
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

            Vsq_out[r_arr > 0] += -prefactor * L * r_eval**L * H_vals

            # Finally add extra piece int_rmax^infty dx x^(1-L) rho_LM(x)
            max_iter = 100
            for i in range(max_iter):
                rmin, rmax = rmax, 2 * rmax
                solution = solve_ivp(integrand, [rmin, rmax], [0], rtol=1e-6, atol=1e-6)
                extra = solution.y[0][-1]

                Vsq_out[r_arr > 0] += -prefactor * L * r_eval**L * extra

                if np.allclose(Vsq_out[r_arr > 0], Vsq_out[r_arr > 0] - extra):
                    break

    # Unorder
    unorder = np.argsort(order)

    return Vsq_out[unorder]
