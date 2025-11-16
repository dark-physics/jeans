"""
nonspherical.py
--------------
Purpose:   Nonspherical Jeans modeling and relaxation routines for SIDM and CDM halos.
Authors:   Sean Tulin, Adam Smith-Orlik
Contact:   stulin@yorku.ca, asorlik@yorku.ca
Status:    Stable Version
Last Edit: 2025-09-16

This file contains the main relaxation solver and supporting routines for nonspherical halo modeling in the nonspherical SIDM Jeans modeling package.
"""

######################################################################
############################## IMPORTS ###############################
######################################################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.interpolate import interp1d
import scipy.sparse as sparse

from inspect import signature

from . import spherical as sphmodel
from .definitions import GN, Z, integrate
from .classes import isothermal_profile, CDM_profile
from .tools import timed


########################################################################
####################### Main relaxation code: ##########################
# Solve for the isothermal profile given CDM boundary conditions at r1 #
########################################################################
@timed  # Uncomment to time function
def relaxation(
    r1,
    outer_halo,
    L_list=[0],
    M_list=[0],
    converge=None,
    max_iter=20,
    init_grid=5,
    r_grid=200,
    verbose=False,
    finaltol=1e-8,
    **extraneous,
):
    """
    relaxation(r1,*outer_halo_inputs) calculates the SIDM profile using nonspherical relaxation method, with
    outside-in matching onto a fixed outer halo.

        Parameters:
            r1: number (matching radius)
            outer_halo_inputs: args quantifying outer halo.
                This is "M200,c" (default) or "rhos,rs" setting "input_NFW=True"

        Additional keyword arguments:
            Phi_b=no_baryons: Phi_b is a function of (r) or (r,theta).
                Default is no baryons (function that returns zero).
            L_list=[0]: List of L modes included
            M_list=[0]: List of M modes included (note M != 0 not supported yet)
            max_iter=100: Maximum iterations of relaxation method before failure
            verbose=False: Print output during relaxation method
            init_grid=5: Size of initial grid (used as initial guess)
            r_grid=200: Number of r points used in grid.
            finaltol=1e-6: Desired absolute and relative tolerance for relaxation solution.
            converge=None: User-defined function converge(profile) to evaluate,
                increasing L modes and r_grid until it converges.
            atol=1e-3: Desired absolute tolerance for convergence function.
            rtol=1e-3: Desired relative tolerance for convergence function.
            input_NFW=False: Setting to True allows user to input rhos,rs instead of M200,c
            q=1: Nonsphericity of outer halo (q<1 is disklike, q>1 is prolate)
            AC_prescription=None: Can be set to 'Cautun' or 'Gnedin' to include adiabatic contraction
            M_b=None: Baryon enclosed mass profile. When including adiabatic contraction,
                this speeds up the outer halo calculation.
            Gnedin_params=(1.6,0.8): (A0,w) parameters from Gnedin prescription.

        Returns "profile, success"
            profile: relaxation profile object with solution
            success: True or False, whether relaxation method successfully converged
    """
    # Baryon potetial
    Phi_b = outer_halo.Phi_b

    # Step 1: Relaxation method with spherical symmetry
    profile, success_flag = sphmodel.relaxation(
        r1,
        outer_halo,
        init_grid=init_grid,
        max_iter=max_iter,
        r_grid=r_grid,
        verbose=verbose,
        finaltol=finaltol,
    )

    # Step 2: Allow for departure from spherical symmetry with higher modes L>0
    if (L_list != [0]) and success_flag and (converge is None):

        if verbose:
            print("Begin nonspherical relaxation")
            print(" Step 1: Computing nonspherical boundary conditions at r1")

        outer_halo.compute_boundary_conditions(r1, L_list=L_list, M_list=M_list)

        # Include higher L modes and perform relaxation method
        if verbose:
            print(" Step 2: Relaxation with L =", L_list)

        profile.update(angular_moments_set=True, L_list=L_list)
        profile, success_flag = loop_relaxation(profile, outer_halo, max_iter=max_iter, verbose=verbose, tol=finaltol)

    # Step 3: Check convergence
    elif success_flag and (converge is not None):

        # Calculate observable
        obs_old = converge(profile)
        convergence_test = False

        # Tolerance for convergence
        atol = max(10 * finaltol, atol)
        rtol = max(10 * finaltol, rtol)

        if verbose:
            print(" Step 3: Checking for convergence with L.")

        # Increase L modes until convergence reached
        while not (convergence_test) and success_flag:

            L_max = profile.L_list[-1]
            L_list = profile.L_list + [L_max + 1, L_max + 2]

            # Compute new boundary conditions allowing for departure from spherical symmetry
            if verbose:
                print("  Recomputing nonspherical boundary conditions at r1")
            outer_halo.compute_boundary_conditions(r1, L_list=L_list, M_list=M_list)

            if verbose:
                print("  Relaxation with L =", L_list)

            # Include higher L modes and perform relaxation method
            profile.update(angular_moments_set=True, L_list=L_list)
            profile, success_flag = loop_relaxation(
                profile, outer_halo, max_iter=max_iter, verbose=verbose, tol=finaltol
            )

            # Calculate new observable
            obs_new = converge(profile)

            # Test convergence
            convergence_test = np.allclose(obs_old, obs_new, atol=atol, rtol=rtol)

            if verbose:
                print(
                    "  Convergence test with increasing L was a success:",
                    convergence_test,
                )

            # Get ready for next loop
            obs_old = obs_new

            # Test success_flag
            if not (success_flag):
                break

        # Increase r grid until convergence reached
        convergence_test = False

        if verbose:
            print(" Step 4: Checking for convergence with number of grid points.")

        while not (convergence_test) and success_flag:

            # Make new r grid
            num_grid = 2 * len(profile.r_list[1:])
            log_min = np.log10(profile.r_list[1] / profile.r1)
            r_list = np.append([0], r1 * np.logspace(log_min, 0, num=num_grid))

            if verbose:
                print("  Begin relaxation with %d grid points" % num_grid)

            # Include higher L modes and perform relaxation method
            profile.update(angular_moments_set=True, r_list=r_list)
            profile, success_flag = loop_relaxation(
                profile, outer_halo, max_iter=max_iter, verbose=verbose, tol=finaltol
            )

            # Calculate new observable
            obs_new = converge(profile)

            # Test convergence
            convergence_test = np.allclose(obs_old, obs_new, atol=atol, rtol=rtol)

            if verbose:
                print("  Convergence test with increasing grid size:", convergence_test)

            # Get ready for next loop
            obs_old = obs_new

            # Test success_flag
            if not (success_flag):
                break

    if verbose:
        print("End nonspherical relaxation.\n")

    return profile, success_flag


# This performs one iteration of the relaxation method
def iterate_relaxation(profile, outer_halo):

    y = profile.y
    params = profile.params
    L_list = profile.L_list
    M_list = profile.M_list
    Phi_b = profile.Phi_b
    r_list = profile.r_list

    # construct M and E
    M = M_matrix(profile, outer_halo)
    M = sparse.csc_matrix(M)
    E = E_vector(profile, outer_halo)

    # solve for shifts
    y_all = np.concatenate((y, params))
    Delta_y_all = -sparse.linalg.spsolve(M, E)

    kappa = 1
    success_flag = False

    while not success_flag:

        y_all_new = y_all + kappa * Delta_y_all
        y_new = np.array(y_all_new[:-2])
        params_new = np.array(y_all_new[-2:])

        # Check params
        if not np.allclose(params, params_new, rtol=0.3, atol=1e-4):
            kappa = 0.1 * kappa
        # Otherwise all good
        else:
            success_flag = True

    y_new = np.array(y_all_new[:-2])
    params_new = np.array(y_all_new[-2:])

    profile.update(y=y_new, params=params_new)

    return profile, kappa


# This performs a loop of multiple relaxation iterations, terminating
# when result converges within tol or max_iter iterations have occured
def loop_relaxation(profile, outer_halo, max_iter=100, plots=False, verbose=False, tol=1e-4):

    num_iter = 1
    converged_flag = False

    while (num_iter < max_iter) & (not converged_flag):

        old_y_all = np.concatenate((profile.y, profile.params))
        profile, kappa = iterate_relaxation(profile, outer_halo)
        new_y_all = np.concatenate((profile.y, profile.params))

        if kappa < 1:
            converged_flag = False
        else:
            converged_flag = np.allclose(new_y_all, old_y_all, atol=tol, rtol=tol)

        if verbose:
            print("  iteration %d: relaxation stepsize = %f, converged %s" % (num_iter, kappa, str(converged_flag)))

        num_iter += 1

    return profile, converged_flag


#####################################################################
# These functions are needed to construct the M-matrix and E-vector #
#####################################################################


def F(profile, i):

    r0, sigma0 = profile.params**0.5
    Phi_b = profile.Phi_b
    r_list = profile.r_list

    # Phi_b is a function that is called as Phi_b(r,th,phi) or Phi_b(r,th) or Phi_b(r)
    # num_variables=2 for azimuthal symmetry, =3 for arbitrary case, =1 for spherical symmetry
    num_variables = len(signature(Phi_b).parameters)

    if (num_variables == 1) & (profile.L_list == [0]):

        # assume L=M=0 only
        phi_00 = profile.phi()
        mu_00 = profile.mu()

        r_bar = 0.5 * (r_list[i] + r_list[i + 1])
        phi_00_bar = 0.5 * (phi_00[i] + phi_00[i + 1])
        mu_00_bar = 0.5 * (mu_00[i] + mu_00[i + 1])

        term_1 = 0
        phi_dm = phi_00_bar * Z(0, 0, 0, 0)
        phi_b = (Phi_b(r_bar) - Phi_b(0)) / sigma0**2
        term_2 = -(r_bar**2) / r0**2 * 4 * np.pi * Z(0, 0, 0, 0) * np.exp(-phi_b - phi_dm)

    elif (num_variables == 1) & (profile.L_list != [0]):

        # Assume M=0 only
        num_func = profile.num_func
        L_list = np.array(profile.L_list)

        # mean values of phi and mu
        y_bar = 0.5 * (
            profile.y[num_func * i : num_func * (i + 1)] + profile.y[num_func * (i + 1) : num_func * (i + 2)]
        )
        phi_L_bar = y_bar[: len(L_list)]
        mu_L_bar = y_bar[len(L_list) :]

        r_bar = 0.5 * (r_list[i] + r_list[i + 1])
        phi_b = (Phi_b(r_bar) - Phi_b(0)) / sigma0**2

        def phi_dm(theta):
            Z_list = np.array([Z(L, 0, theta, 0) for L in L_list])
            return np.sum(phi_L_bar * Z_list)

        def integrand(theta, L):
            return Z(L, 0, theta, 0) * np.exp(-phi_b - phi_dm(theta)) * np.sin(theta)

        term_1 = -L_list * (L_list + 1) * phi_L_bar
        if profile.angular_moments_set:
            term_2 = -(r_bar**2) / r0**2 * 4 * np.pi * np.array([profile.angular_moments(r_bar, L) for L in L_list])
        else:
            term_2 = (
                -(r_bar**2) / r0**2 * np.array([2 * np.pi * integrate(integrand, 0, np.pi, args=[L]) for L in L_list])
            )

    elif num_variables == 2:

        # Assume M=0 only
        num_func = profile.num_func
        L_list = np.array(profile.L_list)

        # mean values of phi and mu
        y_bar = 0.5 * (
            profile.y[num_func * i : num_func * (i + 1)] + profile.y[num_func * (i + 1) : num_func * (i + 2)]
        )
        phi_L_bar = y_bar[: len(L_list)]
        mu_L_bar = y_bar[len(L_list) :]

        r_bar = 0.5 * (r_list[i] + r_list[i + 1])
        phi_b = lambda theta: (Phi_b(r_bar, theta) - Phi_b(0, theta)) / sigma0**2

        def phi_dm(theta):
            Z_list = np.array([Z(L, 0, theta, 0) for L in L_list])
            return np.sum(phi_L_bar * Z_list)

        def integrand(theta, L):
            return Z(L, 0, theta, 0) * np.exp(-phi_b(theta) - phi_dm(theta)) * np.sin(theta)

        term_1 = -L_list * (L_list + 1) * phi_L_bar
        if profile.angular_moments_set:
            term_2 = -(r_bar**2) / r0**2 * 4 * np.pi * np.array([profile.angular_moments(r_bar, L) for L in L_list])
            # check = - r_bar**2/r0**2 * np.array([ 2*np.pi * integrate( integrand, 0, np.pi, args=[L]) for L in L_list ])
            # print('F', term_2, check)
        else:
            term_2 = (
                -(r_bar**2) / r0**2 * np.array([2 * np.pi * integrate(integrand, 0, np.pi, args=[L]) for L in L_list])
            )

    elif num_variables == 3:

        print(
            "Computing F with",
            num_variables,
            "coordinates not supported. Returning zero.",
        )
        term_1 = np.zeros(len(profile.L_list))
        term_2 = np.zeros(len(profile.L_list))

    output = term_1 + term_2

    return output


def G(profile, i):

    num_func = profile.num_func
    L = profile.L_list

    # mean values of phi and mu
    y_bar = 0.5 * (profile.y[num_func * i : num_func * (i + 1)] + profile.y[num_func * (i + 1) : num_func * (i + 2)])
    mu_L_bar = y_bar[len(L) :]

    r_list = profile.r_list
    r_bar = 0.5 * (r_list[i] + r_list[i + 1])

    output = -mu_L_bar / r_bar**2

    return output


def A_matrix(profile, i):

    r0, sigma0 = profile.params**0.5
    Phi_b = profile.Phi_b

    r_list = profile.r_list
    r_bar = 0.5 * (r_list[i] + r_list[i + 1])
    Delta_r = r_list[i + 1] - r_list[i]

    num_variables = len(signature(Phi_b).parameters)

    if (num_variables == 1) & (profile.L_list == [0]):

        # assume L=M=0 only
        phi_00 = profile.phi()
        mu_00 = profile.mu()

        phi_00_bar = 0.5 * (phi_00[i] + phi_00[i + 1])
        mu_00_bar = 0.5 * (mu_00[i] + mu_00[i + 1])

        phi_b = (Phi_b(r_bar) - Phi_b(0)) / sigma0**2
        phi_dm = phi_00_bar * Z(0, 0, 0, 0)

        dFdphi = r_bar**2 / r0**2 * np.exp(-phi_b - phi_dm)
        dFdmu = 0
        dGdphi = 0
        dGdmu = -1 / r_bar**2

        output = np.array([[dGdphi, dGdmu], [dFdphi, dFdmu]])

    elif (num_variables == 1) & (profile.L_list != [0]):

        num_func = profile.num_func
        L_list = profile.L_list
        num_size = len(L_list)

        y_bar = 0.5 * (
            profile.y[num_func * i : num_func * (i + 1)] + profile.y[num_func * (i + 1) : num_func * (i + 2)]
        )
        phi_L_bar = np.array(y_bar[:num_size])
        mu_L_bar = np.array(y_bar[num_size:])

        phi_b = (Phi_b(r_bar) - Phi_b(0)) / sigma0**2
        phi_dm = lambda th: np.sum([phi_L_bar[a] * Z(L_list[a], 0, th, 0) for a in range(num_size)])

        dFdphi = np.zeros((num_size, num_size))
        for a in range(num_size):
            for b in range(a, num_size):

                # check if angular_moments are set and use existing calculations for a=0
                if profile.angular_moments_set and (a == 0):
                    dFdphi[a, b] = (
                        r_bar**2 / r0**2 * 4 * np.pi * Z(0, 0, 0, 0) * profile.angular_moments(r_bar, L_list[b])
                    )

                else:
                    integrand = (
                        lambda th: Z(L_list[a], 0, th, 0)
                        * Z(L_list[b], 0, th, 0)
                        * np.exp(-phi_b - phi_dm(th))
                        * np.sin(th)
                    )
                    dFdphi[a, b] = r_bar**2 / r0**2 * 2 * np.pi * integrate(integrand, 0, np.pi)

                dFdphi[b, a] = dFdphi[a, b]
                if a == b:
                    L = L_list[a]
                    dFdphi[a, b] += -L * (L + 1)

            dFdmu = np.zeros((num_size, num_size))
            dGdphi = np.zeros((num_size, num_size))
            dGdmu = -1 / r_bar**2 * np.identity(num_size)

        output = np.block([[dGdphi, dGdmu], [dFdphi, dFdmu]])

    if num_variables == 2:

        num_func = profile.num_func
        L_list = profile.L_list
        num_size = len(L_list)

        y_bar = 0.5 * (
            profile.y[num_func * i : num_func * (i + 1)] + profile.y[num_func * (i + 1) : num_func * (i + 2)]
        )
        phi_L_bar = np.array(y_bar[:num_size])
        mu_L_bar = np.array(y_bar[num_size:])

        phi_b = lambda th: (Phi_b(r_bar, th) - Phi_b(0, th)) / sigma0**2
        phi_dm = lambda th: np.sum([phi_L_bar[a] * Z(L_list[a], 0, th, 0) for a in range(num_size)])

        dFdphi = np.zeros((num_size, num_size))
        for a in range(num_size):
            for b in range(a, num_size):

                # check if angular_moments are set and use existing calculations for a=0
                if profile.angular_moments_set and (a == 0):
                    dFdphi[a, b] = (
                        r_bar**2 / r0**2 * 4 * np.pi * Z(0, 0, 0, 0) * profile.angular_moments(r_bar, L_list[b])
                    )

                    # check:
                    # integrand = lambda th: Z(L_list[a],0,th,0)*Z(L_list[b],0,th,0)*np.exp( - phi_b(th) - phi_dm(th) )*np.sin(th)
                    # check = r_bar**2/r0**2 * 2*np.pi * integrate(integrand, 0, np.pi)
                    # print('A', dFdphi[a,b], check)

                else:
                    integrand = (
                        lambda th: Z(L_list[a], 0, th, 0)
                        * Z(L_list[b], 0, th, 0)
                        * np.exp(-phi_b(th) - phi_dm(th))
                        * np.sin(th)
                    )
                    dFdphi[a, b] = r_bar**2 / r0**2 * 2 * np.pi * integrate(integrand, 0, np.pi)

                dFdphi[b, a] = dFdphi[a, b]
                if a == b:
                    L = L_list[a]
                    dFdphi[a, b] += -L * (L + 1)

            dFdmu = np.zeros((num_size, num_size))
            dGdphi = np.zeros((num_size, num_size))
            dGdmu = -1 / r_bar**2 * np.identity(num_size)

        output = np.block([[dGdphi, dGdmu], [dFdphi, dFdmu]])

    elif num_variables == 3:

        print(
            "Computing A with",
            num_variables,
            "coordinates not supported. Returning zero.",
        )
        output = np.zeros((num_func, num_func))

    return 0.5 * Delta_r * output


def B_matrix(profile, i):

    r0, sigma0 = profile.params**0.5
    Phi_b = profile.Phi_b

    r_list = profile.r_list
    r_bar = 0.5 * (r_list[i] + r_list[i + 1])
    Delta_r = r_list[i + 1] - r_list[i]

    num_variables = len(signature(Phi_b).parameters)

    if (num_variables == 1) & (profile.L_list == [0]):

        # assume L=M=0 only
        phi_00 = profile.phi()
        mu_00 = profile.mu()

        phi_00_bar = 0.5 * (phi_00[i] + phi_00[i + 1])

        phi_b = (Phi_b(r_bar) - Phi_b(0)) / sigma0**2
        phi_dm = phi_00_bar * Z(0, 0, 0, 0)

        dFdr0_sq = r_bar**2 / r0**4 * np.exp(-phi_b - phi_dm) * np.sqrt(4 * np.pi)
        dGdr0_sq = 0

        dFdsigma0_sq = -(r_bar**2) / r0**2 / sigma0**2 * np.sqrt(4 * np.pi) * phi_b * np.exp(-phi_b - phi_dm)
        dGdsigma0_sq = 0

        output = np.array([[dGdr0_sq, dGdsigma0_sq], [dFdr0_sq, dFdsigma0_sq]])

    elif (num_variables == 1) & (profile.L_list != [0]):

        num_func = profile.num_func
        L_list = profile.L_list
        num_size = len(L_list)

        y_bar = 0.5 * (
            profile.y[num_func * i : num_func * (i + 1)] + profile.y[num_func * (i + 1) : num_func * (i + 2)]
        )
        phi_L_bar = np.array(y_bar[:num_size])
        mu_L_bar = np.array(y_bar[num_size:])

        phi_b = (Phi_b(r_bar) - Phi_b(0)) / sigma0**2
        phi_dm = lambda th: np.sum([phi_L_bar[a] * Z(L_list[a], 0, th, 0) for a in range(num_size)])

        dFdr0_sq = np.zeros((num_size, 1))
        dFdsigma0_sq = np.zeros((num_size, 1))
        dGdr0_sq = np.zeros((num_size, 1))
        dGdsigma0_sq = np.zeros((num_size, 1))

        for a in range(num_size):

            if profile.angular_moments_set:
                dFdr0_sq[a] = r_bar**2 / r0**4 * 4 * np.pi * profile.angular_moments(r_bar, L_list[a])

            else:
                integrand = lambda th: Z(L_list[a], 0, th, 0) * np.exp(-phi_b - phi_dm(th)) * np.sin(th)
                dFdr0_sq[a] = r_bar**2 / r0**4 * 2 * np.pi * integrate(integrand, 0, np.pi)

            integrand = lambda th: Z(L_list[a], 0, th, 0) * phi_b * np.exp(-phi_b - phi_dm(th)) * np.sin(th)
            dFdsigma0_sq[a] = -(r_bar**2) / r0**2 / sigma0**2 * 2 * np.pi * integrate(integrand, 0, np.pi)

        output = np.block([[dGdr0_sq, dGdsigma0_sq], [dFdr0_sq, dFdsigma0_sq]])

    elif num_variables == 2:

        num_func = profile.num_func
        L_list = profile.L_list
        num_size = len(L_list)

        y_bar = 0.5 * (
            profile.y[num_func * i : num_func * (i + 1)] + profile.y[num_func * (i + 1) : num_func * (i + 2)]
        )
        phi_L_bar = np.array(y_bar[:num_size])
        mu_L_bar = np.array(y_bar[num_size:])

        phi_b = lambda th: (Phi_b(r_bar, th) - Phi_b(0, th)) / sigma0**2
        phi_dm = lambda th: np.sum([phi_L_bar[a] * Z(L_list[a], 0, th, 0) for a in range(num_size)])

        dFdr0_sq = np.zeros((num_size, 1))
        dFdsigma0_sq = np.zeros((num_size, 1))
        dGdr0_sq = np.zeros((num_size, 1))
        dGdsigma0_sq = np.zeros((num_size, 1))

        for a in range(num_size):

            if profile.angular_moments_set:
                dFdr0_sq[a] = r_bar**2 / r0**4 * 4 * np.pi * profile.angular_moments(r_bar, L_list[a])
                # check
                # integrand = lambda th: Z(L_list[a],0,th,0)*np.exp( - phi_b(th) - phi_dm(th) )*np.sin(th)
                # check = r_bar**2/r0**4 * 2*np.pi * integrate(integrand, 0, np.pi)
                # print('B', dFdr0_sq[a], check)

            else:
                integrand = lambda th: Z(L_list[a], 0, th, 0) * np.exp(-phi_b(th) - phi_dm(th)) * np.sin(th)
                dFdr0_sq[a] = r_bar**2 / r0**4 * 2 * np.pi * integrate(integrand, 0, np.pi)

            integrand = lambda th: Z(L_list[a], 0, th, 0) * phi_b(th) * np.exp(-phi_b(th) - phi_dm(th)) * np.sin(th)
            dFdsigma0_sq[a] = -(r_bar**2) / r0**2 / sigma0**2 * 2 * np.pi * integrate(integrand, 0, np.pi)

        output = np.block([[dGdr0_sq, dGdsigma0_sq], [dFdr0_sq, dFdsigma0_sq]])

    elif num_variables == 3:

        print(
            "Computing B with",
            num_variables,
            "coordinates not supported. Returning zero.",
        )
        output = np.zeros((num_func, 2))

    return Delta_r * output


def C1_matrix(profile):

    num_func = profile.num_func
    num_modes = len(profile.L_list)
    num_params = len(profile.params)

    C1 = np.zeros((num_func + num_params, num_func))

    # BC for phi_LM at r=0
    for i in range(num_modes):
        C1[i, i] = 1

    # BC for mu_00 at r=0
    C1[-2, num_modes] = 1

    return C1


def CN_matrix(profile):

    num_func = profile.num_func
    num_modes = len(profile.L_list)
    num_params = len(profile.params)

    r1 = profile.r1

    CN = np.zeros((num_func + num_params, num_func))

    # BC for mean rho at r=r1
    if profile.angular_moments_set:
        CN[num_modes, :num_modes] = -profile.rho0 * profile.angular_moments_list[-1]

    else:
        for i, L in enumerate(profile.L_list):
            integrand = lambda th: np.sin(th) * Z(L, 0, th, 0) * profile.rho_dm_sph(r1, th)
            CN[num_modes, i] = -0.5 * integrate(integrand, 0, np.pi)

    # BC for phi_LM at r=r1 (for L>0)
    for i in range(1, num_modes):

        L = profile.L_list[i]
        CN[num_modes + i, i] = r1 * (L + 1)
        CN[num_modes + i, num_modes + i] = 1

    # BC for mu_00 at r=r1
    CN[-1, num_modes] = 1

    return CN


def D_matrix(profile, outer_halo):

    num_func = profile.num_func
    num_modes = len(profile.L_list)
    num_params = len(profile.params)

    Phi_b = profile.Phi_b
    num_variables = len(signature(Phi_b).parameters)

    r0 = profile.r0
    sigma0 = profile.sigma0
    rN = profile.r1
    M1 = profile.M_encl_list()[-1]

    D = np.zeros((num_func + num_params, num_params))

    # BC for mean rho at r1
    if (num_variables == 1) & (profile.L_list == [0]):

        # assume L=M=0 only
        phi_N_00 = profile.phi()[-1]
        phi_dm = phi_N_00 * Z(0, 0, 0, 0)
        phi_b = (Phi_b(rN) - Phi_b(0)) / sigma0**2

        rho = profile.rho0 * np.exp(-phi_dm - phi_b)

        D[num_modes, :] = np.array([-rho / r0**2, rho * (1 + phi_b) / sigma0**2])

    elif (num_variables == 1) & (profile.L_list != [0]):

        L_list = profile.L_list

        phi_N_list = profile.y[-num_func : -num_func + num_modes]
        phi_dm = lambda th: np.sum([phi_N_list[i] * Z(L_list[i], 0, th, 0) for i in range(num_modes)])
        phi_b = (Phi_b(rN) - Phi_b(0)) / sigma0**2

        if profile.angular_moments_set:
            rho_mean = profile.rho0 * profile.angular_moments_list[-1, 0] / Z(0, 0, 0, 0)
        else:
            rho = lambda th: profile.rho0 * np.exp(-phi_dm(th) - phi_b)
            integrand = lambda th: rho(th) * np.sin(th)
            rho_mean = 0.5 * integrate(integrand, 0, np.pi)

        D[num_modes, :] = np.array([-rho_mean / r0**2, rho_mean * (1 + phi_b) / sigma0**2])

    elif num_variables == 2:

        L_list = profile.L_list

        phi_N_list = profile.y[-num_func : -num_func + num_modes]
        phi_dm = lambda th: np.sum([phi_N_list[i] * Z(L_list[i], 0, th, 0) for i in range(num_modes)])
        phi_b = lambda th: (Phi_b(rN, th) - Phi_b(0, th)) / sigma0**2

        rho = lambda th: profile.rho0 * np.exp(-phi_dm(th) - phi_b(th))
        integrand = lambda th: np.sin(th) * (1 + phi_b(th)) * rho(th)
        second_term = 0.5 * integrate(integrand, 0, np.pi) / sigma0**2

        if profile.angular_moments_set:
            rho_mean = profile.rho0 * profile.angular_moments_list[-1, 0] / Z(0, 0, 0, 0)
        else:
            integrand = lambda th: np.sin(th) * rho(th)
            rho_mean = 0.5 * integrate(integrand, 0, np.pi)

        D[num_modes, :] = np.array([-rho_mean / r0**2, second_term])

    # BC for mu_LM, phi_LM at r1 (L>0 only)
    for i in range(1, num_modes):

        L = profile.L_list[i]

        D[num_modes + i, 1] = -4 * np.pi * GN / sigma0**4 * rN ** (L + 1) * outer_halo.potential_moments[i]

    # BC for mu_00 at r1
    D[-1, -1] = np.sqrt(4 * np.pi) * GN * M1 / sigma0**4

    return D


def M_matrix(profile, outer_halo):

    # Information for nonspherical boundary conditions
    # rho1(Omega) is a function
    # M1 = M_encl(r1)

    num_params = len(profile.params)
    num_func = profile.num_func
    num_r = profile.num_r_points
    num_tot = num_func * num_r + num_params

    # initialize matrix M as sparse array
    M_out = sparse.dok_matrix((num_tot, num_tot))

    # Populate A and B terms in M
    for i in range(num_r - 1):

        # define slices in matrix
        i0 = num_func * i
        i1 = num_func * (i + 1)

        # populate A blocks
        A = A_matrix(profile, i)
        AL = -np.identity(num_func) + A
        AR = np.identity(num_func) + A
        for j in range(num_func):
            for k in range(num_func):

                M_out[i0 + j, i0 + k] = AL[j, k]
                M_out[i0 + j, i1 + k] = AR[j, k]

        # populate B blocks
        B = B_matrix(profile, i)
        for j in range(num_func):
            for k in range(num_params):

                M_out[i0 + j, num_func * num_r + k] = B[j, k]

    # populate boundary conditions
    C1 = C1_matrix(profile)
    CN = CN_matrix(profile)
    D = D_matrix(profile, outer_halo)

    # Row where BCs start in M
    row_num = (num_r - 1) * num_func

    M_out[row_num : row_num + num_func + num_params, :num_func] = C1
    M_out[
        row_num : row_num + num_func + num_params,
        ((num_r - 1) * num_func) : (num_r * num_func),
    ] = CN
    M_out[row_num : row_num + num_func + num_params, (num_r * num_func) :] = D

    return M_out


def H_vector(profile, outer_halo):

    num_params = len(profile.params)
    num_func = profile.num_func
    num_r = profile.num_r_points
    num_size = len(profile.L_list)

    Phi_b = profile.Phi_b
    rho0 = profile.rho0

    H_out = np.zeros(num_func + num_params)

    r0, sigma0 = np.array(profile.params) ** 0.5
    rho0 = profile.rho0
    rN = profile.r1

    # rho1 = profile.rho_sph_list()[-1]
    # M1 = profile.M_encl_list()[-1]
    rho1 = outer_halo.rho_sph_avg(rN)
    M1 = outer_halo.M_encl(rN)

    # boundary conditions for phi_LM at r=0
    phi_LM_0 = profile.y[:num_size]
    H_out[:num_size] = phi_LM_0

    # boundary condition for mean density ar r1
    if profile.angular_moments_set:
        rho_mean = profile.rho0 * profile.angular_moments_list[-1, 0] / Z(0, 0, 0, 0)
    else:
        rho_mean = profile.rho_dm_sph(rN)
    H_out[num_size] = rho_mean - rho1

    # boundary conditions from matching potential onto correct asymptotic form at infinity
    phi_LM_N = profile.y[-num_func : -num_func + num_size]
    mu_LM_N = profile.y[-num_func + num_size :]
    L = np.array(profile.L_list)
    J_LM = outer_halo.potential_moments

    boundary_conditions = mu_LM_N + rN * (L + 1) * phi_LM_N + 4 * np.pi * GN / sigma0**2 * rN ** (L + 1) * J_LM
    H_out[num_size + 1 : num_func] = boundary_conditions[1:]

    # boundary conditions for mu_00 at r=0
    mu_00_0 = profile.y[num_size]
    H_out[-2] = mu_00_0

    # boundary conditions for mu_00 at r=rN
    mu_00_N = profile.y[-num_size]

    H_out[-1] = mu_00_N - np.sqrt(4 * np.pi) * GN * M1 / sigma0**2

    return H_out


def E_vector(profile, outer_halo):

    num_params = len(profile.params)
    num_func = profile.num_func
    num_r = profile.num_r_points
    num_tot = num_func * num_r + num_params

    E_out = np.zeros(num_tot)

    y = profile.y
    r_list = profile.r_list

    for i in range(num_r - 1):

        F_vec = F(profile, i)
        G_vec = G(profile, i)

        Delta_y = y[num_func * (i + 1) : num_func * (i + 2)] - y[num_func * i : num_func * (i + 1)]
        Delta_r = r_list[i + 1] - r_list[i]
        E_out[num_func * i : num_func * (i + 1)] = Delta_y + Delta_r * np.block([G_vec, F_vec])

    rN = profile.r1
    H = H_vector(profile, outer_halo)
    E_out[-len(H) :] = H

    return E_out
