"""
spherical.py
------------
Purpose:   Spherical Jeans modeling and relaxation routines for SIDM and CDM halos.
Authors:   Sean Tulin, Adam Smith-Orlik
Contact:   stulin@yorku.ca, asorlik@yorku.ca
Status:    Stable Version
Last Edit: 2025-09-16

This file contains the main relaxation solver and supporting routines for spherical halo modeling in the nonspherical SIDM Jeans modeling package.
"""

######################################################################
############################## IMPORTS ###############################
######################################################################
import numpy as np
import pandas as pd

# import pkg_resources
import importlib.resources as pkg_resources
from inspect import signature
from scipy.interpolate import interp1d
import scipy.sparse as sparse
import time

from . import data

from .definitions import GN, integrate
from .classes import isothermal_profile
from .tools import timed


########################################################################
####################### Main relaxation code: ##########################
# Solve for the isothermal profile given CDM boundary conditions at r1 #
########################################################################
@timed  # Uncomment to time function
def relaxation(
    r1,
    outer_halo,
    Phi_b=None,
    max_iter=100,
    init_grid=5,
    r_grid=200,
    verbose=False,
    finaltol=1e-8,
    **extraneous,
):

    rho1 = outer_halo.rho_sph(r1)
    M1 = outer_halo.M_encl(r1)
    matching = {"r1": r1, "rho1": rho1, "M1": M1}

    # If Phi_b not specified, use Phi_b from outer_halo
    if Phi_b is None:
        Phi_b = outer_halo.Phi_b

    num_variables = len(signature(Phi_b).parameters)

    # Step 1: burn-in the baryon density with a coarse grid
    # assume spherical symmetry and no baryons
    r_list = np.append([0], r1 * np.logspace(-4, 0, num=init_grid))

    Upsilon = 1  # Reduce this to relax more gradually from zero baryons to full baryon density
    num_iter = 1
    success_flag = False

    if verbose:
        print("Begin spherically-symmetric relaxation with coarse grid (%d r points)." % init_grid)
        print(" Step 1: Start with DM-only solution and increase baryon fraction from 0 to 1.")

    while (not success_flag) & (num_iter < max_iter):

        # rescale baryon potential
        if num_variables == 1:
            Phi_b_Upsilon = lambda r: Upsilon * Phi_b(r)
        elif num_variables == 2:
            Phi_b_Upsilon = lambda r, th: Upsilon * Phi_b(r, th)
        else:
            print("Case with", num_variables, "variables not supported. Setting Phi_b=0")
            Phi_b_Upsilon = lambda r: 0

        # initialize
        y, params = initialize_y(rho1, M1, r_list, Phi_b=Phi_b_Upsilon)

        # try to converge
        y, params, success_flag = loop_relaxation(y, params, r_list, matching, Phi_b_Upsilon)

        if verbose:
            print("  iteration %d: baryon fraction = %f, was a success: %s" % (num_iter, Upsilon, str(success_flag)))

        if not success_flag:
            Upsilon = 0.5 * Upsilon

        num_iter += 1

    # Step 1.5: increase Upsilon
    # Not needed if starting out at "full baryons" (Upsilon=1)

    delta_Upsilon = 1.5

    while (Upsilon < 1) & (num_iter < max_iter):

        Upsilon_new = min(Upsilon * delta_Upsilon, 1)

        # rescale baryon potential
        if num_variables == 1:
            Phi_b_Upsilon = lambda r: Upsilon_new * Phi_b(r)
        elif num_variables == 2:
            Phi_b_Upsilon = lambda r, th: Upsilon_new * Phi_b(r, th)

        y_new, params_new, success_flag = loop_relaxation(y, params, r_list, matching, Phi_b_Upsilon)

        if verbose:
            print("  iteration %d: baryon fraction = %f was a success: %s" % (num_iter, Upsilon, str(success_flag)))

        if success_flag:
            y, params, Upsilon = y_new, params_new, Upsilon_new
        else:
            delta_Upsilon = np.sqrt(delta_Upsilon)

        num_iter += 1

    # Step 2: Let solution converge with finer grid

    if verbose:
        print(" Step 2: Increase number of grid points using interpolated coarse-grid result as initial guess.")

    grid_size = min(2 * init_grid, r_grid)

    while grid_size <= r_grid:

        # make interpolating functions
        phi_int = interp1d(r_list, y[::2])
        eta_int = interp1d(r_list, y[1::2])

        # make new y
        r_list = np.append([0], r1 * np.logspace(-4, 0, num=grid_size))
        phi_new = np.array([phi_int(r) for r in r_list])
        eta_new = np.array([eta_int(r) for r in r_list])
        y = np.zeros(2 * len(phi_new))
        y[::2] = phi_new
        y[1::2] = eta_new

        y_new, params_new, success_flag = loop_relaxation(y, params, r_list, matching, Phi_b)

        if verbose:
            print(
                "  iteration %d: number of grid points = %d, was a success: %s"
                % (num_iter, grid_size, str(success_flag))
            )

        num_iter += 1

        # If successful convergence and reached max grid_size, end while loop
        if success_flag & (grid_size == r_grid):
            y, params = y_new, params_new
            break

        # Elif sucessful convergence and still need to increase grid size
        elif success_flag:
            grid_size = min(2 * grid_size, r_grid)
            y, params = y_new, params_new

        # Else failed convergence, exit
        # Allowing more iterations to converge doesn't seem to help
        else:
            if verbose:
                print("Relaxation was not successful.")
            break

    if success_flag:

        # Keep iterating until converged with smaller tolerance
        if verbose:
            print(" Step 3: Continue relaxation until required tolerance %s achieved." % str(finaltol))

        y, params, success_flag = loop_relaxation(y, params, r_list, matching, Phi_b, tol=finaltol)

        if verbose:
            print("  iteration %d: was a success: %s" % (num_iter, str(success_flag)))

    # Make output class instance
    # Need to convert y = [phi, eta] to new_y = [phi_00, mu_00]
    r0 = params[0] ** 0.5
    sigma0 = params[1] ** 0.5
    rho0 = sigma0**2 / (4 * np.pi * GN * r0**2)

    phi = y[::2]
    phi_00 = phi * np.sqrt(4 * np.pi)
    eta = y[1::2]
    M_encl = 4 * np.pi / 3 * rho0 * r_list**3 * np.exp(-eta)
    mu_00 = np.sqrt(4 * np.pi) * GN / sigma0**2 * M_encl

    new_y = np.zeros_like(y)
    new_y[::2] = phi_00
    new_y[1::2] = mu_00

    profile_out = isothermal_profile(new_y, params, r_list, Phi_b=Phi_b)

    if verbose:
        print("End spherically-symmetric relaxation.\n")

    return profile_out, success_flag


# Auxillary functions for relaxation matrices
def A_matrix(phi, eta, params, r, phi_b_moments):

    # phi_b_moment_0 = 1/(4pi) * int dOmega * exp( -phi_b )
    # phi_b_moment_1 = 1/(4pi) * int dOmega * phi_b * exp( - phi_b )

    r0, sigma0 = np.array(params) ** 0.5
    phi_b_moment_0 = phi_b_moments[0]

    dF_dphi = 0
    dF_deta = r / (3 * r0**2) * np.exp(-eta)

    dG_dphi = -3 / r * np.exp(eta - phi) * phi_b_moment_0
    dG_deta = 3 / r * np.exp(eta - phi) * phi_b_moment_0

    output = np.array([[dF_dphi, dF_deta], [dG_dphi, dG_deta]])

    return output


def B_matrix(phi, eta, params, r, phi_b_moments):

    # phi_b_moment_0 = 1/(4pi) * int dOmega * exp( -phi_b )
    # phi_b_moment_1 = 1/(4pi) * int dOmega * phi_b * exp( - phi_b )

    r0, sigma0 = np.array(params) ** 0.5
    phi_b_moment_1 = phi_b_moments[1]

    dF_dr0_sq = r / (3 * r0**4) * np.exp(-eta)
    dF_dsigma0_sq = 0

    dG_dr0_sq = 0
    dG_dsigma0_sq = (3 / r) * 1 / sigma0**2 * np.exp(eta - phi) * phi_b_moment_1

    output = np.array([[dF_dr0_sq, dF_dsigma0_sq], [dG_dr0_sq, dG_dsigma0_sq]])

    return output


def C_matrix(phi_N, eta_N, params, rN, phi_b_moments):

    r0, sigma0 = np.array(params) ** 0.5
    rho0 = sigma0**2 / (4 * np.pi * GN * r0**2)
    phi_b_moment_0 = phi_b_moments[0]

    r1 = rN  # Assume r1 = max point on grid

    c11 = -rho0 * np.exp(-phi_N) * phi_b_moment_0
    c12 = 0
    c21 = 0
    c22 = -4 * np.pi / 3 * rho0 * r1**3 * np.exp(-eta_N)

    output = np.array([[c11, c12], [c21, c22]])

    return output


def D_matrix(phi_N, eta_N, params, rN, phi_b_moments):

    r0, sigma0 = np.array(params) ** 0.5
    rho0 = sigma0**2 / (4 * np.pi * GN * r0**2)
    phi_b_moment_0, phi_b_moment_1 = phi_b_moments

    r1 = rN  # Assume r1 = max point on grid

    d11 = -rho0 / r0**2 * np.exp(-phi_N) * phi_b_moment_0
    d12 = rho0 / sigma0**2 * np.exp(-phi_N) * (phi_b_moment_0 + phi_b_moment_1)
    d21 = -4 * np.pi / 3 * rho0 * r1**3 * np.exp(-eta_N) * 1 / r0**2
    d22 = 4 * np.pi / 3 * rho0 * r1**3 * np.exp(-eta_N) * 1 / sigma0**2

    output = np.array([[d11, d12], [d21, d22]])

    return output


def M_matrix(y, params, r_list, matching, phi_b_moment_func):

    # rho_dm, M_dm at rmax=r1
    rho1 = matching["rho1"]
    M1 = matching["M1"]

    num_r = len(r_list)
    num_tot = 2 * num_r + 2

    # Initialize matrix M as sparse array
    M_out = sparse.dok_matrix((num_tot, num_tot))

    # Populate A and B terms in M
    for i in range(num_r - 1):

        # Mean values of phi and eta
        # Assume y = (phi0, eta0, phi1, eta1, ... , phiN, etaN )
        phi_bar = 0.5 * (y[2 * i] + y[2 * (i + 1)])
        eta_bar = 0.5 * (y[2 * i + 1] + y[2 * (i + 1) + 1])

        # Mean value of r
        ri = 0.5 * (r_list[i] + r_list[i + 1])
        Delta_r = r_list[i + 1] - r_list[i]

        # Calculate moments of phi_b
        phi_b_moments = phi_b_moment_func(ri)

        # populate A blocks
        A = 0.5 * Delta_r * A_matrix(phi_bar, eta_bar, params, ri, phi_b_moments)
        AL = -np.identity(2) + A
        AR = np.identity(2) + A
        for j in range(2):
            for k in range(2):

                M_out[2 * i + j, 2 * i + k] = AL[j, k]
                M_out[2 * i + j, 2 * (i + 1) + k] = AR[j, k]

        # populate B blocks
        B = Delta_r * B_matrix(phi_bar, eta_bar, params, ri, phi_b_moments)
        for j in range(2):
            for k in range(2):

                M_out[2 * i + j, 2 * num_r + k] = B[j, k]

    # Populate boundary conditions

    # Functions at r=rN
    phi_N, eta_N = y[-2:]
    rN = r_list[-1]
    phi_b_moments = phi_b_moment_func(rN)

    # Identity block
    # phi = 0 and eta = 0 at r=0
    for j in range(2):
        M_out[(num_r - 1) * 2 + j, j] = 1

    # C block
    C = C_matrix(phi_N, eta_N, params, rN, phi_b_moments)

    for j in range(2):
        for k in range(2):

            M_out[2 * num_r + j, 2 * (num_r - 1) + k] = C[j, k]

    # D block
    D = D_matrix(phi_N, eta_N, params, rN, phi_b_moments)

    for j in range(2):
        for k in range(2):
            M_out[2 * num_r + j, 2 * num_r + k] = D[j, k]

    return M_out


def H_vector(y, params, r_list, matching, phi_b_moment_func):

    num_r = len(r_list)

    H_out = np.zeros(4)

    rho1 = matching["rho1"]
    M1 = matching["M1"]

    r0, sigma0 = np.array(params) ** 0.5
    rho0 = sigma0**2 / (4 * np.pi * GN * r0**2)

    # boundary conditions at r=0
    phi_0, eta_0 = y[:2]

    H_out[0] = phi_0
    H_out[1] = eta_0

    # boundary conditions at r=rN
    phi_N, eta_N = y[-2:]
    rN = r_list[-1]
    r1 = rN
    phi_b_moment_0 = phi_b_moment_func(rN)[0]

    H_out[2] = rho0 * np.exp(-phi_N) * phi_b_moment_0 - rho1
    H_out[3] = 4 * np.pi / 3 * rho0 * r1**3 * np.exp(-eta_N) - M1

    return H_out


def E_vector(y, params, r_list, matching, phi_b_moment_func):

    r0, sigma0 = np.array(params) ** 0.5

    num_r = len(r_list)
    num_tot = 2 * num_r + 2

    E_out = np.zeros(num_tot)

    # Construct vec{E}_i terms
    for i in range(num_r - 1):

        phi_bar = 0.5 * (y[2 * i] + y[2 * (i + 1)])
        eta_bar = 0.5 * (y[2 * i + 1] + y[2 * (i + 1) + 1])
        r_bar = 0.5 * (r_list[i] + r_list[i + 1])

        phi_b_moment_0 = phi_b_moment_func(r_bar)[0]

        F = -r_bar / (3 * r0**2) * np.exp(-eta_bar)
        G = -3 / r_bar * (1 - np.exp(eta_bar - phi_bar) * phi_b_moment_0)

        Delta_phi = y[2 * (i + 1)] - y[2 * i]
        Delta_eta = y[2 * (i + 1) + 1] - y[2 * i + 1]
        Delta_r = r_list[i + 1] - r_list[i]

        E1 = Delta_phi + Delta_r * F
        E2 = Delta_eta + Delta_r * G

        E_out[2 * i] = E1
        E_out[2 * i + 1] = E2

    rN = r_list[-1]
    H = H_vector(y, params, r_list, matching, phi_b_moment_func)
    E_out[-len(H) :] = H

    return E_out


# Generate initial value of y from isothermal Jeans model without baryons


def initialize_y(rho1, M1, r_list, Phi_b=None):

    # Assume last r point is r1
    r1 = r_list[-1]

    # Read in data table for numerical solution for dimensionless Jeans equation
    # Expressed as two equations for h(x) and j(x)
    # where x = r/r0 and h = log(rho/rho0)
    # and j = log(rho_mean/rho0) i.e. M(r) = 4pi/3 * rho_mean * r^3
    # Jeans equation becomes
    # h'(x) = -x/3 * e^j(x) and j'(x) = 3/x * (e^(h(x)-j(x)) - 1)
    # stream = pkg_resources.resource_stream(__name__, 'initial_guess.csv')

    # # Avoid two processes reading the file at the same time
    # while True:
    #     try:
    #         df = pd.read_csv(stream)
    #         break
    #     except pandas.errors.EmptyDataError:
    #         sec = 3
    #         print('Waiting %d seconds...' % sec)
    #         time.sleep(sec)

    while True:
        try:
            with pkg_resources.files(data).joinpath("initial_guess.csv").open("r") as stream:
                df = pd.read_csv(stream)
            break
        except pd.errors.EmptyDataError:
            sec = 3
            print(f"Waiting {sec} seconds...")
            time.sleep(sec)

    x = np.array(df["x"])
    h = np.array(df["h"])
    j = np.array(df["j"])
    h_int = interp1d(x, h, bounds_error=False, fill_value=(0, h[-1]))
    j_int = interp1d(x, j, bounds_error=False, fill_value=(0, j[-1]))

    # Define ratio = M1/(4pi*r1^3*rho1)
    max_ratio = np.exp(j[-1] - h[-1]) / 3
    # ratio cannot match if larger than ~ max_ratio = 1.26...
    # choose max value as initial guess if wouldn't match otherwise
    ratio = min(M1 / (4 * np.pi * r1**3 * rho1), max_ratio)

    # determine x1 = r1/r0
    x1_int = interp1d(j - h, x)
    x1 = x1_int(np.log(3 * ratio))

    # determine rho0, sigma0, r0
    r0 = r1 / x1
    rho0 = rho1 * np.exp(-h_int(x1))
    sigma0 = np.sqrt(4 * np.pi * GN * rho0) * r0

    params_out = [r0**2, sigma0**2]

    # make output
    phi_list = np.array([-h_int(r / r0) for r in r_list])
    eta_list = np.array([-j_int(r / r0) for r in r_list])

    # Perform shift in phi
    # Default is no shift unless Phi_b is input
    if Phi_b:
        phi_b_moments = make_phi_b_moment_func(Phi_b, params_out, r_list)
        phi_b_list = np.array([-np.log(phi_b_moments(r)[0]) for r in r_list])
    else:
        phi_b_list = np.zeros_like(r_list, "f")

    y_out = np.zeros(2 * len(phi_list))
    y_out[::2] = phi_list - phi_b_list
    y_out[1::2] = eta_list

    return y_out, params_out


# Calculate angular integrals of baryon potential
def make_phi_b_moment_func(Phi_b, params, r_list):

    sigma0 = params[1] ** 0.5

    # Case where Phi_b depends only on r
    if len(signature(Phi_b).parameters) == 1:

        phi_b = lambda r: (Phi_b(r) - Phi_b(0)) / sigma0**2
        moments = lambda r: [np.exp(-phi_b(r)), np.exp(-phi_b(r)) * phi_b(r)]

    elif len(signature(Phi_b).parameters) == 2:

        r_points = np.concatenate(([r_list[0]], 0.5 * (r_list[:-1] + r_list[1:]), [r_list[-1]]))

        phi_b = lambda r, th: (Phi_b(r, th) - Phi_b(0, th)) / sigma0**2

        integrand_0 = lambda th, r: np.exp(-phi_b(r, th)) * np.sin(th)
        integrand_1 = lambda th, r: phi_b(r, th) * np.exp(-phi_b(r, th)) * np.sin(th)

        moment_0_list = [0.5 * integrate(integrand_0, 0, np.pi, args=[r]) for r in r_points]
        moment_1_list = [0.5 * integrate(integrand_1, 0, np.pi, args=[r]) for r in r_points]

        moment_0_int = interp1d(r_points, moment_0_list)
        moment_1_int = interp1d(r_points, moment_1_list)

        moments = lambda r: [moment_0_int(r), moment_1_int(r)]

    else:
        print("Case not supported. Assume Phi_b = 0.")
        moments = lambda r: [0, 0]

    return moments


# Relaxation step
def iterate_relaxation(y, params, r_list, matching, Phi_b):

    # Construct phi_b_moments
    phi_b_moment_func = make_phi_b_moment_func(Phi_b, params, r_list)

    # construct M and E
    M = M_matrix(y, params, r_list, matching, phi_b_moment_func)
    M = sparse.csc_matrix(M)
    E = E_vector(y, params, r_list, matching, phi_b_moment_func)

    # solve for shifts
    y_all = np.concatenate((y, params))
    Delta_y_all = -sparse.linalg.spsolve(M, E)

    kappa = 1
    success_flag = False

    while not success_flag:

        y_all_new = y_all + kappa * Delta_y_all

        params_new = np.array(y_all_new[-2:])

        if not np.allclose(params_new, params, rtol=0.5, atol=1e-8):
            kappa = 0.1 * kappa
        else:
            success_flag = True

    y_new = np.array(y_all_new[:-2])
    params_new = np.array(y_all_new[-2:])

    return y_new, params_new, kappa


# Loop relaxation step until convergence achieved
def loop_relaxation(y_init, params_init, r_list, matching, Phi_b, max_iter=100, tol=1e-4):

    num_iter = 1
    converged_flag = False

    y = y_init
    params = params_init

    while (num_iter < max_iter) & (not converged_flag):

        old_y_all = np.concatenate((y, params))
        y, params, kappa = iterate_relaxation(y, params, r_list, matching, Phi_b)
        new_y_all = np.concatenate((y, params))

        if kappa < 1:
            converged_flag = False
        else:
            converged_flag = np.allclose(new_y_all, old_y_all, atol=tol, rtol=tol)

        num_iter += 1

    return y, params, converged_flag
