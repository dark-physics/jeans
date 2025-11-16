"""
cdm.py
------
Purpose:   NFW profile and adiabatic contraction model definitions for CDM halos. Used as the outer halo in the Jeans model.
Authors:   Sean Tulin, Adam Smith-Orlik
Contact:   stulin@yorku.ca, asorlik@yorku.ca
Status:    Stable Version
Last Edit: 2025-09-16
References: [1] Navarro, Frenk & White 1997 (https://arxiv.org/abs/astro-ph/9611107v4)
            [2] Gnedin et al. 2004 (https://arxiv.org/abs/astro-ph/0406247)
            [3] Cautun et al. 2014 (https://arxiv.org/abs/1911.04557)
            [4] E. Retana-Montenegro et al. 2012 (https://arxiv.org/abs/1202.5242)

This file contains NFW and adiabatic contraction profile functions, mass/concentration conversions, and related utilities for CDM halos.
"""

######################################################################
############################## IMPORTS ###############################
######################################################################
import numpy as np

from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import fsolve
from scipy.special import gammainc, gamma

from .definitions import GN


################
# NFW profiles #
################
def rho_NFW(*params, mass_concentration=False):
    r"""
    Compute the NFW (Navarro-Frenk-White) density profile for a dark matter halo.

    Parameters
    ----------
    *params : tuple
        Either (rho_s, r_s) where rho_s is the characteristic density and r_s is the scale radius,
        or (M200, c200) if mass_concentration=True, where M200 is the halo mass and c200 is the concentration.
    mass_concentration : bool, optional
        If True, interpret *params as (M200, c200) and convert to (rho_s, r_s) internally (default: False).

    Returns
    -------
    rho_function : callable
        Function returning the NFW density profile rho(r) at radius r.

    Notes
    -----
    The NFW profile is given by:
        rho(r) = rho_s / [ (r/r_s) * (1 + r/r_s)^2 ]
    """
    if mass_concentration == True:
        # Assume inputs are M200,c200
        M200, c200 = params
        rho_s, rs, r200 = mass_concentration_to_NFW_parameters(M200, c200)
    else:
        # Assume inputs are rho_s, rs
        rho_s, rs = params

    def rho_function(r):
        x = r / rs
        return rho_s / x / (1 + x) ** 2

    return rho_function


def M_NFW(*params, mass_concentration=False):
    r"""
    Compute the enclosed mass profile M(r) for an NFW (Navarro-Frenk-White) dark matter halo.

    Parameters
    ----------
    *params : tuple
        Either (rho_s, r_s) where rho_s is the characteristic density and r_s is the scale radius,
        or (M200, c200) if mass_concentration=True, where M200 is the halo mass and c200 is the concentration.
    mass_concentration : bool, optional
        If True, interpret *params as (M200, c200) and convert to (rho_s, r_s) internally (default: False).

    Returns
    -------
    M_function : callable
        Function returning the enclosed mass M(r) at radius r for the NFW profile.

    Notes
    -----
    The enclosed mass profile is given by:
        M(r) = 4π rho_s r_s^3 [ln(1 + r/r_s) - r/r_s / (1 + r/r_s)]
    For small r, a quadratic approximation is used for numerical stability.
    """
    if mass_concentration == True:
        # Assume inputs are M200,c200
        M200, c200 = params
        rho_s, rs, r200 = mass_concentration_to_NFW_parameters(M200, c200)
    else:
        # Assume inputs are rho_s, rs
        rho_s, rs = params

    def M_function(r):
        x = r / rs
        if x > 1e-5:
            return 4 * np.pi * rho_s * rs**3 * (np.log(1 + x) - x / (1 + x))
        else:
            return 2 * np.pi * rho_s * rs**3 * x**2

    return np.vectorize(M_function)


def NFW_profiles(*params, **kwargs):

    return rho_NFW(*params, **kwargs), M_NFW(*params, **kwargs)


def rho_Einasto(*params, mass_concentration=False):
    r"""
    Compute the Einasto density profile for a dark matter halo.

    Parameters
    ----------
    *params : tuple
        Either (rho_minus2, r_minus2, alpha) where rho_minus2 is the density at r_minus2, r_minus2 is the scale radius, and alpha is the shape parameter,
        or (M200, c200, alpha) if mass_concentration=True, where M200 is the halo mass, c200 is the concentration, and alpha is the shape parameter.
    mass_concentration : bool, optional
        If True, interpret *params as (M200, c200, alpha) and convert to (rho_minus2, r_minus2, alpha) internally (default: False).

    Returns
    -------
    rho_function : callable
        Function returning the Einasto density profile rho(r) at radius r.

    Notes
    -----
    The Einasto profile is given by:
        rho(r) = rho_minus2 * exp{ - (2/alpha) * [ (r/r_minus2)^alpha - 1 ] }
    """
    if mass_concentration:
        M200, c200, alpha = params
        rho_minus2, r_minus2, r200 = mass_concentration_to_Einasto_parameters(M200, c200, alpha)
    else:
        rho_minus2, r_minus2, alpha = params
        M200, c200, r200 = Einasto_parameters_to_mass_concentration(rho_minus2, r_minus2, alpha)

    def rho_function(r):
        x = np.asarray(r) / r_minus2
        return rho_minus2 * np.exp(-(2.0 / alpha) * (x**alpha - 1))

    return rho_function


def M_Einasto(*params, mass_concentration=False):
    r"""
    Compute the enclosed mass profile M(r) for an Einasto dark matter halo.

    Parameters
    ----------
    *params : tuple
        Either (rho_minus2, r_minus2, alpha) where rho_minus2 is the density at r_minus2, r_minus2 is the scale radius, and alpha is the shape parameter,
        or (M200, c200, alpha) if mass_concentration=True, where M200 is the halo mass, c200 is the concentration, and alpha is the shape parameter.
    mass_concentration : bool, optional
        If True, interpret *params as (M200, c200, alpha) and convert to (rho_minus2, r_minus2, alpha) internally (default: False).

    Returns
    -------
    M_function : callable
        Function returning the enclosed mass M(r) at radius r for the Einasto profile.

    Notes
    -----
    The enclosed mass profile is given by:
        M(r) = M200 * [ gammainc(3/alpha, (2/alpha)*(r/r_minus2)^alpha) / gammainc(3/alpha, (2/alpha)*c200^alpha) ]
    where gammainc is the lower incomplete gamma function. The normalization ensures M(r200) = M200.
    """
    if mass_concentration:
        M200, c200, alpha = params
        rho_minus2, r_minus2, r200 = mass_concentration_to_Einasto_parameters(M200, c200, alpha)
    else:
        rho_minus2, r_minus2, alpha = params
        M200, c200, r200 = Einasto_parameters_to_mass_concentration(rho_minus2, r_minus2, alpha)

    s = 3.0 / alpha
    Pc = gammainc(s, (2.0 / alpha) * (c200**alpha))

    def M_function(r):
        x = (2.0 / alpha) * (np.asarray(r) / r_minus2) ** alpha
        return M200 * (gammainc(s, x) / Pc)

    return np.vectorize(M_function)


def Einasto_profiles(*params, **kwargs):

    return rho_Einasto(*params, **kwargs), M_Einasto(*params, **kwargs)


# f_baryon=0.156352 (Cautun et al value)

#########################################
# Adiabatically contracted NFW profiles #
#########################################


def AC_profiles(M200, c200, M_baryon, AC_prescription="Cautun", Gnedin_params=(1.6, 0.8)):
    r"""
    Compute adiabatically contracted (AC) NFW profiles for a dark matter halo, including baryonic effects.

    Parameters
    ----------
    M200 : float
        Halo mass within r200 (in solar masses).
    c200 : float
        Concentration parameter of the NFW halo.
    M_baryon : callable
        Function returning the enclosed baryon mass M_b(r) at radius r.
    AC_prescription : {'Cautun', 'Gnedin'}, optional
        Choice of adiabatic contraction model (default: 'Cautun').
        'Cautun' uses the prescription from Cautun et al. (2019),
        'Gnedin' uses the Gnedin et al. (2004) iterative model.
    Gnedin_params : tuple, optional
        Parameters (A0, w) for the Gnedin model (default: (1.6, 0.8)).

    Returns
    -------
    rho_function : callable
        Function returning the contracted DM density profile rho(r).
    M_function : callable
        Function returning the contracted DM enclosed mass profile M(r).

    Notes
    -----
    This function modifies the NFW profile to account for baryonic contraction using either the Cautun or Gnedin prescription.
    The returned functions are suitable for use as outer halo profiles in Jeans modeling.
    """
    rho_s, rs, r200 = mass_concentration_to_NFW_parameters(M200, c200)

    # Use cosmological baryon fraction, not estimated value
    # This value matches one used by Cautun et al.
    f_b = 0.156352

    # This is the baryon fraction estimated directy using input baryon profile
    # f_b = M_baryon(r200)/(M200 + M_baryon(r200))

    # range of r values considered
    rmin = 1e-10 * r200
    rmax = 1e4 * r200
    num_points = 1000

    # DM profile without AC
    M_CDM = M_NFW(rho_s, rs)

    # AC prescription following Cautun et al [1911.04557]
    if AC_prescription == "Cautun":

        # Create table of r values
        r_list = np.logspace(np.log10(rmin), np.log10(rmax), num=num_points)
        M_values = np.array(
            [M_CDM(r) * (0.45 + 0.38 * ((1 - f_b) / f_b * M_baryon(r) / M_CDM(r) + 1.16) ** 0.53) for r in r_list]
        )

        M_values = np.append(0, M_values)
        r_list = np.append(0, r_list)

        M_function = InterpolatedUnivariateSpline(r_list, M_values)

    # AC prescription following Gnedin et al [1108.5736]
    elif AC_prescription == "Gnedin":

        A0, w = Gnedin_params

        # Define orbit-averaged radius bar(r)/r0 = A0*(r/r0)**w
        def bar(r):
            r0 = 0.03 * r200
            return A0 * r0 * (r / r0) ** w

        # Mass within ri -> mass within rf
        # Use iterative approach to obtain original radius ri
        def find_ri(rf, rtol=1e-6, weight=0.5, max_iter=1000):

            # Initialize
            ri = rf
            ri_old, ri_new = 0, 0
            iter_num = 0

            while (abs(1 - ri_old / ri) > rtol) & (iter_num < max_iter):

                ri_old = ri
                ri_new = rf * (1 - f_b) * (1 + M_baryon(bar(rf)) / M_CDM(bar(ri)))
                ri = weight * ri_new + (1 - weight) * ri

                # Check that ri never becomes negative
                if ri < 0:
                    weight = 0.5 * weight
                    ri = ri_old
                else:
                    weight = 0.5

                iter_num += 1

            if iter_num == max_iter:
                raise Exception("AC error: finding r_i never converged within 'rtol'")

            return ri, M_CDM(ri)

        # Create table of (ri, M_CDM(ri)) values
        r_list = np.logspace(np.log10(rmin), np.log10(rmax), num=num_points)
        table = np.array([find_ri(rf) for rf in r_list])

        M_values = np.append(0, table[:, 1])
        r_list = np.append(0, r_list)

        M_function = InterpolatedUnivariateSpline(r_list, M_values)

    else:
        raise Exception("AC_prescription=" + AC_prescription + " not found.")

    # Compute density rho(r) numerically from M(r):
    # Extrapolate beyond rmin and rmax using power law
    dlogM_dlogr = np.gradient(np.log(M_values[1:]), np.log(r_list[1:]))
    log_rho = np.log(M_values[1:] / (4 * np.pi * r_list[1:] ** 3) * dlogM_dlogr)
    log_rho_function = InterpolatedUnivariateSpline(np.log(r_list[1:]), log_rho, k=1, ext=0)

    def rho_function(r):
        return np.exp(log_rho_function(np.log(r)))

    return rho_function, M_function


###################
# Other functions #
###################
def mass_concentration_to_NFW_parameters(Mvir, c, h=0.7, del_c=200, Omega_m=0.3, Omega_Lambda=0.7, z=0):
    r"""
    Convert halo mass and concentration to NFW profile parameters.

    Parameters
    ----------
    Mvir : float
        Virial mass of the halo (in solar masses).
    c : float
        Concentration parameter of the halo.
    h : float, optional
        Dimensionless Hubble parameter (default: 0.7).
    del_c : float, optional
        Overdensity parameter (default: 200).
    Omega_m : float, optional
        Matter density parameter (default: 0.3).
    Omega_Lambda : float, optional
        Cosmological constant density parameter (default: 0.7).
    z : float, optional
        Redshift (default: 0).

    Returns
    -------
    rhos : float
        Characteristic density of the NFW profile.
    rs : float
        Scale radius of the NFW profile.
    rvir : float
        Virial radius of the halo.

    Notes
    -----
    This function uses the standard NFW relations to convert mass and concentration to profile parameters.
    """
    # Constants
    H0 = h * 100 * 1e-3  # km/kpc/s
    rho_crit = (3 * H0**2) / (8 * np.pi * GN) * (Omega_m * (1 + z) ** 3 + Omega_Lambda)

    # Overdensity
    over_density = (del_c / 3) * ((c**3) / (np.log(1 + c) - (c / (1 + c))))

    # Evaluate parameters
    rvir = np.cbrt(3 * Mvir / (del_c * 4 * np.pi * rho_crit))
    rs = rvir / c
    rhos = rho_crit * over_density

    return rhos, rs, rvir


def NFW_parameters_to_mass_concentration(rhos, rs, h=0.7, del_c=200, Omega_m=0.3, Omega_Lambda=0.7, z=0):
    r"""
    Convert NFW profile parameters to halo mass and concentration.

    Parameters
    ----------
    rhos : float
        Characteristic density of the NFW profile.
    rs : float
        Scale radius of the NFW profile.
    h : float, optional
        Dimensionless Hubble parameter (default: 0.7).
    del_c : float, optional
        Overdensity parameter (default: 200).
    Omega_m : float, optional
        Matter density parameter (default: 0.3).
    Omega_Lambda : float, optional
        Cosmological constant density parameter (default: 0.7).
    z : float, optional
        Redshift (default: 0).

    Returns
    -------
    Mvir : float
        Virial mass of the halo (in solar masses).
    c : float
        Concentration parameter of the halo.
    rvir : float
        Virial radius of the halo.

    Notes
    -----
    This function numerically inverts the NFW relations to recover mass and concentration from profile parameters.
    """
    # Constants
    H0 = h * 100 * 1e-3  # km/kpc/s
    rho_crit = (3 * H0**2) / (8 * np.pi * GN) * (Omega_m * (1 + z) ** 3 + Omega_Lambda)

    # Solve for c
    g = lambda c: c**3 / (np.log(1 + c) - c / (1 + c))
    eqn = lambda c: 3 * rhos / (rho_crit * del_c) - g(c)
    c = fsolve(eqn, 10)[0]

    # Solve for rvir
    rvir = rs * c

    # Sove for Mvir
    Mvir = del_c * rho_crit * 4 * np.pi / 3 * rvir**3

    return Mvir, c, rvir


def _m_c_alpha(c, alpha):
    a = 3.0 / alpha
    x = (2.0 / alpha) * (c**alpha)

    # lower incomplete gamma function
    lower_inc = gamma(a) * gammainc(a, x)
    prefac = np.exp(2.0 / alpha) / alpha * (2.0 / alpha) ** (-3.0 / alpha)
    return prefac * lower_inc


def mass_concentration_to_Einasto_parameters(Mvir, c, alpha, h=0.7, del_c=200, Omega_m=0.3, Omega_Lambda=0.7, z=0):
    r"""
    Convert halo mass, concentration, and shape parameter to Einasto profile parameters.

    Parameters
    ----------
    Mvir : float
        Virial mass of the halo (in solar masses).
    c : float
        Concentration parameter of the halo.
    alpha : float
        Einasto shape parameter.
    h : float, optional
        Dimensionless Hubble parameter (default: 0.7).
    del_c : float, optional
        Overdensity parameter (default: 200).
    Omega_m : float, optional
        Matter density parameter (default: 0.3).
    Omega_Lambda : float, optional
        Cosmological constant density parameter (default: 0.7).
    z : float, optional
        Redshift (default: 0).

    Returns
    -------
    rho_minus2 : float
        Density at the scale radius r_minus2.
    r_minus2 : float
        Scale radius where the logarithmic slope is -2.
    rvir : float
        Virial radius of the halo.

    Notes
    -----
    This function uses the standard Einasto relations to convert mass, concentration, and shape to profile parameters.
    """
    # Constants
    H0 = h * 100 * 1e-3  # km/kpc/s
    rho_crit = (3 * H0**2) / (8 * np.pi * GN) * (Omega_m * (1 + z) ** 3 + Omega_Lambda)

    # Virial radius from spherical overdensity
    rvir = (3 * Mvir / (4 * np.pi * del_c * rho_crit)) ** (1 / 3)
    r_minus2 = rvir / c

    # Mass factor at c
    mca = _m_c_alpha(c, alpha)

    # Solve for rho_-2 from Mvir = 4π rho_-2 r_-2^3 m(c,alpha)
    rho_minus2 = Mvir / (4 * np.pi * r_minus2**3 * mca)
    return rho_minus2, r_minus2, rvir


def Einasto_parameters_to_mass_concentration(
    rho_minus2,
    r_minus2,
    alpha,
    h=0.7,
    del_c=200,
    Omega_m=0.3,
    Omega_Lambda=0.7,
    z=0,
    c_init=10.0,
):
    r"""
    Convert Einasto profile parameters to halo mass, concentration, and shape parameter.

    Parameters
    ----------
    rho_minus2 : float
        Density at the scale radius r_minus2.
    r_minus2 : float
        Scale radius where the logarithmic slope is -2.
    alpha : float
        Einasto shape parameter.
    h : float, optional
        Dimensionless Hubble parameter (default: 0.7).
    del_c : float, optional
        Overdensity parameter (default: 200).
    Omega_m : float, optional
        Matter density parameter (default: 0.3).
    Omega_Lambda : float, optional
        Cosmological constant density parameter (default: 0.7).
    z : float, optional
        Redshift (default: 0).
    c_init : float, optional
        Initial guess for concentration (default: 10.0).

    Returns
    -------
    Mvir : float
        Virial mass of the halo (in solar masses).
    c : float
        Concentration parameter of the halo.
    rvir : float
        Virial radius of the halo.

    Notes
    -----
    This function numerically inverts the Einasto relations to recover mass and concentration from profile parameters.
    """
    # Constants
    H0 = h * 100 * 1e-3  # km/kpc/s
    rho_crit = (3 * H0**2) / (8 * np.pi * GN) * (Omega_m * (1 + z) ** 3 + Omega_Lambda)

    rhs_coeff = (del_c * rho_crit) / 3.0

    def eqn(c):
        return rho_minus2 * _m_c_alpha(c, alpha) - rhs_coeff * c**3

    # Solve for concentration
    c = float(fsolve(eqn, c_init))
    rvir = c * r_minus2

    # Mvir from either definition
    Mvir = (4 * np.pi / 3) * del_c * rho_crit * rvir**3
    # (equivalently: Mvir = 4π ρ_-2 r_-2^3 m(c,α))
    return Mvir, c, rvir
