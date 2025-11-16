"""
classes.py
----------
Purpose:   Core class definitions for isothermal, CDM profiles, and the combined profile object.
           Includes methods for scientific calculations (density, potential, rotation curve, shape).
Authors:   Sean Tulin, Adam Smith-Orlik
Contact:   stulin@yorku.ca, asorlik@yorku.ca
Status:    Active development
Last Edit: 2025-09-16

This file contains the main class hierarchy for the nonspherical SIDM Jeans modeling package, including serialization logic and scientific methods for profile objects.
"""

######################################################################
############################## IMPORTS ###############################
######################################################################
import numpy as np
from inspect import signature
import inspect
import os
import dill

from scipy.integrate import solve_ivp, dblquad
from scipy.interpolate import (
    interp1d,
    InterpolatedUnivariateSpline,
    RectBivariateSpline,
)
from scipy.optimize import brentq, fsolve, curve_fit


from .definitions import GN, Z, integrate, no_baryons, central_derivative
from .cdm import (
    AC_profiles,
    Einasto_profiles,
    NFW_profiles,
    mass_concentration_to_NFW_parameters,
    NFW_parameters_to_mass_concentration,
    mass_concentration_to_Einasto_parameters,
    Einasto_parameters_to_mass_concentration,
)
from . import rotcurve as rotcurve
from . import potential as potential

from .tools import compute_Mb, compute_r_sph_grid


######################################################################
######################### CLASS DEFINITIONS ##########################
######################################################################
class profile:

    def __init__(self, inner=None, outer=None, q=None):

        self.inner = inner
        self.outer = outer

        if not (outer):
            raise Exception("No outer halo set.")

        if inner:
            self.r1 = inner.r_list[-1]  # SIDM case
        else:
            self.r1 = 0  # CDM case

        # External q to squash profile
        # q(r_sph) is a function of r_sph
        if not (q):
            # Default value: no squashing
            self.q = lambda r_sph: 1
            self.squash_flag = False
            self.r_sph_interp = lambda r, th: r

        elif callable(q):
            # q is a function
            self.q = q
            self.squash_flag = True
            self.r_sph_interp = compute_r_sph_grid(self.outer, self.q)

        else:
            # q is a fixed number
            self.q = lambda r_sph: q
            self.squash_flag = True
            self.r_sph_interp = compute_r_sph_grid(self.outer, self.q)

        # Set flag if profile is spherically symmetric
        if inner:
            self.sph_sym_flag = inner.sph_sym_flag and outer.sph_sym_flag and not self.squash_flag
        else:
            self.sph_sym_flag = outer.sph_sym_flag and not self.squash_flag

    # Spheroidal radius function
    def r_sph(self, r, th, max_iter=200, interp=True, **kwargs):

        if interp and (r < self.outer.r200):
            return self.r_sph_interp(r, th)

        else:

            R = r * np.sin(th)
            z = r * np.cos(th)

            rsph_old = r
            for i in range(max_iter):

                rsph_new = np.sqrt(R**2 * self.q(rsph_old) ** (2 / 3) + z**2 * self.q(rsph_old) ** (-4 / 3))

                if np.allclose(rsph_new, rsph_old, **kwargs):
                    break

                rsph_old = rsph_new

            else:
                raise Exception("rsph did not converge within tolerance with max_iter=%d iterations" % max_iter)

            return rsph_new

    # Spherically averaged density profile
    def rho_sph_avg(self, r, interp=True, zsym=True):

        # Check if r is a single number
        if np.ndim(r) == 0:

            # Density function
            def rho(r):
                if r < self.r1:
                    return self.inner.rho_sph_avg(r)
                else:
                    return self.outer.rho_sph_avg(r)

            # Take spherical average of squashed profile
            if self.squash_flag:

                integrand = lambda th: rho(self.r_sph(r, th, interp=interp)) * np.sin(th)
                # return 0.5 * integrate(integrand,0,np.pi)

                # If symmetric about xy plane, only integrate half of integration range and multiply by 2
                if zsym:
                    return integrate(integrand, 0, np.pi / 2)
                else:
                    return 0.5 * integrate(integrand, 0, np.pi)

            # Usual density function of spherical Jeans model
            else:
                return rho(r)

        # Else r is a list or array
        else:
            return np.array([self.rho_sph_avg(ri, interp=interp, zsym=zsym) for ri in r])

    # Helper function for calculating squashed Jeans profile
    def rho_spherical(self, r):

        if np.ndim(r) == 0:
            return self.inner.rho_sph_avg(r) if r < self.r1 else self.outer.rho_sph_avg(r)

        else:
            return np.array([self.rho_spherical(ri) for ri in r])

    # Density profile in spherical coordinates
    # theta coordinates are optional. If omitted, returns spherically averaged density
    def rho_sph(self, r, th=None, interp=True, zsym=True):

        # No theta specified, return spherically average density
        if th == None:
            return self.rho_sph_avg(r, interp=interp, zsym=zsym)

        # Case where r and theta are single numbers
        elif (np.ndim(r) == 0) and (np.ndim(th) == 0):

            # Squashed profile
            if self.squash_flag:
                return self.rho_spherical(self.r_sph(r, th, interp=interp))

            # Inner profile without squashing
            # Usual prescription
            else:
                if r < self.r1:
                    return self.inner.rho_sph(r, th)
                else:
                    return self.outer.rho_sph(r, th)

        # Case where r is a list and theta is a single number
        elif (np.ndim(r) > 0) and (np.ndim(th) == 0):
            return np.array([self.rho_sph(ri, th, interp=interp) for ri in r])

        # Case where r is a single number and theta is a list
        elif (np.ndim(r) == 0) and (np.ndim(th) > 0):
            return np.array([self.rho_sph(r, th_i, interp=interp) for th_i in th])

        # Case where both r and theta are lists
        else:
            # Check have same lengths
            if len(r) != len(th):
                raise Exception("Not sure how to interpret r with len=%d and theta with len=%d" % (len(r), len(th)))

            else:
                return np.array([self.rho_sph(ri, th_i, interp=interp) for ri, th_i in zip(r, th)])

        # End of rho_sph (DM density)

    def rho_cyl(self, R, z):

        r = np.sqrt(R**2 + z**2)
        th = np.arctan2(R, z)

        return self.rho_sph(r, th)

    def rho_cyl_array(self, R_array, z_array):

        len_R = len(R_array)
        len_z = len(z_array)

        # array of r values
        r_array = np.sqrt(np.add.outer(R_array**2, z_array**2).flatten())

        # array of theta values
        th_array = np.arctan2.outer(R_array, z_array).flatten()

        return np.array([self.rho_sph(r, th) for r, th in zip(r_array, th_array)]).reshape((len_R, len_z))

    # Angular integral rho_LM(r) = int dOmega Z(L,M,th,phi) rho(r,th,phi)
    # Only M = 0 case supported
    def rho_LM(self, L, M, r):

        # Case where r is a number
        if np.ndim(r) == 0:

            # Automatically return result for spherically symmetric profile
            if self.sph_sym_flag:
                return self.rho_sph_avg(r) / Z(0, 0, 0, 0) if L == 0 else 0

            # M != 0 not supported
            elif M != 0:
                return 0.0

            # Compute integral
            else:
                integrand = lambda th: 2 * np.pi * np.sin(th) * self.rho_sph(r, th) * Z(L, M, th, 0)
                return integrate(integrand, 0, np.pi)

        # If r is an array, loop over those values recursively
        else:
            return np.array([self.rho_LM(L, M, ri) for ri in r])

    # Makes interpolation function for rho_LM
    # Useful for faster calculations of rotation curve and potential
    def rho_LM_interp(self, L, M, k=3, num=50):

        # Define radii
        r200 = self.outer.r200
        rmin, rmax = 1e-6 * r200, 10 * r200
        r1 = self.r1

        # L = 0 case
        # Code is simpler since always positive and continuous
        if L == 0:

            # Interpolation function
            r_list = np.geomspace(rmin, rmax, num=num)
            rho_00_list = self.rho_sph_avg(r_list) / Z(0, 0, 0, 0)
            x = np.log(r_list)
            y = np.log(rho_00_list)
            interp = InterpolatedUnivariateSpline(x, y, k=k)

            # Power law extrapolation
            # Get inner power law
            inner_slope = (y[1] - y[0]) / (x[1] - x[0])
            inner = lambda x_: y[0] + inner_slope * (x_ - x[0])

            # Get outer power law
            outer_slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
            outer = lambda x_: y[-1] + outer_slope * (x_ - x[-1])

            def output_func(r):
                x = np.log(r)

                if r < rmin:
                    return np.exp(inner(x))
                elif r > rmax:
                    return np.exp(outer(x))
                else:
                    return np.exp(interp(x))

            return np.vectorize(output_func)

        # Case where profile is spherically symmetric (but L > 0)
        # Function that returns zero
        elif self.sph_sym_flag:

            def output_func(r):
                if np.ndim(r) == 0:
                    return 0.0
                else:
                    return np.zeros_like(r, "f")

            return output_func

        # Remaining nonspherical cases
        r_list = np.geomspace(rmin, rmax, num=num)
        rho_LM_list = self.rho_LM(L, M, r_list)

        # Case where moments are all zero
        # Just test first entry
        if rho_LM_list[0] == 0:

            def output_func(r):
                if np.ndim(r) == 0:
                    return 0
                else:
                    return np.zeros_like(r)

            return output_func

        # Moments are nonzero
        else:

            # Inner halo interp
            if r1 > rmin:

                # Interpolate abs and sign
                sel = r_list < r1
                r1eps = r1 * (1 - 1e-6)
                rho_LM_1 = self.rho_LM(L, M, r1eps)
                x = np.log(np.append(r_list[sel], r1eps))
                y = np.log(np.abs(np.append(rho_LM_list[sel], rho_LM_1)))
                sign = np.sign(np.append(rho_LM_list[sel], rho_LM_1))

                inner_interp = InterpolatedUnivariateSpline(x, y, k=k)
                inner_sign_interp = InterpolatedUnivariateSpline(x, sign, k=1)

                # Full interpolation function for range [rmin,r1]
                inner = lambda r: np.exp(inner_interp(np.log(r))) * inner_sign_interp(np.log(r))

            # Inner power law below rmin
            inner_slope = np.log(rho_LM_list[1] / rho_LM_list[0]) / np.log(r_list[1] / r_list[0])
            inner_extrap = lambda r: rho_LM_list[0] * (r / rmin) ** inner_slope

            # Outer halo interp
            sel = r_list >= r1

            # Outer halo is spherically symmetric (and L > 0)
            if self.outer.sph_sym_flag and not (self.squash_flag):
                outer = lambda r: 0.0
                outer_extrap = lambda r: 0.0

            # Outer halo is not spherically symmetric
            else:

                # Interpolate abs and sign
                x = np.log(r_list[sel])
                y = np.log(np.abs(rho_LM_list[sel]))
                sign = np.sign(rho_LM_list[sel])

                if r1 > 0:
                    x = np.append(np.log(r1), x)
                    rho_LM_1 = self.rho_LM(L, M, r1)
                    y = np.append(np.log(np.abs(rho_LM_1)), y)
                    sign = np.append(np.sign(rho_LM_1), sign)

                outer_interp = InterpolatedUnivariateSpline(x, y, k=k)
                outer_sign_interp = InterpolatedUnivariateSpline(x, sign, k=1)

                # Full interpolation function for range [r1,rmax]
                outer = lambda r: np.exp(outer_interp(np.log(r))) * outer_sign_interp(np.log(r))

                # Outer power law beyond rmax
                outer_slope = np.log(rho_LM_list[-1] / rho_LM_list[-2]) / np.log(r_list[-1] / r_list[-2])
                outer_extrap = lambda r: rho_LM_list[-1] * (r / rmax) ** outer_slope

            def output_func(r):
                if r < rmin:
                    return inner_extrap(r)
                elif r > rmax:
                    return outer_extrap(r)
                elif r < r1:
                    return inner(r)
                else:
                    return outer(r)

            return np.vectorize(output_func)

    def cross_section(self, **kwargs):

        # SIDM case
        if self.inner:
            return self.inner.cross_section(**kwargs)

        # CDM case
        else:
            return 0

    def M_encl(self, r_encl, num=300):

        # Case where r_encl is a number
        if np.ndim(r_encl) == 0:
            return float(self.M_encl(np.array([r_encl]))[0])

        else:

            M_out = np.zeros_like(r_encl, "f")

            # Use M_encl from inner or outer profiles if no squashing
            if not (self.squash_flag):

                sel = r_encl >= self.r1
                M_out[sel] = self.outer.M_encl(r_encl[sel])

                if self.inner:
                    sel = r_encl < self.r1
                    M_out[sel] = self.inner.M_encl(r_encl[sel])

                return M_out

            # Calculate M_encl directly from squashed profiles using interpolation of the integrand
            else:

                rmax = max(r_encl)
                rmin = min(1e-4, min(r_encl))

                # Case where only r_encl=0 input
                if rmax == 0:
                    return np.zeros_like(r_encl, "f")

                def integrand(r):
                    return 4 * np.pi * r**2 * self.rho_sph(r) if r > 0 else 0

                # Enclosed mass at rmin
                Mmin = integrate(integrand, 0, rmin)

                # Next use interpolation
                r_list = np.geomspace(rmin, rmax, num=num)
                integrand_list = np.array([integrand(ri) for ri in r_list])

                integrand_interp = InterpolatedUnivariateSpline(r_list, integrand_list)
                M_out = np.array([integrand_interp.integral(rmin, ri) + Mmin for ri in r_encl])

                return M_out

        # End M_encl

    # Rotation curves
    def Vsq_baryon(self, r):

        # Phi_b from outer profile
        Phi_b = self.outer.Phi_b

        return rotcurve.Vsq_baryon(Phi_b, r)

    def V_baryon(self, r):
        return np.sqrt(self.Vsq_baryon(r))

    def Vsq_dm(self, r, Lmax=10, **kwargs):

        theta = np.pi / 2  # plane of disk

        Vsq_tot = np.zeros_like(r, "f")  # initialize

        for L in range(0, Lmax + 1, 2):

            # Make interpolation function for rho_LM
            rho_LM = self.rho_LM_interp(L, 0, **kwargs)

            # Calculate LM mode contribution
            Vsq_old = np.array(Vsq_tot)
            Vsq_tot += rotcurve.Vsq_LM(rho_LM, r, L) * Z(L, 0, theta, 0)

            if np.allclose(Vsq_tot, Vsq_old, rtol=1e-3, atol=1e-3):
                break

        return Vsq_tot

    def Vsq_dm_sph_sym(self, r):

        r_arr = np.array(r)
        Vsq = GN * self.M_encl(r_arr) / r_arr

        if np.ndim(r_arr) == 0:
            return float(Vsq)
        else:
            return Vsq

    def Vdm(self, r, Lmax=10):
        return np.sqrt(self.Vsq_dm(r, Lmax=Lmax))

    def Vsq(self, r, Lmax=10):
        return self.Vsq_baryon(r) + self.Vsq_dm(r, Lmax=Lmax)

    def V(self, r, Lmax=10):
        return np.sqrt(self.Vsq(r, Lmax=Lmax))

    def Vsq_LM(self, r, L, M=0, **kwargs):
        rho_LM = self.rho_LM_interp(L, 0, **kwargs)
        return rotcurve.Vsq_LM(rho_LM, r, L, M=M)

    # Potentials

    def Phi(self, r, theta=None, Lmax=10):
        if (self.sph_sym_flag) or (theta == None):
            return self.Phi_LM(r, 0) * Z(0, 0, 0, 0)

        else:
            return np.sum([self.Phi_LM(r, L) * Z(L, 0, theta, 0) for L in range(Lmax + 1)], axis=0)

    def Phi_LM(self, r, L, M=0, **kwargs):
        rho_LM = self.rho_LM_interp(L, 0, **kwargs)
        return potential.Phi_LM(rho_LM, r, L, M=M)

    def Phi_dm(self, r, th, Lmax=10, **kwargs):

        # Handle different cases where r and/or theta are single numbers
        if (np.ndim(r) == 0) and (np.ndim(r) == 0):
            return self.Phi_dm(self, [r], [th], Lmax=Lmax, **kwargs)
        elif np.ndim(r) == 0:
            return self.Phi_dm(self, [r], th, Lmax=Lmax, **kwargs)
        elif np.ndim(th) == 0:
            return self.Phi_dm(self, r, [th], Lmax=Lmax, **kwargs)
        else:
            pass

        Phi_tot = np.zeros((len(r), len(th)), "f")  # initialize

        for L in range(0, Lmax + 1, 2):

            # Make interpolation function for rho_LM
            rho_LM = self.rho_LM_interp(L, 0, **kwargs)

            # Calculate LM mode contribution
            Phi_old = np.array(Phi_tot)
            Phi_tot += np.multiply.outer(potential.Phi_LM(rho_LM, r, L), Z(L, 0, theta, 0))

            if np.allclose(Phi_tot, Phi_old, rtol=1e-3, atol=1e-3):
                break

        return Phi_tot

    # Shape variables

    def q_eff(self, r_sph, qmax=1000, qmin=0.001):

        # r_sph = np.sqrt((R * self.q**(1/3.))**2 + (z * self.q**(-2/3.))**2)

        rho = self.rho_cyl

        def find_q(r):
            # f = log(rho(R,0) / rho(0,z))
            f = lambda q: np.log(rho(r * q ** (-1 / 3.0), 0) / rho(0, r * q ** (2 / 3.0)))
            try:
                return brentq(f, qmin, qmax)
            except:
                return 0

        if np.ndim(r_sph) == 0:
            return find_q(r_sph)

        elif np.ndim(r_sph) == 1:
            return np.array([find_q(r) for r in r_sph])

        else:
            raise Exception("r_sph must be 0D or 1D.")

    def q_shell(
        self,
        r_eff,
        qmax=2,
        qmin=0.5,
        method="brentq",
        maxiter=100,
        rtol=1e-3,
        atol=1e-3,
        verbose=False,
    ):

        if np.ndim(r_eff) == 0:

            rho = self.rho_sph
            r = lambda th, q: r_eff / np.sqrt(
                # (np.sin(th) * q ** (2 / 3)) ** 2 + (np.cos(th) * q ** (-1 / 3)) ** 2
                (np.sin(th) * q ** (1 / 3)) ** 2
                + (np.cos(th) * q ** (-2 / 3)) ** 2
            )

            if method == "iterate":

                q_old = 1
                num_iter = 0

                while num_iter < maxiter:

                    integrand = lambda th: 2 * np.sin(th) * np.cos(th) ** 2 * rho(r(th, q_old), th) * r(th, q_old) ** 5
                    I_zz = integrate(integrand, 0, np.pi)

                    integrand = lambda th: np.sin(th) * np.sin(th) ** 2 * rho(r(th, q_old), th) * r(th, q_old) ** 5
                    I_RR = integrate(integrand, 0, np.pi)

                    q_new = np.sqrt(I_zz / I_RR)

                    if np.isclose(q_old, q_new, rtol=rtol, atol=atol):
                        break
                    else:
                        q_old = q_new
                        num_iter += 1
                else:
                    print("'iterate' method did not converge in %d iterations." % maxiter)

                return q_new

            elif method == "brentq":

                def f(q):

                    integrand = lambda th: 2 * np.sin(th) * np.cos(th) ** 2 * rho(r(th, q), th) * r(th, q) ** 5
                    I_zz = integrate(integrand, 0, np.pi)

                    integrand = lambda th: np.sin(th) * np.sin(th) ** 2 * rho(r(th, q), th) * r(th, q) ** 5
                    I_RR = integrate(integrand, 0, np.pi)

                    return q**2 - I_zz / I_RR

                try:
                    output = brentq(f, qmin, qmax, xtol=atol, rtol=rtol)
                except:
                    output = self.q_shell(
                        r_eff,
                        qmax=2 * qmax,
                        qmin=0.5 * qmin,
                        atol=atol,
                        rtol=rtol,
                        method=method,
                    )

                return output

            else:
                raise Exception("Unknown method")

        elif np.ndim(r_eff) == 1:
            return np.array(
                [
                    self.q_shell(
                        ri,
                        method=method,
                        qmax=qmax,
                        qmin=qmin,
                        maxiter=maxiter,
                        atol=atol,
                        rtol=rtol,
                    )
                    for ri in r_eff
                ]
            )

        else:
            raise Exception("r must be 0D or 1D.")

    def q_encl(
        self,
        r_eff,
        qmax=2,
        qmin=0.5,
        q_init=1,
        method="brentq",
        maxiter=100,
        rtol=1e-3,
        atol=1e-3,
        epsabs=1e-5,
        epsrel=1e-5,
        verbose=False,
    ):

        if np.ndim(r_eff) == 0:
            # print('r_eff: ', r_eff)

            rho = self.rho_dm_sph
            r_spheroid = lambda th, q: r_eff / np.sqrt(
                (np.sin(th) * q ** (1 / 3)) ** 2 + (np.cos(th) * q ** (-2 / 3)) ** 2
            )
            r_min = lambda th: 0

            if method == "iterate":

                q_old = q_init
                num_iter = 0

                while num_iter < maxiter:

                    if verbose:
                        print(num_iter, q_old)

                    r_max = lambda th: r_spheroid(th, q_old)

                    integrand = lambda r, th: 2 * np.sin(th) * np.cos(th) ** 2 * rho(r, th) * r**4
                    I_zz = dblquad(integrand, 0, np.pi, r_min, r_max, epsabs=epsabs, epsrel=epsrel)[0]

                    integrand = lambda r, th: np.sin(th) * np.sin(th) ** 2 * rho(r, th) * r**4
                    I_RR = dblquad(integrand, 0, np.pi, r_min, r_max, epsabs=epsabs, epsrel=epsrel)[0]

                    q_new = np.sqrt(I_zz / I_RR)

                    if np.isclose(q_old, q_new, rtol=rtol, atol=atol):
                        break
                    else:
                        q_old = q_new
                        num_iter += 1
                else:
                    print("'iterate' method did not converge in %d iterations." % maxiter)

                return q_new

            elif method == "brentq":

                def f(q):

                    r_max = lambda th: r_spheroid(th, q)

                    integrand = lambda r, th: 2 * np.sin(th) * np.cos(th) ** 2 * rho(r, th) * r**4
                    I_zz = dblquad(integrand, 0, np.pi, r_min, r_max, epsabs=epsabs, epsrel=epsrel)[0]

                    integrand = lambda r, th: np.sin(th) * np.sin(th) ** 2 * rho(r, th) * r**4
                    I_RR = dblquad(integrand, 0, np.pi, r_min, r_max, epsabs=epsabs, epsrel=epsrel)[0]

                    return q**2 - I_zz / I_RR

                try:
                    output = brentq(f, qmin, qmax, xtol=atol, rtol=rtol)
                except:
                    output = self.q_encl(
                        r_eff,
                        qmax=2 * qmax,
                        qmin=0.5 * qmin,
                        atol=atol,
                        rtol=rtol,
                        method=method,
                    )

                return output

            else:
                raise Exception("Unknown method")

        elif np.ndim(r_eff) == 1:
            # print([ri for ri in r_eff])
            return np.array(
                [
                    self.q_encl(
                        ri,
                        method=method,
                        qmax=qmax,
                        qmin=qmin,
                        maxiter=maxiter,
                        atol=atol,
                        rtol=rtol,
                    )
                    for ri in r_eff
                ]
            )

        else:
            raise Exception("r must be 0D or 1D.")

    def V_and_q_from_force(self, r, Lmax=10, **kwargs):
        if np.ndim(r) == 0:
            r = np.array([r])
        thetas = np.linspace(0, np.pi / 2, 7)
        theta = np.pi / 2  # plane of disk
        rgrid, tgrid = np.meshgrid(r, thetas)

        Fr = np.zeros(rgrid.shape)
        Ftheta = np.zeros(rgrid.shape)

        Vsq_tot = np.zeros_like(r, "f")  # initialize

        for L in range(0, Lmax + 1, 2):
            # print('L = ', L)
            Vc_sq_L = self.Vsq_LM(r, L)
            Fr_L = Vc_sq_L / r

            Z_theta = lambda x: Z(L, 0, x, 0)
            Z_L = Z_theta(thetas)

            Fr = Fr - np.array([Fr_L * z_l for z_l in Z_L])

            if L > 0:
                Phi_L = self.Phi_LM(r, L) / r
                dZdtheta_L = central_derivative(Z_theta, thetas, dx=1e-2 * np.pi)
                Ftheta = Ftheta - np.array([Phi_L * dz_l for dz_l in dZdtheta_L])

            # rho_LM = self.rho_LM_interp(L,0,**kwargs)
            # Calculate LM mode contribution
            Vsq_old = np.array(Vsq_tot)
            Vsq_tot += Vc_sq_L * Z(L, 0, theta, 0)

            if np.allclose(Vsq_tot, Vsq_old, rtol=1e-3, atol=1e-3):
                # print('Converged at L = ', L)
                break

        # pure monopole, spherical case
        if Lmax == 0:
            V_sq_b = self.Vsq_baryon(r)
            return np.sqrt(Vsq_tot + V_sq_b), np.ones_like(r)

        def obj_curve(r):
            def fix_r(theta, q):
                rho = r * np.sin(theta)
                z = r * np.cos(theta)
                m = np.sqrt(rho**2 + z**2 / q**2)
                return r**2 / 2 / (m**2) * (1 - 1 / q**2) * np.sin(2 * theta)

            return fix_r

        qr = []
        perr = []
        for i in range(len(r)):
            obj = obj_curve(r[i])
            x = thetas
            y = Ftheta[:, i] / Fr[:, i]
            popt, pcov = curve_fit(obj, x, y)
            perr.append(np.sqrt(np.diag(pcov)))
            qr.append(popt[0])

        qr = np.array(qr)
        V_sq_b = self.Vsq_baryon(r)

        return np.sqrt(Vsq_tot + V_sq_b), qr

    def save(self, filename):
        with open(filename, "wb") as f:
            dill.dump(self, f, protocol=dill.HIGHEST_PROTOCOL)


############################
# isothermal jeans profile #
############################


class isothermal_profile:

    def __init__(self, y, params, r_list, Phi_b=None, L_list=[0], M_list=[0]):

        # Some preliminary definitions
        self.num_func = 2 * len(L_list)
        self.num_r_points = len(r_list)

        self.y = y
        self.r_list = r_list

        self.params = params
        self.r0 = params[0] ** 0.5
        self.sigma0 = params[1] ** 0.5
        self.rho0 = self.sigma0**2 / (4 * np.pi * GN * self.r0**2)

        self.vrel = self.sigma0 * 4 / np.sqrt(np.pi)
        self.r1 = r_list[-1]

        self.L_list = L_list
        # Don't need to explicitly input list M=[0,0,0...]
        if len(M_list) != len(L_list):
            self.M_list = [0 for i in range(len(L_list))]
        else:
            self.M_list = M_list

        if Phi_b:
            self.Phi_b = Phi_b
            self._Phi_b_user = Phi_b  # Store original user-supplied function for serialization
        else:
            self.Phi_b = no_baryons
            self._Phi_b_user = None

        self.num_Phi_b_variables = len(signature(self.Phi_b).parameters)
        if self.num_Phi_b_variables > 2:
            raise Exception("Phi_b has %d > 2 variables." % self.num_Phi_b_variables)

        # Can keep track of commonly-used angular integrals to evaluate them only once
        # __init__ does not set them, but update(angular_moments_set=True) sets them
        # Relaxation code will use these automatically if they are defined
        self.angular_moments_set = False
        self.angular_moments_list = np.zeros((self.num_r_points + 1, len(L_list)))
        r_points = np.concatenate(([r_list[0]], 0.5 * (r_list[:-1] + r_list[1:]), [r_list[-1]]))
        self.angular_moments_interp = interp1d(r_points, self.angular_moments_list, axis=0)

        # Flag for spherical symmetry if Phi_b only depends on r and only L=0 mode
        self.sph_sym_flag = (self.num_Phi_b_variables == 1) & (self.L_list == [0])

    # end __init__

    # General helper function for getting moments phi_LM or mu_LM
    # If no r value, returns list at grid points
    # If r is input, returns interpolation between grid or power-law extrapolation
    # below innermost nonzero grid point
    def get_moment(self, moment_type, L, M, *r, k=3):

        # Find index for desired L,M value
        LM_index = np.where((np.array(self.L_list) == L) & (np.array(self.M_list) == M))[0]

        # Should find exactly one match for L,M values
        if len(LM_index) == 1:
            i = LM_index[0]
        else:
            raise Exception("Error: %d matches found for L=%d and M=%d" % (len(LM_index), L, M))

        if moment_type == "phi":
            moment = self.y[i :: self.num_func]
        elif moment_type == "mu":
            moment = self.y[i + len(self.L_list) :: self.num_func]
        else:
            raise Exception("moment=%s must be 'phi' or 'mu'")

        # No r value specified, return entire list at grid points
        if not (r):
            return moment

        # Return interpolated result at r
        else:

            # Check within grid
            if np.amax(r) > self.r1:
                raise Exception("All r values must be < r1. max(r)=%f outside r1." % np.amax(r))

            # Get inner slope
            if moment[1] == moment[2]:
                alpha = 0
            else:
                alpha = np.log(moment[2] / moment[1]) / np.log(self.r_list[2] / self.r_list[1])
            inner = lambda r: moment[1] * (r / self.r_list[1]) ** alpha

            # Make interpolation function
            interp = InterpolatedUnivariateSpline(self.r_list, moment, k=k)

            # Need extra [0] to get rid of outermost bracket
            output = np.where(r > self.r_list[1], interp(r), inner(r))[0]

            if np.ndim(r) == 0:
                return float(output)
            else:
                return output

    def phi(self, L, M, *r):
        return self.get_moment("phi", L, M, *r)

    def mu(self, L, M, *r):
        return self.get_moment("mu", L, M, *r)

    # Mencl at grid points
    def M_encl_list(self):

        mu_00 = self.mu(0, 0)
        sigma0 = self.sigma0
        factor = np.sqrt(4 * np.pi) * GN / sigma0**2
        M = mu_00 / factor
        return M

    def M_encl(self, *r_list):

        # If no r values input, return list of M_encl at grid points
        if not (r_list):
            return self.M_encl_list()

        else:
            r_grid = self.r_list[1:]
            M_grid = self.M_encl_list()[1:]

            rmin = r_grid[0]
            rmax = self.r1

            r_arr = np.array(r_list, dtype="f")
            M = np.zeros_like(r_arr)

            # r within grid
            sel = (r_arr > rmin) & (r_arr < rmax)
            interp = InterpolatedUnivariateSpline(np.log(r_grid), np.log(M_grid))
            M[sel] = np.exp(interp(np.log(r_arr[sel])))

            # r below first grid point
            # Assume cubic enclosed mass (constant density)
            sel = r_arr < rmin
            M[sel] = (r_arr[sel] / rmin) ** 3 * M_grid[0]

            # r above last grid point
            # Return constant mass if r > r1
            sel = r_arr >= rmax
            M[sel] = M_grid[-1]

            # Remove outer bracket
            M = M[0]

            return M

    # Spherically averaged density at grid points
    def rho_sph_list(self):
        return self.rho_sph_avg()

    def rho_LM(self, L, M, *r):

        # If no r values, return evaluated at grid points
        if r == None:
            r_list = self.r_list

        # Else if r specified, need to remove outer brackets
        else:
            r_list = r[0]

        # Case with M != 0 not supported
        # Return 0
        if M != 0:
            return 0 if (np.ndim(r_list) == 0) else np.zeros_like(r_list)

        # Case with spherical symmetry
        if self.sph_sym_flag:

            if L != 0:
                return 0

            else:
                phi_b = (self.Phi_b(r_list) - self.Phi_b(0)) / self.sigma0**2
                phi_dm = self.phi(0, 0, r_list) * Z(0, 0, 0, 0)
                return self.rho0 * np.exp(-phi_dm - phi_b) * np.sqrt(4 * np.pi)

        # General cose
        else:

            # Construct array of phi_LM values
            phi_LM_arr = np.array([self.phi(L_, M_, r_list) for L_, M_ in zip(self.L_list, self.M_list)])

            # Construct phi_b
            if self.num_Phi_b_variables == 1:
                phi_b = lambda r, th: (self.Phi_b(r) - self.Phi_b(0)) / self.sigma0**2
            elif self.num_Phi_b_variables == 2:
                phi_b = lambda r, th: (self.Phi_b(r, th) - self.Phi_b(0, 0)) / self.sigma0**2
            else:
                raise Exception("Too many variables %d > 2 for Phi_b." % self.num_Phi_b_variables)

            # Case with one r value
            if np.ndim(r_list) == 0:

                def phi_dm(th):
                    Z_list = [Z(L_, M_, th, 0) for L_, M_ in zip(self.L_list, self.M_list)]
                    return np.dot(phi_LM_arr, Z_list)

                integrand = lambda th: 2 * np.pi * np.exp(-phi_dm(th) - phi_b(r_list, th)) * np.sin(th) * Z(L, M, th, 0)
                return self.rho0 * integrate(integrand, 0, np.pi)

            # Case with many r values
            else:

                rho = np.zeros_like(r_list, "f")
                for i, ri in enumerate(r_list):

                    def phi_dm(th):
                        Z_list = [Z(L_, M_, th, 0) for L_, M_ in zip(self.L_list, self.M_list)]
                        return np.dot(phi_LM_arr[:, i], Z_list)

                    integrand = lambda th: 2 * np.pi * np.exp(-phi_dm(th) - phi_b(ri, th)) * np.sin(th) * Z(L, M, th, 0)
                    rho[i] = self.rho0 * integrate(integrand, 0, np.pi)

                return rho

    def rho_sph_avg(self, *r):

        # If no r values, return evaluated at grid points
        if r == None:
            r_list = self.r_list

        # Else if r specified, need to remove outer brackets
        else:
            r_list = r[0]

        # Case with spherical symmetry
        if self.sph_sym_flag:
            phi_b = (self.Phi_b(r_list) - self.Phi_b(0)) / self.sigma0**2
            phi_dm = self.phi(0, 0, r_list) * Z(0, 0, 0, 0)
            return self.rho0 * np.exp(-phi_dm - phi_b)

        # General cose
        else:

            # Construct array of phi_LM values
            phi_LM_arr = np.array([self.phi(L, M, r_list) for L, M in zip(self.L_list, self.M_list)])

            # Construct phi_b
            if self.num_Phi_b_variables == 1:
                phi_b = lambda r, th: (self.Phi_b(r) - self.Phi_b(0)) / self.sigma0**2
            elif self.num_Phi_b_variables == 2:
                phi_b = lambda r, th: (self.Phi_b(r, th) - self.Phi_b(0, 0)) / self.sigma0**2
            else:
                raise Exception("Too many variables %d > 2 for Phi_b." % self.num_Phi_b_variables)

            # Case with one r value
            if np.ndim(r_list) == 0:

                def phi_dm(th):
                    Z_list = [Z(L, M, th, 0) for L, M in zip(self.L_list, self.M_list)]
                    return np.dot(phi_LM_arr, Z_list)

                integrand = lambda th: 0.5 * np.exp(-phi_dm(th) - phi_b(r_list, th)) * np.sin(th)
                return self.rho0 * integrate(integrand, 0, np.pi)

            # Case with many r values
            else:

                rho = np.zeros_like(r_list, "f")
                for i, ri in enumerate(r_list):

                    def phi_dm(th):
                        Z_list = [Z(L, M, th, 0) for L, M in zip(self.L_list, self.M_list)]
                        return np.dot(phi_LM_arr[:, i], Z_list)

                    integrand = lambda th: 0.5 * np.exp(-phi_dm(th) - phi_b(ri, th)) * np.sin(th)
                    rho[i] = self.rho0 * integrate(integrand, 0, np.pi)

                return rho

    def rho_sph(self, r, th=None):

        # No theta specified or spherically symmetric, return spherically average density
        if (th == None) or (self.sph_sym_flag):
            return self.rho_sph_avg(r)

        # Case where r and theta are single numbers
        elif (np.ndim(r) == 0) and (np.ndim(th) == 0):

            # Inner profile
            if r <= self.r1:

                if self.num_Phi_b_variables == 1:
                    phi_b = (self.Phi_b(r) - self.Phi_b(0)) / self.sigma0**2

                elif self.num_Phi_b_variables == 2:
                    phi_b = (self.Phi_b(r, th) - self.Phi_b(0, 0)) / self.sigma0**2

                else:
                    raise Exception("Too many variables %d > 2 for Phi_b." % self.num_Phi_b_variables)

                phi_LM_list = [self.phi(L, M, r) for L, M in zip(self.L_list, self.M_list)]
                Z_list = [Z(L, M, th, 0) for L, M in zip(self.L_list, self.M_list)]
                phi_dm = np.dot(phi_LM_list, Z_list)

                return self.rho0 * np.exp(-phi_dm - phi_b)

            # Outside grid
            else:
                return 0

        # Case where r is a list and theta is a single number
        elif (np.ndim(r) > 0) and (np.ndim(th) == 0):
            return np.array([self.rho_sph(ri, th) for ri in r])

        # Case where r is a single number and theta is a list
        elif (np.ndim(r) == 0) and (np.ndim(th) > 0):
            return np.array([self.rho_sph(r, th_i) for th_i in th])

        # Case where both r and theta are lists
        else:
            # Check have same lengths
            if len(r) != len(th):
                raise Exception("Not sure how to interpret r with len=%d and theta with len=%d" % (len(r), len(th)))

            else:
                return np.array([self.rho_sph(ri, th_i) for ri, th_i in zip(r, th)])

        # End of rho_sph (DM density)

    def cross_section(self, t_age=10, Nm=1):

        rho1 = self.rho_sph(self.r1)
        vrel = self.vrel

        # convert 1/(M_sol/kpc^3 * km/s * Gyr) to cm^2/g
        conversion = 4.685e9

        sigma_over_m = Nm / (rho1 * vrel * t_age) * conversion
        return sigma_over_m

    def angular_moments(self, r, L=0):

        if L in self.L_list:
            i = self.L_list.index(L)
            output = self.angular_moments_interp(r)[i]
        else:
            raise Exception("L=%d not found." % L)

        return output

    def update_angular_moments(self):

        # Calculate commonly-used integrals over angle
        # moment(r,L,M) = 1/(4pi) * int dOmeta Z(L,M,th,phi) * exp( - phi_dm(r,th,phi) - phi_b(r,th,phi) )
        # Currently assumes M=0 and no phi dependence

        self.angular_moments_list = np.zeros((self.num_r_points + 1, len(self.L_list)))

        # Want to evaluate these quantities at intermediate positions between grid points or at boundaries
        r_list = self.r_list
        r_points = np.concatenate(([r_list[0]], 0.5 * (r_list[:-1] + r_list[1:]), [r_list[-1]]))

        L_list = self.L_list
        num_size = len(L_list)

        phi_list = np.array(
            [self.y[i * self.num_func : i * self.num_func + num_size] for i in range(self.num_r_points)]
        )
        phi_at_r_points = np.concatenate(([phi_list[0]], 0.5 * (phi_list[:-1] + phi_list[1:]), [phi_list[-1]]))

        for i in range(len(r_points)):

            r = r_points[i]

            if (self.num_Phi_b_variables == 1) & (L_list == [0]):

                phi_b = (self.Phi_b(r) - self.Phi_b(0)) / self.sigma0**2
                phi_dm = Z(0, 0, 0, 0) * phi_at_r_points[i, 0]

                self.angular_moments_list[i, 0] = Z(0, 0, 0, 0) * np.exp(-phi_b - phi_dm)

            elif (self.num_Phi_b_variables == 1) & (L_list != [0]):

                phi_b = (self.Phi_b(r) - self.Phi_b(0)) / self.sigma0**2
                phi_dm = lambda th: np.sum([Z(L_list[_j], 0, th, 0) * phi_at_r_points[i, _j] for _j in range(num_size)])

                for j in range(num_size):

                    integrand = lambda th: Z(L_list[j], 0, th, 0) * np.exp(-phi_b - phi_dm(th)) * np.sin(th)
                    self.angular_moments_list[i, j] = 0.5 * integrate(integrand, 0, np.pi)

            elif self.num_Phi_b_variables == 2:

                phi_b = lambda th: (self.Phi_b(r, th) - self.Phi_b(0, th)) / self.sigma0**2
                phi_dm = lambda th: np.sum([Z(L_list[_j], 0, th, 0) * phi_at_r_points[i, _j] for _j in range(num_size)])

                for j in range(num_size):

                    integrand = lambda th: Z(L_list[j], 0, th, 0) * np.exp(-phi_b(th) - phi_dm(th)) * np.sin(th)
                    self.angular_moments_list[i, j] = 0.5 * integrate(integrand, 0, np.pi)

            else:
                raise Exception("Case with %d not supported for angular_moments." % self.num_Phi_b_variables)

        self.angular_moments_interp = interp1d(r_points, self.angular_moments_list, axis=0)

    # end of update_angular_moments()

    def update(self, **updated_inputs):

        for key, value in updated_inputs.items():

            # Simply updates values
            if key == "y":
                self.y = value
            if key == "params":
                self.params = value
            if key == "Phi_b":
                self.Phi_b = value

            # Updates y to a new grid by making interpolating functions in phi and mu
            # Note: need to have same r1 value, otherwise will generate an error if r1_new > r1_old
            if key == "r_list":

                new_r_list = value
                old_r_list = self.r_list
                new_y = np.zeros(2 * len(self.L_list) * len(new_r_list))

                for i in range(len(self.L_list)):

                    L = self.L_list[i]
                    M = self.M_list[i]

                    phi_int = interp1d(old_r_list, self.phi(L, M), kind="linear")
                    new_y[i :: self.num_func] = [phi_int(r) for r in new_r_list]

                    mu_int = interp1d(old_r_list, self.mu(L, M), kind="linear")
                    new_y[i + len(self.L_list) :: self.num_func] = [mu_int(r) for r in new_r_list]

                self.y = new_y
                self.r_list = new_r_list

            # Set True/False if want to compute commonly-used integrals
            if key == "angular_moments_set":
                self.angular_moments_set = value

        # Update all dependent quantities
        self.r0 = self.params[0] ** 0.5
        self.sigma0 = self.params[1] ** 0.5
        self.rho0 = self.sigma0**2 / (4 * np.pi * GN * self.r0**2)

        self.vrel = self.sigma0 * 4 / np.sqrt(np.pi)
        self.r1 = self.r_list[-1]

        self.num_r_points = len(self.r_list)

        # need to update L_list as last step
        for key, value in updated_inputs.items():

            # Update L,M lists
            if key == "L_list":

                old_L_list = self.L_list

                # Update L_list and M_list
                # Only take unique values, assuming M=0 only
                new_L_list = list(np.unique(value))
                new_M_list = [0] * len(new_L_list)

                # Existing phi,mu values retained, new ones set to zero
                new_y = np.zeros(2 * len(new_L_list) * len(self.r_list))

                for L in old_L_list:

                    if L in new_L_list:

                        i = new_L_list.index(L)
                        M = new_M_list[i]
                        new_y[i :: 2 * len(new_L_list)] = self.phi(L, M)
                        new_y[i + len(new_L_list) :: 2 * len(new_L_list)] = self.mu(L, M)

                self.y = new_y
                self.L_list = new_L_list
                self.M_list = new_M_list
                self.num_func = 2 * len(self.L_list)

            if key == "M_list":
                raise Exception("Updating M values is not supported.")

        if self.angular_moments_set:
            self.update_angular_moments()

    # end of update()

    def copy(self):
        return isothermal_profile(
            self.y,
            self.params,
            self.r_list,
            Phi_b=self.Phi_b,
            L_list=self.L_list,
            M_list=self.M_list,
        )

    # # Makes dictionary of outputs that can be saved
    # def tosave(self, **extra):
    #     return {
    #         "y": self.y,
    #         "params": self.params,
    #         "r_list": self.r_list,
    #         "L_list": self.L_list,
    #         "M_list": self.M_list,
    #     }


###############
# CDM_profile #
###############


class CDM_profile:

    def __init__(
        self,
        *inputs,
        q0=1,
        # alpha=None,
        M_b=None,
        Phi_b=None,
        AC_prescription=None,
        halo_type="NFW",
        gamma=0.3,
        Gnedin_params=(1.6, 0.8),
        input_from_profile_params=False,
        **extraneous,
    ):

        # # Check inputs are correct
        # if len(inputs) != 2:
        #     raise Exception(
        #         "inputs must be 'M200,c' (defaults) or 'rhos,rs' (with input_NFW=True)"
        #     )

        # # Assume input rhos, rs
        # if input_NFW:
        #     self.rhos, self.rs = inputs
        #     self.M200, self.c, self.r200 = NFW_parameters_to_mass_concentration(*inputs)
        # # Assume input M200, c
        # else:
        #     self.M200, self.c = inputs
        #     self.rhos, self.rs, self.r200 = mass_concentration_to_NFW_parameters(
        #         *inputs
        #     )

        # # Virial mass and concentration parameters
        # M200, c = self.M200, self.c

        # Check inputs are correct
        if len(inputs) != 2:
            raise Exception("inputs must be 'M200,c' (defaults) or 'rhos,rs' (with input_NFW=True).")

        self.halo_type = halo_type

        if halo_type == "NFW":
            if input_from_profile_params:
                # Assume rhos, rs inputs
                self.rhos, self.rs = inputs
                self.M200, self.c, self.r200 = NFW_parameters_to_mass_concentration(*inputs)
            else:
                # Assume M200, c inputs
                self.M200, self.c = inputs
                self.rhos, self.rs, self.r200 = mass_concentration_to_NFW_parameters(*inputs)

        elif halo_type == "Einasto":
            self.gamma = gamma
            if input_from_profile_params:
                # Assume rhos, rs inputs
                self.rhos, self.rs = inputs
                self.M200, self.c, self.r200 = Einasto_parameters_to_mass_concentration(
                    *inputs, alpha=self.gamma
                )  # alpha keyword used to avoid name conflict with gamma function used in cdm.py
            else:
                # Assume M200, c inputs
                self.M200, self.c = inputs
                self.rhos, self.rs, self.r200 = mass_concentration_to_Einasto_parameters(
                    *inputs, alpha=self.gamma
                )  # alpha keyword used to avoid name conflict with gamma function used in cdm.py
        else:
            raise Exception("halo_type must be 'NFW' or 'Einasto'.")

        # # Assume input rhos, rs
        # if input_NFW:
        #     self.rhos, self.rs = inputs
        #     self.M200, self.c, self.r200 = NFW_parameters_to_mass_concentration(*inputs)
        # # Assume input M200, c
        # elif (not input_NFW) and (not alpha):
        #     self.M200, self.c = inputs
        #     self.rhos, self.rs, self.r200 = mass_concentration_to_NFW_parameters(*inputs)
        # # Assume input M200, c, alpha
        # elif (not input_NFW) and (alpha):
        #     self.M200, self.c, self.alpha = inputs + (alpha,)  # inputs is a tuple
        #     self.rhos, self.rs, self.r200 = mass_concentration_to_NFW_parameters(*inputs)
        # else:
        #     raise Exception("inputs inconsistent with NFW or Einasto profiles.")

        # Virial mass and concentration parameters
        M200, c = self.M200, self.c

        # Nonsphericity
        self.q0 = q0
        self.sph_sym_flag = q0 == 1

        # AC parameters
        self.AC_prescription = AC_prescription
        self.Gnedin_params = Gnedin_params

        # Baryon potential
        if Phi_b:
            self.Phi_b = Phi_b
            self._Phi_b_user = Phi_b  # Store original user-supplied function for serialization
        else:
            self.Phi_b = no_baryons
            self._Phi_b_user = None

        # Calculate baryon enclosed mass profile
        # If M_b set, use input function M_b for baryon profile
        if M_b:
            self.M_b = M_b

        # Use Phi_b to calculate M_b according to d/dr <Phi_b>_spherical_avg = GM_b/r^2
        # Only if Phi_b is nonzero
        elif Phi_b:
            self.M_b = compute_Mb(Phi_b, 1e-10 * self.r200, 1e2 * self.r200)

        # No baryons
        else:
            self.M_b = no_baryons

        # # Load spherically symmetric density and enclosed mass functions
        # # NFW case
        # if AC_prescription == None:
        #     rho, M_encl = NFW_profiles(M200, c, mass_concentration=True)

        # # Adiabatic contraction cases
        # # AC_prescription = 'Gnedin' or 'Cautun'
        # else:
        #     rho, M_encl = AC_profiles(
        #         M200,
        #         c,
        #         self.M_b,
        #         AC_prescription=AC_prescription,
        #         Gnedin_params=Gnedin_params,
        #     )
        # Load spherically symmetric density and enclosed mass functions
        # NFW case
        if AC_prescription == None and halo_type == "NFW":
            rho, M_encl = NFW_profiles(M200, c, mass_concentration=True)
        # Einasto case
        elif AC_prescription == None and halo_type == "Einasto":
            rho, M_encl = Einasto_profiles(M200, c, self.gamma, mass_concentration=True)
        # Adiabatic contraction cases
        # AC_prescription = 'Gnedin' or 'Cautun'
        else:
            rho, M_encl = AC_profiles(
                M200,
                c,
                self.M_b,
                AC_prescription=AC_prescription,
                Gnedin_params=Gnedin_params,
            )

        # Spherically symmetric density profile
        def rho_sph_sym(r):
            if np.ndim(r) == 0:
                return float(rho(r))
            else:
                return np.array([rho(ri) for ri in r])

        self.rho_sph_sym = rho_sph_sym

        # Enclosed mass profile of spherically symmetric density profile
        def M_encl_sym(r):
            if np.ndim(r) == 0:
                return float(M_encl(r))
            else:
                return np.array([M_encl(ri) for ri in r])

        self.M_encl_sym = M_encl_sym

    # end __init__

    # Define spheroidal density function in cylindrical coordinates by substituting r -> r_sph
    def rho_cyl(self, R, z):
        # spheroidal r
        r_sph = np.sqrt((R * self.q0 ** (1 / 3.0)) ** 2 + (z * self.q0 ** (-2 / 3.0)) ** 2)
        return self.rho_sph_sym(r_sph)

    # Density in spherical coordinates
    def rho_sph(self, r, th=None):

        # Spherically symmetric case
        if self.q0 == 1:
            return self.rho_sph_sym(r)

        # No theta specified
        elif th == None:
            return self.rho_sph_avg(r)

        # Case where r and theta are single numbers
        elif (np.ndim(r) == 0) and (np.ndim(th) == 0):
            return self.rho_cyl(r * np.sin(th), r * np.cos(th))

        # Case where r is a list and theta is a single number
        elif (np.ndim(r) > 0) and (np.ndim(th) == 0):
            return np.array([self.rho_sph(ri, th) for ri in r])

        # Case where r is a single number and theta is a list
        elif (np.ndim(r) == 0) and (np.ndim(th) > 0):
            return np.array([self.rho_sph(r, th_i) for th_i in th])

        # Case where both r and theta are lists
        else:
            # Check have same lengths
            if len(r) != len(th):
                raise Exception("Not sure how to interpret r with len=%d and theta with len=%d" % (len(r), len(th)))

            else:
                return np.array([self.rho_sph(ri, th_i) for ri, th_i in zip(r, th)])

        # End of rho_sph (DM density)

    def rho_sph_avg(self, r):

        # Spherically symmetric case
        if self.q0 == 1:
            return self.rho_sph_sym(r)

        # Case where r is single number
        elif np.ndim(r) == 0:
            integrand = lambda th: 0.5 * self.rho_cyl(r * np.sin(th), r * np.cos(th)) * np.sin(th)
            return integrate(integrand, 0, np.pi)

        else:
            return np.array([self.rho_sph_avg(ri) for ri in r])

    def M_encl(self, r_encl, num=300, k=3):

        # r_encl is a number
        if np.ndim(r_encl) == 0:
            if r_encl < 0:
                raise Exception("r=%f is not >0." % r_encl)
            elif r_encl == 0:
                return 0
            else:
                rmin = 1e-4 * min(self.rs, r_encl)
                rmax = r_encl

        # r_encl is list or array
        else:
            if np.any(np.array(r_encl) < 0):
                raise Exception("r is not >0.")
            elif np.all(np.array(r_encl) == 0):
                return np.zeros_like(r_encl)
            else:
                rmin = 1e-4 * min(self.rs, np.amin(np.array(r_encl)[np.array(r_encl) > 0]))
                rmax = np.amax(r_encl)

        def integrand(r):
            if r == 0:
                return 0
            else:
                return 4 * np.pi * r**2 * self.rho_sph(r)

        # Generate interpolation function for M_encl
        r_list = np.geomspace(rmin, rmax, num=num)
        integrand_list = [integrand(r) for r in r_list]
        Mmin = integrate(integrand, 0, rmin)
        integrand_interp = InterpolatedUnivariateSpline(r_list, integrand_list, k=k)

        if np.ndim(r_encl) == 0:
            return float(integrand_interp.integral(rmin, r_encl) + Mmin)
        else:
            sel = np.array(r_encl) > 0
            output = np.zeros_like(r_encl, "f")
            output[sel] = np.array([integrand_interp.integral(rmin, r) + Mmin for r in np.array(r_encl)[sel]])
            return output

    # Compute boundary conditions
    def compute_boundary_conditions(self, r1, L_list=[0], M_list=[0]):

        # Integrals over outer halo needed for matching onto potential with
        # correct asymptotic form
        self.potential_moments = self.compute_potential_moments(r1, L_list=L_list, M_list=M_list)

        # Remember which L,M values the moments correspond to
        self.potential_moments_L_list = L_list
        self.potential_moments_M_list = M_list

    # Note currently neglects azimuthal phi dependence and M values
    # Moment is int_r1^inf dx x^(1-L) rho_LM(x)
    def compute_potential_moments(self, r1, L_list=[0], M_list=[0]):

        if M_list != [0]:
            raise Exception("M != 0 not supported.")
        else:
            pass

        moments = []

        # Try to use moments that were previously evaluated to avoid repeating calculations
        try:
            old_moments = self.potential_moments
            old_L_list = self.potential_moments_L_list
            old_M_list = self.potential_moments_M_list

        except:
            old_L_list = []

        for i, L in enumerate(L_list):

            # Try to use previous value if it exists
            if L in old_L_list:
                moments.append(old_moments[old_L_list.index(L)])

            # Neglect L=0 moment
            # Not needed
            # elif L == 0:
            #    moments.append(0)

            # Moment vanishes if for L>0 if q=1
            elif (L > 0) & (self.q0 == 1):
                moments.append(0)

            else:

                # Initialize
                moment = 0
                rmin = r1
                rmax = 10 * rmin

                # Loop until moment converges
                max_iter = 10
                for i in range(max_iter):

                    # Compute rho_L for CDM halo
                    def rho_L(r):
                        integrand = lambda th: 2 * np.pi * np.sin(th) * Z(L, 0, th, 0) * self.rho_sph(r, th)
                        return integrate(integrand, 0, np.pi)

                    dJ_dr = lambda r, y: r ** (1 - L) * rho_L(r)
                    solution = solve_ivp(dJ_dr, [rmin, rmax], [0], rtol=1e-8)
                    extra = solution.y[0][-1]
                    moment += extra

                    # Check if converged
                    if np.allclose(moment, moment - extra, rtol=1e-5):
                        moments.append(moment)
                        break
                    else:
                        rmin = rmax
                        rmax = 10 * rmin

                # End while loop

        # End for loop over L

        return np.array(moments)
