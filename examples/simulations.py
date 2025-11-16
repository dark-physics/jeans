import numpy as np
import os
import h5py
from scipy.interpolate import (
    interp1d,
    RectBivariateSpline,
    InterpolatedUnivariateSpline,
)
from scipy.integrate import solve_ivp, quad
from scipy.optimize import brentq
from scipy.special import lpmv, factorial

base_path = os.getcwd()  # change for your setup


class fit:
    def __init__(self, halo_id, data_path="/data/EAGLE-50-data/", model="SIDM1b", verbose=False):
        self.data_path = base_path + data_path
        self.halo_id = halo_id
        self.model = model
        self._model_list = ["SIDM1b", "CDMb", "vdSIDMb"]
        self.hid = int(halo_id) - 1  # halo index in the data files (0-indexed)

        # Check is data path exists
        if not os.path.exists(self.data_path):
            raise Exception(f"Data path {self.data_path} does not exist.")

        # Check if model is valid
        if self.model not in self._model_list:
            raise Exception(f"Model {self.model} not recognized. Choose from {self._model_list}.")

        if verbose:
            print(f"Loading data for halo {self.halo_id} with model {self.model}...")

        sph_filename = self.data_path + f"{self.model}_sphericallyAveraged_density_profiles.hdf5"
        cyl_filename = self.data_path + f"{self.model}_cylindrical_density_and_potential.hdf5"
        axi_filename = self.data_path + f"{self.model}_axisymmetric_shape_profiles.hdf5"

        # Check if files exist
        for filename in [sph_filename, cyl_filename, axi_filename]:
            if not os.path.isfile(filename):
                print(f"File {filename} not found. If you do not need this file, ignore this message...")

        # Load spherical data
        self.sph_data = {}
        try:
            with h5py.File(sph_filename, "r") as f:
                for key in f.keys():
                    self.sph_data[key] = f[key][self.hid]
        except Exception as e:
            print(f"Error loading spherical data: {e}")

        self.cyl_data = {}
        try:
            with h5py.File(cyl_filename, "r") as f:
                for key in f.keys():
                    self.cyl_data[key] = f[key][self.hid]
        except Exception as e:
            print(f"Error loading cylindrical data: {e}")

        self.shape_data = {}
        try:
            with h5py.File(axi_filename, "r") as f:
                for key in f.keys():
                    self.shape_data[key] = f[key][self.hid]
        except Exception as e:
            print(f"Error loading axisymmetric data: {e}")

        if verbose:
            print("Data loaded into dictionary attributes: sph_data, cyl_data, shape_data.")

        # Load halo properties as attributed of the class
        self.M200 = self.sph_data.get("M200", np.nan)
        self.R200 = self.sph_data.get("r200", np.nan) * 1e3  # convert to kpc
        self.Redges = self.cyl_data.get("Redges", np.nan) * 1e3  # convert to kpc
        self.Zedges = self.cyl_data.get("Zedges", np.nan) * 1e3  # convert to kpc
        self.Rcent = 0.5 * (self.Redges[1:] + self.Redges[:-1])
        self.Zcent = 0.5 * (self.Zedges[1:] + self.Zedges[:-1])

        # 1D baryon potential
        self.Phi_b_sph = compute_Phi_b_spherical(self.sph_data)

        # 2D baryon potential
        self.Phi_b = compute_Phi_b_cylindrical(self.cyl_data, self.sph_data)

        # Baryon enclosed mass profile
        self.Mb = compute_Mb(self.sph_data)

        # Mass per particle in simulations
        self.mass_dm_particle = 9.70e6  # Msol

    # Class Methods -- easy access to common profiles
    # SPHERICALLY AVERAGED PROFILES
    def sph_avg_dm_density(self):
        r_list = self.sph_data["rs"] * 1e3  # convert to kpc
        rho = self.sph_data["dm_rho"] * 1e-9  # convert to Msol/kpc^3
        return {"r": r_list, "rho": rho}

    def sph_avg_star_density(self):
        r_list = self.sph_data["rs"] * 1e3  # convert to kpc
        rho = self.sph_data["star_rho"] * 1e-9  # convert to Msol/kpc^3
        return {"r": r_list, "rho": rho}

    def sph_avg_gas_density(self):
        r_list = self.sph_data["rs"] * 1e3  # convert to kpc
        rho = self.sph_data["gas_rho"] * 1e-9  # convert to Msol/kpc^3
        return {"r": r_list, "rho": rho}

    def sph_avg_bh_density(self):
        r_list = self.sph_data["rs"] * 1e3  # convert to kpc
        rho = self.sph_data["bh_rho"] * 1e-9  # convert to Msol/kpc^3
        return {"r": r_list, "rho": rho}

    def sph_avg_baryon_density(self):
        r_list = self.sph_data["rs"] * 1e3  # convert to kpc
        rho_star = self.sph_data["star_rho"] * 1e-9  # convert to Msol/kpc^3
        rho_gas = self.sph_data["gas_rho"] * 1e-9  # convert to Msol/kpc^3
        rho_bh = self.sph_data["bh_rho"] * 1e-9  # convert to Msol/kpc^3
        rho = rho_star + rho_gas + rho_bh
        return {"r": r_list, "rho": rho}

    def dm_mass_enclosed(self):
        r_list = self.sph_data["redges"] * 1e3  # convert to kpc
        M_dm = self.sph_data["dm_M"]  # Msol
        return {"r": r_list, "M_dm": M_dm}

    def star_mass_enclosed(self):
        r_list = self.sph_data["rs"] * 1e3  # convert to kpc
        M_star = self.sph_data["star_M"]  # Msol
        return {"r": r_list, "M_star": M_star}

    def gas_mass_enclosed(self):
        r_list = self.sph_data["rs"] * 1e3  # convert to kpc
        M_gas = self.sph_data["gas_M"]  # Msol
        return {"r": r_list, "M_gas": M_gas}

    def bh_mass_enclosed(self):
        r_list = self.sph_data["rs"] * 1e3  # convert to kpc
        M_bh = self.sph_data["bh_M"]  # Msol
        return {"r": r_list, "M_bh": M_bh}

    def cyl_dm_density(self):
        rho_2D = self.cyl_data["smooth_dm_rho"] * 1e-9  # convert to Msol/kpc^3
        extent = [self.Redges[0], self.Redges[-1], self.Zedges[0], self.Zedges[-1]]
        return {"R": self.Rcent, "z": self.Zcent, "rho_2D": rho_2D, "extent": extent}

    def cyl_star_density(self):
        rho_2D = self.cyl_data["star_rho"] * 1e-9  # convert to Msol/kpc^3
        extent = [self.Redges[0], self.Redges[-1], self.Zedges[0], self.Zedges[-1]]
        return {"R": self.Rcent, "z": self.Zcent, "rho_2D": rho_2D, "extent": extent}

    def cyl_gas_density(self):
        rho_2D = self.cyl_data["gas_rho"] * 1e-9  # convert to Msol/kpc^3
        extent = [self.Redges[0], self.Redges[-1], self.Zedges[0], self.Zedges[-1]]
        return {"R": self.Rcent, "z": self.Zcent, "rho_2D": rho_2D, "extent": extent}

    def cyl_bh_density(self):
        rho_2D = self.cyl_data["bh_rho"] * 1e-9  # convert to Msol/kpc^3
        extent = [self.Redges[0], self.Redges[-1], self.Zedges[0], self.Zedges[-1]]
        return {"R": self.Rcent, "z": self.Zcent, "rho_2D": rho_2D, "extent": extent}

    def cyl_baryon_density(self):
        rho_2D = (
            self.cyl_data["star_rho"] + self.cyl_data["gas_rho"] + self.cyl_data["bh_rho"]
        ) * 1e-9  # convert to Msol/kpc^3
        extent = [self.Redges[0], self.Redges[-1], self.Zedges[0], self.Zedges[-1]]
        return {"R": self.Rcent, "z": self.Zcent, "rho_2D": rho_2D, "extent": extent}

    def cyl_dm_density_func(self):
        rho_2D = self.cyl_data["smooth_dm_rho"] * 1e-9  # convert to Msol/kpc^3
        extent = [self.Redges[0], self.Redges[-1], self.Zedges[0], self.Zedges[-1]]
        rho_2D_int = RectBivariateSpline(self.Rcent, self.Zcent, rho_2D, bbox=extent, kx=3, ky=3)
        return rho_2D_int

    # SHAPE PROFILES
    def q_dm(self):
        r_list = self.shape_data["reff"] * 1e3  # convert to kpc
        q = self.shape_data["dm_Q"]
        error = self.shape_data["dm_Q_error"]
        N = self.shape_data["dm_N"]
        return {"r": r_list, "q": q, "error": error, "N": N}

    def q_stars(self):
        r_list = self.shape_data["reff"] * 1e3  # convert to kpc
        q = self.shape_data["star_Q"]
        N = self.shape_data["star_N"]
        return {"r": r_list, "q": q, "N": N}

    def q_gas(self):
        r_list = self.shape_data["reff"] * 1e3  # convert to kpc
        q = self.shape_data["gas_Q"]
        N = self.shape_data["gas_N"]
        return {"r": r_list, "q": q, "N": N}

    # def q_baryon(self):
    #     r_list = self.shape_data["reff"] * 1e3  # convert to kpc
    #     q_star = self.shape_data["star_Q"]
    #     q_gas = self.shape_data["gas_Q"]
    #     q_baryon = np.average([q_star, q_gas])  # simple average
    #     return {"r": r_list, "q": q_baryon}


# Helper functions
def compute_Mb(data, rmin=0.5):

    # Define interpolating function
    M_stars_raw = data["star_M"]
    r_stars_raw = data["redges"] * 1e3  # kpc

    # If no rmin set, use entire range
    # Can have numerical issues for adiabatic contraction at small radii
    if rmin:
        select = r_stars_raw > rmin
        r_stars = r_stars_raw[select]
        M_stars = M_stars_raw[select]
    else:
        r_stars = r_stars_raw
        M_stars = M_stars_raw

    rmin = min(r_stars)
    rmax = max(r_stars)

    M_int = InterpolatedUnivariateSpline(r_stars, M_stars, k=1)

    # Define Mb as piecewise function
    def Mb(r):
        if (r <= rmin) and (r >= 0):
            output = M_stars[0] * (r / rmin) ** 3
        elif (r > rmin) and (r < rmax):
            output = M_int(r)
        elif r >= rmax:
            output = M_stars[-1]
        else:
            print(r, "value for r not valid")
            output = 0
        return output

    return np.vectorize(Mb)


def compute_Phi_b_spherical(data):

    GN = 4.302e-6  # km^2/s^2*kpc/Msol

    Mb = compute_Mb(data)

    rmin = data["redges"][0] * 1e3  # kpc
    rmax = data["redges"][-1] * 1e3  # kpc

    Mmin = data["star_M"][0]
    Mmax = data["star_M"][-1]

    # Let y = Phi(r) - Phi(0)
    # RHS = lambda r, y: GN*Mb(r)/r**2
    # Added by Adam to fix multiprocessing pickle issue
    def RHS(r, y):
        return GN * Mb(r) / r**2

    # Boundary condition y(rmin) = 0.5*GN*M(rmin)/rmin
    # Assuming constant density sphere for r < rmin
    ymin = 0.5 * GN * Mmin / rmin
    solution = solve_ivp(RHS, [rmin, rmax], [ymin], dense_output=True, atol=1e-8, rtol=1e-8)
    y_int = solution.sol
    ymax = y_int(rmax)[0]

    # Assume Phi(r) like a point charge for r >= rmax
    # Then Phi(rmax) = y(rmax) + Phi(0) = -GN*Mmax/rmax
    # which implies Phi(0) = -GN*Mmax/rmax - ymax
    Phi_0 = -GN * Mb(rmax) / rmax - ymax

    def Phi_b(r):
        if (r <= rmin) and (r >= 0):
            output = ymin * (r / rmin) ** 2 + Phi_0
        elif (r > rmin) and (r < rmax):
            output = y_int(r)[0] + Phi_0
        elif r >= rmax:
            output = -GN * Mb(rmax) / r
        else:
            print(r, "value for r not valid")
            output = 0
        return output

    return Phi_b


def compute_Phi_b_cylindrical(data, sphdata, file="star_potential_sphere", Lmax=10):

    Phi_b_2D_data = -abs(data[file])
    R_edges = data["Redges"] * 1e3  # kpc
    z_edges = data["Zedges"] * 1e3  # kpc

    R_cent = 0.5 * (R_edges[1:] + R_edges[:-1])
    z_cent = 0.5 * (z_edges[1:] + z_edges[:-1])

    rmax = R_edges[-1]
    extent = [R_edges[0], R_edges[-1], z_edges[0], z_edges[-1]]

    Phi_b_2D_int = RectBivariateSpline(R_cent, z_cent, Phi_b_2D_data, bbox=extent, kx=3, ky=3)

    Phi_b_spherical = compute_Phi_b_spherical(sphdata)

    integrand = lambda th: Phi_b_2D_int(rmax * np.sin(th), rmax * np.cos(th))[0][0] * np.sin(th)
    Delta_Phi = Phi_b_spherical(rmax) - 0.5 * integrate(integrand, 0, np.pi)

    # Compute multipole moments
    # Needs to be generalized for 3D case
    Q = np.zeros(Lmax + 1)
    for L in range(1, Lmax + 1):

        integrand_L = lambda th: Z(L, 0, th, 0) * integrand(th)
        Q[L] = -2 * np.pi * integrate(integrand_L, 0, np.pi)

    # print(Q)

    def Phi_b_multipole(r, th):
        return np.sum([-Q[L] * (rmax / r) ** L * Z(L, 0, th, 0) for L in range(Lmax + 1)])

    def Phi_b(r, th):

        if r >= rmax:
            output = Phi_b_spherical(r) + Phi_b_multipole(r, th)
        else:
            R = r * np.sin(th)
            z = r * np.cos(th)

            # Delta_Phi = Phi_b_spherical(rmax) - Phi_b_2D_int(rmax*np.sin(th),rmax*np.cos(th))[0][0]
            output = Phi_b_2D_int(R, z)[0][0] + Delta_Phi

        return output

    return Phi_b


# q_baryon for baryon potential
def compute_q_phib(data, file="star_potential_sphere"):

    Phi_data = data[file]
    Phi_radial_slice = 0.5 * (Phi_data[:, 99] + Phi_data[:, 100])
    Phi_axial_slice = 0.5 * (np.flip(Phi_data[0, :100]) + Phi_data[0, 100:])

    z0 = 0.5
    R0 = 0.5

    R = np.arange(0.5, 100, 1)
    z = np.arange(0.5, 100, 1)

    log_Phi_radial_interp = InterpolatedUnivariateSpline(np.log(R), np.log(Phi_radial_slice), ext=3)
    log_Phi_axial_interp = InterpolatedUnivariateSpline(np.log(z), np.log(Phi_axial_slice), ext=3)

    r_list = np.linspace(1, 50, num=100)
    q_list = []

    for r in r_list:

        # Require arguments of logs to be positive
        # This requires (z0/r)^3/2 < q < (r/R0)^3
        # Include a buffer so avoid exactly at min or max
        q_abs_min = 1.1 * (z0 / r) ** 1.5
        q_abs_max = 0.9 * (r / R0) ** 3

        def f(q):
            logz = 0.5 * np.log(r**2 * q ** (4 / 3) - R0**2 * q**2)
            logR = 0.5 * np.log(r**2 * q ** (-2 / 3) - z0**2 * q**-2)
            return log_Phi_radial_interp(logR) - log_Phi_axial_interp(logz)

        qmin, qmax = 0.5, 2
        while True:

            # Check whether reached absolute min or max
            if (qmax == q_abs_max) and (qmin == q_abs_min):
                q_list.append(1)
                break

            # Find solution
            try:
                q_list.append(brentq(f, qmin, qmax))
                break
            except:
                # Don't look outside min/max range for q
                qmin = max(0.8 * qmin, q_abs_min)
                qmax = min(2 * qmax, q_abs_max)

    return InterpolatedUnivariateSpline(r_list, q_list)


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
