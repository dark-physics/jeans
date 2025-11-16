import numpy as np
import jeans
from jeans.definitions import GN


# Miyamoto-Nagai
def Phi_MN(Md, a, b):
    func = (
        lambda r, th: -GN
        * Md
        / np.sqrt(
            r**2 * np.sin(th) ** 2 + (a + np.sqrt(b**2 + r**2 * np.cos(th) ** 2)) ** 2
        )
    )
    return func


def Phi_MN_shifted(Md, a, b, z0):
    func = (
        lambda r, th: -GN
        * Md
        / np.sqrt(
            r**2 * np.sin(th) ** 2
            + (a + np.sqrt(b**2 + (z0 + r * np.cos(th)) ** 2)) ** 2
        )
    )
    return func


# Spheroidal Hernquist
def Phi_Hern(MH, a, q):
    r_sph = lambda R, z: np.sqrt((R * q ** (1 / 3.0)) ** 2 + (z * q ** (-2 / 3.0)) ** 2)
    func = lambda r, th: -GN * MH / (r_sph(r * np.sin(th), r * np.cos(th)) + a)
    return func


# Hernquist
def M_Hernquist(M, a):
    func = lambda r: M * r**2 / (r + a) ** 2
    return func


def M_Hernquist_shift(M, a, z0):
    func = (
        lambda R, z: M
        * (R**2 + (z + z0) ** 2)
        / (np.sqrt(R**2 + (z + z0) ** 2) + a) ** 2
    )
    return func


def Phi_Hernquist(M, a):
    func = lambda r: -GN * M / (r + a)
    return func


def Phi_Hernquist_shifted(M, a, z0):
    func = lambda r, z: -GN * M / (np.sqrt(r**2 + (z + z0) ** 2) + a)
    return func


# Plummer
def Phi_Plummer(M, b):
    func = lambda r: -GN * M / np.sqrt(r**2 + b**2)
    return func


def Phi_Plummer_shifted(M, b, z0):
    func = lambda r, th: -GN * M / np.sqrt(r**2 + (z0 + r * np.cos(th)) ** 2 + b**2)
    return func


# exponential disk potential (spherical approximation)
def Phi_exp_disk(Md, Rs):
    func = lambda r, th: -GN * Md * (1 - np.exp(-r / Rs) * (1 + r / Rs)) / r
    return func
