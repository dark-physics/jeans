"""
run_dict.py
-----------
This is the main configuration file for the nonspherical SIDM Jeans modeling package.

Edit this file to define and customize your dark matter halo and baryon profile models.

Instructions:
- Modify the `run_dictionary` below to set model parameters, halo properties, and options.
- If r1, M200, c, q0, or gamma are set to scalars, the run script will generate a single profile.
- If r1, M200, c, q0, or gamma are set to lists or arrays, the run script will scan over all combinations of parameters.
- Define or import your baryon potential functions as needed.
- Define a custom halo shape function if desired.
- Uncomment and update lines as necessary to include your custom functions or settings.
- Save and run your run_jeans.py with 'python run_jeans.py' to generate and save profiles using these settings.

For more information, see the documentation or example notebooks.
"""

from analytic_potentials import Phi_MN  # Add any custom baryon potential functions here
import numpy as np


# fmt: off

###################### DICTIONARY CONFIGURATION ######################
# EXAMPLE RUN DICTIONARY. EDIT THIS TO SET PARAMETERS AND OPTIONS.
# SINGLE PROFILE GENERATION IF ALL PARAMETERS ARE SCALARS.
run_dictionary = {
    "model": "spherical",  # Options: 'spherical', 'cdm', 'squashed' or 'isothermal'. Setting to cdm will override r1 st r1=0. Setting to spherical will override q0 so that q0=1.
    "r1": 10,  # kpc. Matching radius for SIDM and CDM halos. If r1=0 CDM halo is returned.
    "M200": 1e12,  # Msun
    "c": 10.0,  # Dimensionless. Concentration parameter. c = r200/rs.
    "q0": 1.0,  # Dimensionless. Initial outer halo shape. If q0=1, spherical halo is assumed.
    "halo_type": "Einasto",  # Options: 'NFW' or 'Einasto'. Outer halo type.
    "gamma": 0.31,  # Dimensionless. Halo flattening parameter for Einasto profile. 
    "Phi_b": None,  # (km/s)^2. Baryon potential. Must be a function with signature Phi_b(r, theta), even if spherical.
    "AC_prescription": None,  # Adiabatic contraction prescription. Options: 'Cautun' or 'Gnedin'.
    "Gnedin_params": (1.6, 0.8,),  # Only used if AC_prescription='Gnedin'. (A, w) parameters.
    "save_profile": True,  # If True, saves the profile to a .npz file.
    "save_dir": "data/",  # Relative path to save the profile .npz file.
    "verbose": False,  # If True, prints progress and warnings.
    "L_list": [0],  # Angular momentum modes to include in the isothermal model.
    "M_list": [0],  # M > 0 not yet implemented.
    "q_mode": "smooth",  # Only used if model='squashed'. Options: 'uniform' or 'smooth'.
}

# EXAMPLE SCAN DICTIONARY. ONE OR MULTIPLE PARAMETERS CAN BE SET TO A LIST OR ARRAY TO SCAN OVER.
# SCANABLE PARAMETERS: r1, M200, c, q0, gamma
# LIST/ARRAY INPUTS WILL GENERATE A GRID OVER ALL COMBINATIONS OF THE INPUT VALUES.
# run_dictionary = {
#     "model": "squashed", 
#     "r1": [10,20,30], 
#     "M200": 1e12, 
#     "c": 10.0,  
#     "q0": 1.0, 
#     "halo_type": "NFW",
#     "gamma": None, 
#     "Phi_b": None, 
#     "AC_prescription": None,
#     "Gnedin_params": (1.6, 0.8,), 
#     "save_profile": True, 
#     "save_dir": "data/",  
#     "verbose": False,  
#     "L_list": [0],  
#     "M_list": [0],  
#     "q_mode": "smooth"
# }

###################### FUNCTION CONFIGURATION ######################
# Baryon potential function
Md = 6e10  # Msun
a = 3.0  # kpc
b = 0.28  # kpc

Phi_b = Phi_MN(Md, a, b)  # Miyamoto-Nagai potential function

# ~~~~~~~~~~~~~~update the run dictionary~~~~~~~~~~~~~~ #
# run_dictionary["Phi_b"] = Phi_b # uncomment to include baryon function

# CDM halo shape function
q0 = 0.8  # Dimensionless. Outer halo shape.
r200 = 200  # kpc. Virial radius.
alpha = 0.2  # Dimensionless. Shape parameter.

# Non-constant CDM halo shape function
def q_cdm(q0, r200=200, alpha=0.3):
    return lambda r: q0 * (r / r200) ** alpha

# ~~~~~~~~~~~~~~update the run dictionary~~~~~~~~~~~~~~ #
# run_dictionary["q0"] = q_cdm(q0, r200, alpha) # uncomment to include non-constant halo shape function
