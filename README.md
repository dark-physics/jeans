# jeans

**jeans** is a Python package for computing dark matter halo profiles with baryons used in the paper [_Jeans Model for the Shapes of Self-interacting Dark Matter Halos_](link). The jeans package can be used to model collisioinless dark matter (CDM) and self-interacting dark matter (SIDM) in one dimension assuming spherical symmetry and two dimensions assuming axisymmetry.

This package is intended for researchers and students working in astrophysics and cosmology, especially those interested in the structure and morphology of SIDM halos.

## Features

The jeans model incorporates the effects on the dark matter halo due to the baryon distributions, axis ratios, and adiabatic contraction for a wide range of halo masses. Currently the jeans model is implemented in one- and two-dimensions, with further work intended to extend two the full three-dimensional model. A shortlist of the inputs and outputs of the model are listed below:

Inputs

- Matching radius: r1
- Virial mass: M200
- Concentration: c
- Axis Ratio (constant or function): q0
- Baryon Potential: Phi_b
- CDM Halo Type: NFW or Einasto
- Adiabatic Contraction: Cautun or Gnedin

Outputs

- Densities (one- and two-dimensions)
- Rotation Curves (Total, DM, and Baryons)
- Axis Ratios (c/a or q(r))
- Self-interaction Cross Section
- Mass Enclosed Profiles
- Gravitational Potentials

_See profile notebook in examples for more._

## Requirements

- Python 3.9+
- numpy
- scipy
- h5py (for EAGLE-50 data)
- matplotlib (for plotting)

All dependencies are listed in `requirements.txt` and will be installed automatically.

## Installation

Clone the repository and install:

```bash
git clone https://github.com/AdamSmithOrlik/nonspherical-sidm-jeans-model.git
cd nonspherical-sidm-jeans-model
pip install .
```

## Usage Example

### Use in Jupyter Notebooks

```python
import jeans

# Example: create a nonspherical SIDM squashed halo with baryons and adiabatic contraction
# r1: matching radius
# M200: mass enclosed within 200 kpc
# c: concentration within 200 kpc
# q0: shape of outer halo at matching radius
# Phi_b: baryon potential as a function of (r, th)
# AC_prescription: prescription for adiabatic contraction

# Generate profile
profile = jeans.squashed(r1, M200, c, q0=q0, Phi_b=Phi_b, **{'AC_prescription':'Cautun'})

# Generate the denisty profile
r = np.logspace(-1, 3, 100)  # kpc
rho_sph = profile.rho_sph_avg(r)

# Plot the density profile
import matplotlib.pyplot as plt
plt.loglog(r, rho_sph)
plt.xlabel('Radius [kpc]')
plt.ylabel('Density [Msun/kpc^3]')
plt.show()
```

_See the `examples/` folder for Jupyter notebooks and scripts demonstrating more advanced usage._

### Use in Command Line

In the `scripts/` folder you may update the configuration dictionaries in the file `run_dict.py`. To generate profiles run `python run.py` in the command line. Profiles will be saved in the `data/` folder and can be loaded with the jeans method `jeans.load(<filename>)`.

The `run_dict.py` configuration file allows the user to run a single profile at a time, or run a scan of input parameters. The user may also wish to define their own bayon potential and/or outer halo shape profile, which they may do by updating the dictionaries at the end of the file.

---

## License

This project is licensed under the terms of the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this package in your research, please cite the associated paper (see [this paper](link)).

---

## Contact

For questions or support, please open an issue on GitHub or contact the maintainer at [asorlik@yorku.ca].

---

## Affiliation

This project is developed and maintained at the Department of Physics and Astronomy, York University, Toronto, Canada.
