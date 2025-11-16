import os
import numpy as np
import jeans
import datetime as dt
import time as t
from itertools import product
import copy

from run_dict import run_dictionary as rd


def main(rd, filename=None):
    start = t.time()
    now_str = dt.datetime.now().strftime("%Y_%m_%d_%H_%M")

    # check is save directory exists, if not create it
    if rd["save_profile"]:
        os.makedirs(rd["save_dir"], exist_ok=True)

    ac_input_dict = {}
    if rd["AC_prescription"] is not None:
        ac_input_dict["AC_prescription"] = rd["AC_prescription"]
        if rd["AC_prescription"] == "Gnedin":
            ac_input_dict["Gnedin_params"] = rd["Gnedin_params"]
        else:
            print("Warning: AC_prescription not recognized. No adiabatic contraction will be applied.")

    if rd["model"] == "spherical":
        print("Running spherical model...")
        profile = jeans.spherical(
            rd["r1"],
            rd["M200"],
            rd["c"],
            Phi_b=rd["Phi_b"],
            halo_type=rd["halo_type"],
            gamma=rd["gamma"],
            verbose=rd["verbose"],
            **ac_input_dict,
        )
        if profile:
            print(f"Spherical profile generated successfully.")
        else:
            print("Spherical profile computation failed.")
            return None
    elif rd["model"] == "cdm":
        rd["r1"] = 0.0  # override r1 so that CDM halo is assumed

        print("Running CDM model...")
        profile = jeans.cdm(
            rd["r1"],
            rd["M200"],
            rd["c"],
            q0=rd["q0"],
            Phi_b=rd["Phi_b"],
            halo_type=rd["halo_type"],
            gamma=rd["gamma"],
            verbose=rd["verbose"],
            **ac_input_dict,
        )
        if profile:
            print(f"CDM profile generated successfully.")
        else:
            print("CDM profile computation failed.")
            return None

    elif rd["model"] == "squashed":
        print("Running squashed model...")
        profile = jeans.squashed(
            rd["r1"],
            rd["M200"],
            rd["c"],
            q0=rd["q0"],
            Phi_b=rd["Phi_b"],
            halo_type=rd["halo_type"],
            gamma=rd["gamma"],
            q_mode=rd["q_mode"],
            verbose=rd["verbose"],
            **ac_input_dict,
        )
        if profile:
            print(f"Squashed profile generated successfully.")
        else:
            print("Squashed profile computation failed.")
            return None
    elif rd["model"] == "isothermal":
        print("Running isothermal model...")
        profile = jeans.isothermal(
            rd["r1"],
            rd["M200"],
            rd["c"],
            q0=rd["q0"],
            Phi_b=rd["Phi_b"],
            halo_type=rd["halo_type"],
            gamma=rd["gamma"],
            L_list=rd["L_list"],
            M_list=rd["M_list"],
            verbose=rd["verbose"],
            **ac_input_dict,
        )
        if profile:
            print(f"Isothermal profile generated successfully.")
        else:
            print("Isothermal profile computation failed.")
            return None
    else:
        print("Error: model not recognized. Choose from 'spherical', 'cdm', 'squashed' or 'isothermal'.")
        return profile

    end = t.time()
    print(f"Time taken to generate profile: {end - start:.2f} seconds")

    if rd["save_profile"]:
        if rd["gamma"]:
            gamma_str = f"_gamma_{rd['gamma']:.2f}"
        else:
            gamma_str = ""
        if rd["AC_prescription"]:
            ac_str = f"_AC_{rd['AC_prescription']}"
            if rd["AC_prescription"] == "Gnedin":
                ac_str += f"_A_{rd['Gnedin_params'][0]}_w_{rd['Gnedin_params'][1]}"
        else:
            ac_str = ""
        file_identifier = f"{rd['model']}_r1_{rd['r1']:.1f}_logM200_{np.log10(rd['M200']):.1f}_c_{rd['c']}_{rd['halo_type']}{gamma_str}{ac_str}_{now_str}"

        if filename is None:
            filename = f"{file_identifier}.npz"
            filepath = os.path.join(rd["save_dir"], filename)
            profile.save(filepath)
            print(f"Profile saved to {filepath}")
        else:
            filename = f"{filename}_{file_identifier}.npz"
            filepath = os.path.join(rd["save_dir"], filename)
            profile.save(filepath)
            print(f"Profile saved to {filepath}")

    return None


def run_jeans_multi(rd, key="r1", values=[10]):
    for val in values:
        rd[key] = val
        print(f"Running model with {key}={val}")
        profile = main(rd)
        if profile:
            print(f"Model with {key}={val} completed successfully.\n")
        else:
            print(f"Model with {key}={val} failed.\n")
    return None


def expand_over_keys(d, scan_keys):
    """
    If any of scan_keys is a np.ndarray or list this function generates the cartesian product
    of all list values while keeping other keys fixed, enabling parameter scans.
    """
    # Separate fixed and varying pieces
    fixed = {k: v for k, v in d.items() if k not in scan_keys}
    vary = {k: d[k] for k in scan_keys if k in d}

    def to_list(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (list, tuple)) and not isinstance(x, (str, bytes)):
            return list(x)
        return [x]

    keys_vary = list(vary.keys())
    vals_vary = [to_list(vary[k]) for k in keys_vary]

    for combo in product(*vals_vary):
        new_d = copy.deepcopy(fixed)  # keep non-scan keys untouched
        new_d.update(zip(keys_vary, combo))  # add the varied keys
        yield new_d


if __name__ == "__main__":
    filename = None
    # filename = "MW"  # specify for a custom filename

    # logic for scanning over parameters
    scan_keys = ["r1", "M200", "c", "q0", "gamma"]

    if np.any([isinstance(rd[key], (np.ndarray, list)) for key in scan_keys]):
        varied_dicts = list(expand_over_keys(rd, scan_keys))
        num_dicts = len(varied_dicts)
        if num_dicts > 10:
            print(f"Warning: attempting to run {num_dicts} models. Would you like to continue? (y/n)")
            ans = input()
            if ans.lower() != "y":
                print("Exiting.")
                raise SystemExit

        for rd_i in varied_dicts:
            main(rd_i, filename=filename)
    else:
        main(rd, filename=filename)
