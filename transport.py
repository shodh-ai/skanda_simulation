import taufactor as tau
import numpy as np
import tifffile


def calculate_properties(vol_data=None, file_path=None):
    """
    Calculates Porosity, Tortuosity, and Diffusivity.
    Can accept either raw numpy volume or a file path.
    """

    if vol_data is None:
        if file_path is None:
            raise ValueError("Must provide vol_data or file_path")
        vol_data = tifffile.imread(file_path)
        vol_data = (vol_data > 0).astype(np.uint8)

    phi = float(vol_data.mean())

    solver = tau.Solver(vol_data)
    solver.solve(verbose=False)

    results = {
        "porosity_measured": phi,
        "tau_factor": float(solver.tau),
        "D_eff": float(solver.D_eff),
    }

    return results
