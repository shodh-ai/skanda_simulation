import pybamm
import numpy as np
import math


def fit_bruggeman_b(epsilon, tau):
    """Fits b such that epsilon^b = epsilon / tau"""
    if epsilon <= 0 or epsilon >= 1 or tau <= 0:
        return 1.5
    return math.log(epsilon / tau) / math.log(epsilon)


def run_battery_simulation(transport_results, config):
    """
    Runs a PyBaMM DFN simulation based on transport results.
    Returns: Compressed time/voltage/current data.
    """

    eps = transport_results["porosity_measured"]
    tau = transport_results["tau_factor"]
    b_val = fit_bruggeman_b(eps, tau)

    param = pybamm.ParameterValues("Chen2020")

    comp = config["simulation"]["component"]
    updates = {}

    if comp == "separator":
        updates = {
            "Separator porosity": eps,
            "Separator Bruggeman coefficient (electrolyte)": b_val,
        }
    elif comp == "negative":
        updates = {
            "Negative electrode porosity": eps,
            "Negative electrode Bruggeman coefficient (electrolyte)": b_val,
        }

    param.update(updates)

    c_dis = config["simulation"]["c_rate_discharge"]
    c_chg = config["simulation"]["c_rate_charge"]
    v_min = config["simulation"]["voltage_min"]
    v_max = config["simulation"]["voltage_max"]
    n_cycles = config["simulation"]["cycles"]

    cycle_steps = (
        f"Discharge at {c_dis}C until {v_min}V",
        "Rest for 15 minutes",
        f"Charge at {c_chg}C until {v_max}V",
        f"Hold at {v_max}V until C/20",
        "Rest for 15 minutes",
    )

    experiment = pybamm.Experiment([cycle_steps] * n_cycles)

    model = pybamm.lithium_ion.DFN()
    sim = pybamm.Simulation(model, parameter_values=param, experiment=experiment)

    try:
        solution = sim.solve()
    except Exception as e:
        print(f"Simulation failed: {e}")
        return []

    t = solution["Time [s]"].entries
    v = solution["Terminal voltage [V]"].entries
    i = solution["Current [A]"].entries

    indices = np.linspace(0, len(t) - 1, 500, dtype=int)

    sim_data = []
    for idx in indices:
        sim_data.append([float(t[idx]), float(v[idx]), float(i[idx])])

    return sim_data
