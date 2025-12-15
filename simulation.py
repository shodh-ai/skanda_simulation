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
    sim_conf = config["simulation"]
    param = pybamm.ParameterValues("Chen2020")

    common_updates = {
        "Ambient temperature [K]": sim_conf["temperature_K"],
        "Initial temperature [K]": sim_conf["temperature_K"],
        "Reference temperature [K]": sim_conf["temperature_K"],
        "Separator thickness [m]": sim_conf["separator_thickness_m"],
        "Negative current collector thickness [m]": sim_conf["cc_thickness_neg_m"],
        "Positive current collector thickness [m]": sim_conf["cc_thickness_pos_m"],
    }

    eps = transport_results["porosity_measured"]
    tau = transport_results["tau_factor"]

    if config["transport"]["bruggeman_from_tau"]:
        b_val = fit_bruggeman_b(eps, tau)
    else:
        b_val = 1.5

    comp = sim_conf["component"]

    if comp == "separator":
        common_updates.update(
            {
                "Separator porosity": eps,
                "Separator Bruggeman coefficient (electrolyte)": b_val,
            }
        )
    elif comp == "negative":
        common_updates.update(
            {
                "Negative electrode porosity": eps,
                "Negative electrode thickness [m]": sim_conf["electrode_thickness_m"],
                "Negative electrode Bruggeman coefficient (electrolyte)": b_val,
            }
        )
    elif comp == "positive":
        common_updates.update(
            {
                "Positive electrode porosity": eps,
                "Positive electrode thickness [m]": sim_conf["electrode_thickness_m"],
                "Positive electrode Bruggeman coefficient (electrolyte)": b_val,
            }
        )

    param.update(common_updates)
    base_steps = sim_conf["experiment"]
    n_cycles = sim_conf["cycles"]

    experiment = pybamm.Experiment(base_steps * n_cycles)

    if sim_conf.get("model") == "DFN":
        model = pybamm.lithium_ion.DFN()
    else:
        model = pybamm.lithium_ion.SPM()

    try:
        sim = pybamm.Simulation(model, parameter_values=param, experiment=experiment)
        solution = sim.solve()

        t = solution["Time [s]"].entries
        v = solution["Terminal voltage [V]"].entries
        i = solution["Current [A]"].entries

        indices = np.linspace(0, len(t) - 1, 500, dtype=int)
        return [[float(t[k]), float(v[k]), float(i[k])] for k in indices]

    except Exception as e:
        print(f"PyBaMM Error: {e}")
        return []
