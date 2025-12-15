import os
import yaml
import pandas as pd
import json
import sys
from tqdm import tqdm
import numpy as np

import structure
import transport
import simulation


def load_config(path="config.yaml"):
    """
    Load the YAML configuration file.
    """
    if not os.path.exists(path):
        print(f"Error: Configuration file '{path}' not found.")
        sys.exit(1)

    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    out_dir = config["general"]["output_dir"]
    img_dir = os.path.join(out_dir, "tiff_stacks")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    with open(os.path.join(out_dir, "run_config.yaml"), "w") as f:
        yaml.dump(config, f)

    num_samples = config["general"]["num_samples"]
    base_seed = config["general"]["base_seed"]
    rng = np.random.default_rng(base_seed)

    trans_conf = config["transport"]
    min_eps = trans_conf.get("min_epsilon", 0.3)
    max_eps = trans_conf.get("max_epsilon", 0.6)

    results_list = []

    print(f"--- Starting Battery Simulation Pipeline ---")
    print(f"Samples: {num_samples}")
    print(f"Output: {out_dir}")
    print(f"Porosity Range: {min_eps} - {max_eps}")
    print(f"--------------------------------------------")

    for i in tqdm(range(num_samples)):
        current_target_porosity = rng.uniform(min_eps, max_eps)
        config["structure"]["target_porosity"] = current_target_porosity

        run_data = {
            "id": i,
            "seed_used": config["general"]["base_seed"] + i,
            "input_psd": config["structure"]["psd_power"],
            "input_porosity": current_target_porosity,
            "voxel_size_um": config["structure"]["voxel_size_um"],
        }

        try:
            file_path, binary_vol = structure.generate_structure(i, config, img_dir)
            run_data["tiff_path"] = file_path

            trans_props = transport.calculate_properties(vol_data=binary_vol)
            run_data.update(trans_props)

            if "error" not in trans_props:
                sim_data = simulation.run_battery_simulation(trans_props, config)
                if sim_data:
                    run_data["charging_data"] = json.dumps(sim_data)
                    run_data["sim_status"] = "success"
                else:
                    run_data["sim_status"] = "failed_solver"
                    run_data["charging_data"] = "[]"
            else:
                run_data["sim_status"] = "skipped_geometry"
                run_data["charging_data"] = "[]"

        except Exception as e:
            print(f"Error in run {i}: {e}")
            run_data["sim_status"] = "failed"
            run_data["error_msg"] = str(e)

        results_list.append(run_data)

        if (i + 1) % max(1, num_samples // 10) == 0:
            df_partial = pd.DataFrame(results_list)
            df_partial.to_csv(os.path.join(out_dir, "results_partial.csv"), index=False)

    df = pd.DataFrame(results_list)
    final_csv = os.path.join(out_dir, "final_results.csv")
    df.to_csv(final_csv, index=False)

    print(f"Pipeline complete. Results saved to {final_csv}. {len(df)} records saved.")


if __name__ == "__main__":
    main()
