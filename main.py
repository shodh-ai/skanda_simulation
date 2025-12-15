import os
import yaml
import pandas as pd
import json
from tqdm import tqdm

import structure
import transport
import simulation


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    out_dir = config["general"]["output_dir"]
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    num_samples = config["general"]["num_samples"]

    results_list = []

    print(f"Starting pipeline for {num_samples} structures...")

    for i in tqdm(range(num_samples)):
        run_data = {
            "id": i,
            "seed_used": config["general"]["base_seed"] + i,
            "input_psd": config["structure"]["psd_power"],
            "input_porosity": config["structure"]["target_porosity"],
        }

        try:
            file_path, binary_vol = structure.generate_structure(i, config, img_dir)
            run_data["tiff_path"] = file_path

            trans_props = transport.calculate_properties(vol_data=binary_vol)
            run_data.update(trans_props)

            sim_data = simulation.run_battery_simulation(trans_props, config)

            run_data["charging_data"] = json.dumps(sim_data)
            run_data["sim_status"] = "success"

        except Exception as e:
            print(f"Error in run {i}: {e}")
            run_data["sim_status"] = "failed"
            run_data["error_msg"] = str(e)

        results_list.append(run_data)

        if i % 10 == 0:
            df_partial = pd.DataFrame(results_list)
            df_partial.to_csv(os.path.join(out_dir, "results_partial.csv"), index=False)

    df = pd.DataFrame(results_list)
    final_csv = os.path.join(out_dir, "final_results.csv")
    df.to_csv(final_csv, index=False)

    print(f"Pipeline complete. Results saved to {final_csv}")


if __name__ == "__main__":
    main()
