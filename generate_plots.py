import os
import numpy as np
from utils import gen_tasks, plot_solution
from main import run_mmoea_dl
from baselines import run_mmode_cscd, run_immode_sa, run_ss_mopso, run_dsc_moagde

# ------------------------------------
# Settings
# ------------------------------------
TASK_SIZES = [20, 30, 40, 50]
DISTRIBUTIONS = ["uniform", "nonuniform"]
N_RUNS = 1   # Only 1 run per case for plotting
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ------------------------------------
# Algorithm List
# ------------------------------------
ALGORITHMS = {
    "MMOEA-DL": run_mmoea_dl,
    "MMODE-CSCD": run_mmode_cscd,
    "IMMODE-SA": run_immode_sa,
    "SS-MOPSO": run_ss_mopso,
    "DSC-MOAGDE": run_dsc_moagde
}

# ------------------------------------
# Main Script
# ------------------------------------
if __name__ == "__main__":
    for n_tasks in TASK_SIZES:
        for dist in DISTRIBUTIONS:
            clustered = (dist == "nonuniform")
            points = gen_tasks(n_tasks, clustered=clustered, seed=42)
            depots = [np.array([0,0]), np.array([100,0]), np.array([50,100])]

            for algo_name, algo_func in ALGORITHMS.items():
                print(f"[INFO] Running {algo_name} for {n_tasks} tasks ({dist})...")

                # Run once per case to get a solution
                sols = algo_func(points, depots, pop=80, gens=40)
                best_solution = min(sols, key=lambda s: sum(s["objs"]))

                # Save route plot
                plot_file = f"{PLOT_DIR}/{algo_name}_{n_tasks}_{dist}.png"
                plot_solution(points, depots, best_solution["tours"],
                              title=f"{algo_name} | Tasks={n_tasks}, Dist={dist}",
                              output_file=plot_file)

    print(f"\n[INFO] All route plots saved in {PLOT_DIR}/")
