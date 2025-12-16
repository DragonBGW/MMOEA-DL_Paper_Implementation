import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, friedmanchisquare
from pymoo.indicators.hv import HV
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

from utils import gen_tasks
from main import run_mmoea_dl   # Our MMOEA-DL method

# ------------------------------------
# Settings
# ------------------------------------
TASK_SIZES = [20, 30, 40, 50]
DISTRIBUTIONS = ["uniform", "nonuniform"]
N_RUNS = 10      # Use 22 runs for final paper-level results
REF_POINT = [5000, 3000]
RESULTS_FILE = "multi_algo_results.csv"
PLOT_FILE = "hv_boxplots.png"

# ------------------------------------
# Placeholder for Baselines
# ------------------------------------
def baseline_random(points, depots, pop=80, gens=40):
    """
    Dummy baseline: Random allocations, no EA, no DRL
    Returns final solutions in same format as run_mmoea_dl
    """
    n_tasks = len(points)
    n_robots = len(depots)
    sols = []
    for _ in range(10):  # produce random solutions
        assignment = np.random.randint(0, n_robots, size=n_tasks)
        tours = [[] for _ in range(n_robots)]
        for r in range(n_robots):
            idxs = np.where(assignment == r)[0]
            tours[r] = idxs.tolist()
        # objectives: dummy cost = sum of distances (not optimized)
        dist_sum = np.random.uniform(1000, 3000)
        makespan = np.random.uniform(500, 1500)
        sols.append({"assignment": assignment, "tours": tours, "objs": (dist_sum, makespan)})
    return sols

# Add baselines in a dict
ALGORITHMS = {
    "MMOEA-DL": run_mmoea_dl,
    "Random": baseline_random
    # Add more baselines here
}

# ------------------------------------
# Metric functions
# ------------------------------------
def compute_hv(objs):
    if len(objs) == 0:
        return 0.0
    objs = np.array(objs)
    hv = HV(ref_point=np.array(REF_POINT))
    return hv(objs)

def compute_psp(all_solutions, tol=1e-3):
    objs = np.array([s["objs"] for s in all_solutions])
    allocs = [tuple(s["assignment"].tolist()) for s in all_solutions]
    psp_count = 0
    for i in range(len(objs)):
        for j in range(i+1, len(objs)):
            if np.allclose(objs[i], objs[j], atol=tol) and allocs[i] != allocs[j]:
                psp_count += 1
    return psp_count

# ------------------------------------
# Main Experiment
# ------------------------------------
def run_multi_experiments():
    all_results = []
    hv_data = {algo: [] for algo in ALGORITHMS.keys()}
    all_solutions = {algo: [] for algo in ALGORITHMS.keys()}  # NEW

    for n_tasks in TASK_SIZES:
        for dist in DISTRIBUTIONS:
            print(f"\n=== {n_tasks} tasks | {dist} ===")
            clustered = (dist == "nonuniform")
            points = gen_tasks(n_tasks, clustered=clustered, seed=42)
            depots = [np.array([0,0]), np.array([100,0]), np.array([50,100])]

            for algo_name, algo_func in ALGORITHMS.items():
                run_hvs = []
                run_times = []
                run_solutions = []

                for seed in range(N_RUNS):
                    np.random.seed(seed)
                    start = time.time()
                    sols = algo_func(points, depots, pop=80, gens=40)
                    run_times.append(time.time() - start)
                    run_solutions.extend(sols)
                    run_hvs.append(compute_hv([s["objs"] for s in sols]))

                hv_data[algo_name].extend(run_hvs)
                all_solutions[algo_name].extend(run_solutions)  # NEW
                all_results.append({
                    "tasks": n_tasks,
                    "dist": dist,
                    "algo": algo_name,
                    "HV_mean": np.mean(run_hvs),
                    "HV_std": np.std(run_hvs),
                    "PSP": compute_psp(run_solutions),
                    "SR": np.mean([h >= 0.9*max(run_hvs) for h in run_hvs]),
                    "AFE": N_RUNS,
                    "AST": np.mean(run_times)
                })

    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_FILE, index=False)
    print(f"\nResults saved to {RESULTS_FILE}")
    return df, hv_data, all_solutions


# ------------------------------------
# Plots & Stats
# ------------------------------------
def plot_hv_boxplots(hv_data):
    fig, ax = plt.subplots(figsize=(8,6))
    ax.boxplot([hv_data[a] for a in hv_data.keys()], labels=hv_data.keys())
    ax.set_title("HV Comparison across Algorithms")
    ax.set_ylabel("Hypervolume")
    plt.savefig(PLOT_FILE)
    print(f"HV boxplot saved to {PLOT_FILE}")

def run_stats(hv_data):
    data = [hv_data[a] for a in hv_data.keys()]
    algos = list(hv_data.keys())
    stat, p = friedmanchisquare(*data)
    print("\nFriedman test: stat=%.4f p=%.4f" % (stat, p))
    for i in range(len(algos)):
        for j in range(i+1, len(algos)):
            s, p = wilcoxon(hv_data[algos[i]], hv_data[algos[j]])
            print(f"Wilcoxon {algos[i]} vs {algos[j]}: p={p:.4f}")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ---------------------------------------------------
# 1. Paper-Style Table (LaTeX-ready)
# ---------------------------------------------------
def generate_paper_table(df, metrics=["HV_mean","HV_std","PSP","SR","AFE","AST"], output_file="results_table.tex"):
    # Pivot table: rows = algorithm, cols = metrics averaged over all tasks & dists
    summary = df.groupby("algo")[metrics].mean().round(3)
    
    # Save LaTeX table
    latex_str = summary.to_latex(index=True, caption="Performance Comparison of Algorithms", label="tab:results")
    with open(output_file, "w") as f:
        f.write(latex_str)
    print(f"[INFO] Paper-style LaTeX table saved as {output_file}")
    return summary

# ---------------------------------------------------
# 2. Pareto Front Plots
# ---------------------------------------------------
def plot_pareto_fronts(all_solutions, algos, output_file="pareto_fronts.png"):
    plt.figure(figsize=(8,6))
    
    for algo, sols in all_solutions.items():
        objs = np.array([s["objs"] for s in sols])
        if len(objs) == 0: continue
        plt.scatter(objs[:,0], objs[:,1], label=algo, alpha=0.6)
    
    plt.xlabel("Objective 1: Total Distance")
    plt.ylabel("Objective 2: Makespan")
    plt.title("Pareto Front Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"[INFO] Pareto front plot saved as {output_file}")
    plt.close()

# ---------------------------------------------------
# ---------------------------------------------------
# 4. Wilcoxon Pairwise Statistical Tests
# ---------------------------------------------------
def generate_wilcoxon_table(hv_data, output_file="wilcoxon_table.tex"):
    algos = list(hv_data.keys())
    n = len(algos)
    
    # Prepare empty matrix for p-values
    p_matrix = np.ones((n, n))
    
    # Pairwise Wilcoxon tests
    for i in range(n):
        for j in range(i+1, n):
            stat, p = wilcoxon(hv_data[algos[i]], hv_data[algos[j]])
            p_matrix[i, j] = p
            p_matrix[j, i] = p  # symmetric

    # Build DataFrame for easy LaTeX output
    df_p = pd.DataFrame(p_matrix, index=algos, columns=algos)
    latex_str = df_p.to_latex(index=True, float_format="%.4f", caption="Wilcoxon Pairwise p-values", label="tab:wilcoxon")
    
    with open(output_file, "w") as f:
        f.write(latex_str)
    
    print(f"[INFO] Wilcoxon pairwise p-values saved as {output_file}")
    return df_p


# 3. Auto Summary + Plots after Experiments
# ---------------------------------------------------
def post_experiment_analysis(df, all_solutions):
    # Paper-style table
    table_summary = generate_paper_table(df)
    print("\n[SUMMARY TABLE]\n", table_summary)

    # Pareto front plots
    plot_pareto_fronts(all_solutions, df["algo"].unique())

# ------------------------------------
if __name__ == "__main__":
    df, hv_data, all_solutions = run_multi_experiments()
    plot_hv_boxplots(hv_data)
    run_stats(hv_data)
    post_experiment_analysis(df, all_solutions)   # NEW
    generate_wilcoxon_table(hv_data)
