import numpy as np
import pandas as pd
import time
from scipy.stats import wilcoxon, friedmanchisquare
from pymoo.indicators.hv import HV   # ✅ correct for pymoo ≥ 0.6
from utils import gen_tasks
from main import demo_run  # assumes demo_run() returns final solutions with objectives

# ---------------------------
# Settings
# ---------------------------
TASK_SIZES = [20, 30, 40, 50]
DISTRIBUTIONS = ["uniform", "nonuniform"]
N_RUNS = 22
REF_POINT = [5000, 3000]  # Reference point for HV; adjust for your problem scale
RESULTS_CSV = "results_summary.csv"

# ---------------------------
# Metric functions
# ---------------------------
def compute_hv(objs):
    """Compute hypervolume for given set of objective vectors."""
    if len(objs) == 0:
        return 0.0
    objs = np.array(objs)
    hv = HV(ref_point=np.array(REF_POINT))
    return hv(objs)

def compute_psp(all_solutions, tol=1e-3):
    """
    Pareto Set Proximity: counts distinct solutions with same objective values (within tol).
    Here we approximate: if two solutions have objs within tol but diff allocations => +1 PSP.
    """
    objs = np.array([s["objs"] for s in all_solutions])
    allocs = [tuple(s["assignment"].tolist()) for s in all_solutions]
    psp_count = 0
    for i in range(len(objs)):
        for j in range(i+1, len(objs)):
            if np.allclose(objs[i], objs[j], atol=tol) and allocs[i] != allocs[j]:
                psp_count += 1
    return psp_count

def success_rate(hv_scores, hv_threshold):
    """Success if HV above some fraction of best HV."""
    best = max(hv_scores)
    return sum(h >= hv_threshold*best for h in hv_scores)/len(hv_scores)

# ---------------------------
# Experiment runner
# ---------------------------
def run_experiments():
    results = []
    for n_tasks in TASK_SIZES:
        for dist in DISTRIBUTIONS:
            print(f"\n=== Instance: {n_tasks} tasks, {dist} ===")
            # Generate tasks
            clustered = (dist == "nonuniform")
            points = gen_tasks(n_tasks, clustered=clustered, seed=42)
            depots = [np.array([0,0]), np.array([100,0]), np.array([50,100])]
            run_objs = []
            run_solutions = []
            run_times = []
            for seed in range(N_RUNS):
                np.random.seed(seed)
                start = time.time()
                final_solutions = demo_run()  # returns list of dicts: {assignment, tours, objs}
                elapsed = time.time() - start
                run_times.append(elapsed)
                # Collect objectives from final solutions
                all_objs = [sol["objs"] for sol in final_solutions]
                run_objs.append(all_objs)
                run_solutions.extend(final_solutions)

            # Compute metrics
            hv_scores = [compute_hv(objs) for objs in run_objs]
            psp_score = compute_psp(run_solutions)
            sr = success_rate(hv_scores, hv_threshold=0.9)
            afe = N_RUNS  # here each run = one function evaluation budget
            ast = np.mean(run_times)

            results.append({
                "tasks": n_tasks,
                "dist": dist,
                "HV_mean": np.mean(hv_scores),
                "HV_std": np.std(hv_scores),
                "PSP": psp_score,
                "SR": sr,
                "AFE": afe,
                "AST": ast
            })

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nResults saved to {RESULTS_CSV}")
    return df

# ---------------------------
# Statistical tests (example)
# ---------------------------
def run_stats(hv_matrix):
    """
    hv_matrix: dict algo_name -> list of HV scores across instances
    """
    algos = list(hv_matrix.keys())
    data = [hv_matrix[a] for a in algos]
    # Friedman test
    stat, p = friedmanchisquare(*data)
    print("\nFriedman test: stat=%.4f p=%.4f" % (stat, p))
    # Pairwise Wilcoxon
    for i in range(len(algos)):
        for j in range(i+1, len(algos)):
            s, p = wilcoxon(data[i], data[j])
            print(f"Wilcoxon {algos[i]} vs {algos[j]}: p={p:.4f}")

# ---------------------------
if __name__ == "__main__":
    df = run_experiments()
    print(df)
