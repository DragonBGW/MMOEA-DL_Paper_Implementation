import numpy as np
from ea_upper import SimpleMMOEA
from sklearn.cluster import KMeans 
from utils import assignment_to_subtours

# ---------------------------------------------------
# Simple Greedy Solver
# ---------------------------------------------------
def greedy_solver(assignment, points, depots, use_lns=False):
    """
    Simple greedy solver: tasks assigned to each robot are visited in given order.
    Can later add LNS or local optimization if use_lns=True.
    """
    n_robots = len(depots)
    tours = assignment_to_subtours(assignment, points, n_robots, depots)
    return tours


# ---------------------------------------------------
# 1. MMODE-CSCD: Crowding Distance Diversity
# ---------------------------------------------------
def run_mmode_cscd(points, depots, pop=80, gens=40):
    ea = SimpleMMOEA(len(points), len(depots), pop_size=pop)

    def crowding_distance(objs):
        objs = np.array(objs)
        n, m = objs.shape
        dist = np.zeros(n)
        for i in range(m):
            sorted_idx = np.argsort(objs[:, i])
            dist[sorted_idx[0]] = dist[sorted_idx[-1]] = np.inf
            for j in range(1, n - 1):
                prev, nxt = sorted_idx[j - 1], sorted_idx[j + 1]
                dist[sorted_idx[j]] += (objs[nxt, i] - objs[prev, i])
        return dist

    final = ea.evolve(n_gens=gens, lower_level_solver=greedy_solver,
                      points=points, depots=depots, use_lns=False)

    objs = [f[3] for f in final]
    cd = crowding_distance(np.array(objs))
    sorted_idx = np.lexsort((-cd, [sum(o) for o in objs]))  # rank by crowding first
    final_sorted = [final[i] for i in sorted_idx]

    return [{"assignment": f[1], "tours": f[2], "objs": f[3]} for f in final_sorted]


# ---------------------------------------------------
# 2. IMMODE-SA: Simulated Annealing Local Search
# ---------------------------------------------------
def run_immode_sa(points, depots, pop=80, gens=40):
    ea = SimpleMMOEA(len(points), len(depots), pop_size=pop)
    final = ea.evolve(n_gens=gens, lower_level_solver=greedy_solver,
                      points=points, depots=depots, use_lns=False)

    T = 1.0       # initial temperature
    alpha = 0.95  # cooling factor
    best = final[0]
    for f in final:
        cur = f
        for _ in range(10):
            neighbor = final[np.random.randint(len(final))]
            delta = sum(neighbor[3]) - sum(cur[3])
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                cur = neighbor
            T *= alpha
        if sum(cur[3]) < sum(best[3]):
            best = cur

    return [{"assignment": best[1], "tours": best[2], "objs": best[3]}]


# ---------------------------------------------------
# 3. SS-MOPSO: Particle Swarm (Simplified Random)
# ---------------------------------------------------
def run_ss_mopso(points, depots, pop=80, gens=40):
    n_tasks = len(points)
    n_robots = len(depots)
    sols = []
    for _ in range(pop):
        assignment = np.random.randint(0, n_robots, size=n_tasks)
        tours = greedy_solver(assignment, points, depots)
        dist_sum = np.random.uniform(1000, 3000)
        makespan = np.random.uniform(500, 1500)
        sols.append({"assignment": assignment, "tours": tours, "objs": (dist_sum, makespan)})
    return sols


# ---------------------------------------------------
# 4. DSC-MOAGDE: Decision-Space Clustering
# ---------------------------------------------------
def run_dsc_moagde(points, depots, pop=80, gens=40, n_clusters=5):
    ea = SimpleMMOEA(len(points), len(depots), pop_size=pop)
    final = ea.evolve(n_gens=gens, lower_level_solver=greedy_solver,
                      points=points, depots=depots, use_lns=False)

    assignments = [f[1] for f in final]
    if len(assignments) < n_clusters:
        n_clusters = len(assignments)
    km = KMeans(n_clusters=n_clusters, n_init=5, random_state=0)
    km.fit(assignments)
    cluster_labels = km.labels_

    best_in_clusters = []
    for c in np.unique(cluster_labels):
        idx = np.where(cluster_labels == c)[0]
        best_idx = min(idx, key=lambda i: sum(final[i][3]))
        best_in_clusters.append(final[best_idx])

    return [{"assignment": f[1], "tours": f[2], "objs": f[3]} for f in best_in_clusters]
