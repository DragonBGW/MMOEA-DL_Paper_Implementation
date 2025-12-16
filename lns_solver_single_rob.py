# lns_solver.py
import numpy as np

# 2-opt swap for route improvement
def two_opt_swap(route, i, k):
    """Perform 2-opt swap correctly using NumPy concatenation"""
    return np.concatenate([route[:i], route[i:k+1][::-1], route[k+1:]])

# Large Neighborhood Search (LNS) using 2-opt
def lns_improve_route(route, points, depot, iterations=50):
    best_route = route.copy()
    best_distance = compute_tour_length(best_route, points, depot)

    for _ in range(iterations):
        i, k = np.sort(np.random.choice(len(route), 2, replace=False))
        new_route = two_opt_swap(best_route, i, k)
        new_distance = compute_tour_length(new_route, points, depot)

        if new_distance < best_distance:
            best_route = new_route
            best_distance = new_distance

    return best_route, best_distance

# Helper: compute tour length (reuse for single/multi-robot)
def compute_tour_length(tour, points, depot):
    if len(tour) == 0:
        return 0.0
    seq = points[np.array(tour)]
    path = np.vstack([depot, seq, depot])
    return np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
