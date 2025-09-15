# utils.py
import numpy as np
import math
from scipy.spatial.distance import cdist

def gen_tasks(n_tasks, bbox=(0,0,100,100), seed=None, clustered=False):
    rng = np.random.RandomState(seed)
    x0,y0,x1,y1 = bbox
    if clustered:
        centers = rng.uniform([x0,y0],[x1,y1], size=(max(1,n_tasks//10),2))
        points = []
        for i in range(n_tasks):
            c = centers[rng.randint(len(centers))]
            points.append(c + rng.normal(scale=5.0, size=2))
        pts = np.clip(np.array(points), [x0,y0],[x1,y1])
    else:
        pts = rng.uniform([x0,y0],[x1,y1], size=(n_tasks,2))
    return pts

def compute_tour_length(tour, points, depot):
    # tour: sequence of indices into points
    if len(tour)==0:
        return 0.0
    seq = points[np.array(tour)]
    # path: depot -> seq -> depot
    path = np.vstack([depot, seq, depot])
    d = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
    return float(d)

def assignment_to_subtours(assignment, points, n_robots, depots):
    """
    assignment: array len = n_tasks, values in [0..n_robots-1]
    returns dict robot_id -> list of task indices
    """
    subs = {r: [] for r in range(n_robots)}
    for i, r in enumerate(assignment):
        subs[r].append(i)
    tours = []
    for r in range(n_robots):
        tours.append(subs[r])
    return tours

def compute_objectives(assignment, tours, points, depots):
    # objectives: (1) sum of distances (total distance), (2) makespan (max tour length)
    n_robots = len(depots)
    dist_sum = 0.0
    makespan = 0.0
    for r in range(n_robots):
        d = compute_tour_length(tours[r], points, depots[r])
        dist_sum += d
        if d > makespan:
            makespan = d
    return dist_sum, makespan

def euclidean(a,b):
    return np.linalg.norm(a-b)
