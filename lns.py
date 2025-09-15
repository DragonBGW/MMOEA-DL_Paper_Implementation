# lns.py
import numpy as np
from utils import compute_tour_length

def lns_repair(tour, points, depot, remove_k=3):
    """
    tour: list of indices (current tour)
    remove_k: how many nodes to remove and reinsert greedily
    """
    if len(tour)<=2:
        return tour
    n = len(tour)
    remove_k = min(remove_k, n)
    idxs = list(range(n))
    # remove contiguous block or random nodes
    start = np.random.randint(0, n)
    rem = [(start + i) % n for i in range(remove_k)]
    remaining = [tour[i] for i in idxs if i not in rem]
    removed = [tour[i] for i in rem]
    # greedy insert removed nodes one-by-one into best position
    for node in removed:
        best_pos = None
        best_len = float('inf')
        for pos in range(len(remaining)+1):
            cand = remaining[:pos] + [node] + remaining[pos:]
            L = compute_tour_length(cand, points, depot)
            if L < best_len:
                best_len = L
                best_pos = pos
        remaining = remaining[:best_pos] + [node] + remaining[best_pos:]
    return remaining
