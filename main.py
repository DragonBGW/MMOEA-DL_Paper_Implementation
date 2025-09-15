# main.py
import numpy as np
import torch
from utils import gen_tasks, assignment_to_subtours
from pointer_net import PointerNet
from lns import lns_repair
from ea_upper import SimpleMMOEA

# Lower-level: takes assignment and returns tours per robot using pointer net
class LowerLevel:
    def __init__(self, actor: PointerNet, device='cpu'):
        self.actor = actor
        self.device = device

    def __call__(self, assignment, points, depots, use_lns=False):
        n_robots = len(depots)
        grouped = assignment_to_subtours(assignment, points, n_robots, depots)
        tours = [[] for _ in range(n_robots)]
        for r in range(n_robots):
            idxs = grouped[r]
            if len(idxs) == 0:
                tours[r] = []
                continue
            subpts = points[idxs].astype(np.float32)
            with torch.no_grad():
                inp = torch.from_numpy(subpts).unsqueeze(0)
                seqs, _ = self.actor(inp.to(self.device))
                seq = seqs[0].cpu().numpy().tolist()
            ordered = [idxs[i] for i in seq]
            if use_lns:
                ordered = lns_repair(ordered, points, depots[r],
                                     remove_k=max(1, len(ordered)//6))
            tours[r] = ordered
        return tours

def run_mmoea_dl(points, depots, pop=80, gens=40, device='cpu', use_lns=True):
    """
    Run MMOEA-DL on given tasks & depots.
    Returns list of dicts: {assignment, tours, objs}
    """
    actor = PointerNet(input_dim=2, hidden_dim=128)
    lower = LowerLevel(actor, device=device)
    ea = SimpleMMOEA(len(points), len(depots), pop_size=pop)
    final = ea.evolve(n_gens=gens, lower_level_solver=lower,
                      points=points, depots=depots, use_lns=use_lns)

    results = []
    for genome, assignment, tours, objs in final:
        results.append({
            "assignment": assignment,
            "tours": tours,
            "objs": objs
        })
    return results

def demo_run():
    """
    Default demo run: 30 tasks, 3 depots, 40 gens
    Used by evaluate.py for automatic evaluation
    """
    n_tasks = 30
    n_robots = 3
    points = gen_tasks(n_tasks, seed=123, clustered=True)
    depots = [np.array([0.0,0.0]), np.array([100.0,0.0]), np.array([50.0,100.0])]

    results = run_mmoea_dl(points, depots, pop=80, gens=40, device='cpu', use_lns=True)
    return results

if __name__ == '__main__':
    out = demo_run()
    print(f"Demo run completed. {len(out)} solutions found.")
    print("First few solutions:", out[:3])
