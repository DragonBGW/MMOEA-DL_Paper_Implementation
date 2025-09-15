# ea_upper.py
import numpy as np
from utils import assignment_to_subtours, compute_objectives

class SimpleMMOEA:
    """
    Simple multi-objective evolutionary algorithm for allocations.
    Genome: real-valued vector length n_tasks with values in [0, n_robots).
    Rounding: int(genome[i]) % n_robots -> robot id
    DE/current-to-rand/1-inspired operators.
    """
    def __init__(self, n_tasks, n_robots, pop_size=100, F=0.5, Cr=0.9, rng=None):
        self.n_tasks = n_tasks
        self.n_robots = n_robots
        self.pop = pop_size
        self.F = F
        self.Cr = Cr
        self.rng = np.random.RandomState(42 if rng is None else rng)
        # initialize population
        self.population = self.rng.uniform(0, n_robots, size=(self.pop, n_tasks))
        self.archive = []  # store (genome, objectives, tours)
    
    def decode(self, genome):
        # map to integer assignment
        assn = (np.floor(genome).astype(int) % self.n_robots)
        return assn

    def evaluate_population(self, pop, lower_level_solver, points, depots, use_lns=False):
        results = []
        for genome in pop:
            assn = self.decode(genome)
            tours = lower_level_solver(assn, points, depots, use_lns=use_lns)
            objs = compute_objectives(assn, tours, points, depots)
            results.append((genome.copy(), assn, tours, objs))
        return results

    def tournament_select(self, ranks):
        # simple selection: pick best by Pareto dominance count or scalarization
        i = self.rng.randint(0, self.pop)
        j = self.rng.randint(0, self.pop)
        return i if ranks[i] < ranks[j] else j

    def pareto_ranks(self, objs_list):
        # objs_list: list of (obj1,obj2) tuples
        pop = len(objs_list)
        dominated_counts = [0]*pop
        for i in range(pop):
            for j in range(pop):
                if i==j: continue
                a = objs_list[i]; b = objs_list[j]
                if (b[0] <= a[0] and b[1] <= a[1]) and (b[0] < a[0] or b[1] < a[1]):
                    dominated_counts[i] += 1
        return dominated_counts

    def evolve(self, n_gens, lower_level_solver, points, depots, use_lns=False):
        # initial evaluation
        evals = self.evaluate_population(self.population, lower_level_solver, points, depots, use_lns)
        genomes = np.array([e[0] for e in evals])
        objs = [e[3] for e in evals]
        dominated = self.pareto_ranks(objs)
        for gen in range(n_gens):
            new_pop = genomes.copy()
            for i in range(self.pop):
                # DE mutation
                idxs = [x for x in range(self.pop) if x!=i]
                a,b,c = self.rng.choice(idxs, 3, replace=False)
                mutant = genomes[a] + self.F*(genomes[b]-genomes[c])
                # crossover
                cross = genomes[i].copy()
                mask = self.rng.rand(self.n_tasks) < self.Cr
                cross[mask] = mutant[mask]
                # clip
                cross = np.clip(cross, 0, self.n_robots-1e-6)
                # evaluate trial
                eval_trial = self.evaluate_population([cross], lower_level_solver, points, depots, use_lns)[0]
                trial_obj = eval_trial[3]
                # compare using Pareto: if trial dominates or better scalar, replace
                if self._is_better(trial_obj, objs[i]):
                    new_pop[i] = cross
                    objs[i] = trial_obj
            genomes = new_pop
            # reassign pop and ranks
            dominated = self.pareto_ranks(objs)
            print(f"[EA] gen {gen+1}/{n_gens} best dominated counts min={min(dominated)} mean={np.mean(dominated):.2f}")
        # final packaging
        res = self.evaluate_population(genomes, lower_level_solver, points, depots, use_lns)
        return res

    def _is_better(self, o1, o2):
        # prefer domination, else sum
        if (o1[0] <= o2[0] and o1[1] <= o2[1]) and (o1[0] < o2[0] or o1[1] < o2[1]):
            return True
        if (o2[0] <= o1[0] and o2[1] <= o1[1]) and (o2[0] < o1[0] or o2[1] < o1[1]):
            return False
        return (o1[0]+o1[1]) < (o2[0]+o2[1])
