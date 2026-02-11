
from environment import *
from pso import *
import time

if __name__ == "__main__":
    problem = load_problem("./scenarios/scenario4.txt")
    
    def fitness(path: Path) -> float:
        penalty = 10000.
        return path.nb_pair_collision(problem.obstacles) * penalty + path.length()
    
    t0 = time.time()

    best_f, g = particle_swarm_optimization(
        problem,
        fitness,
        max_iter=300,
        S=300,
        c1=2.0,
        c2=1.2,
        w = 0.8,
        random_restart=True,
        random_period=15,
        simulated_annealing=True,
        dim_learning=True,
        nb_update_before_dim=15
    )

    print(f"total time:{time.time()-t0}")

