
from environment import *
from pso import *
import time

if __name__ == "__main__":
    problem = load_problem("./scenarios/scenario4.txt")
    
    def fitness(path: Path) -> float:
        penalty = 10000.
        #return  (path.nb_pair_collision(problem.obstacles),path.length())
        return path.nb_pair_collision(problem.obstacles) * penalty + path.length()
    
    # overall_best = 100000000
    # for T in range(100, 1000, 100):
    #     for beta in range(5, 100, 2):
    #         print(f"{T} and {beta} #########################################")
    #         best_f, _ = particle_swarm_optimization(problem,fitness,max_iter=100,S = 500,random_restart=True, simulated_annealing=True, beta = beta/100,T0=T)
    #         if best_f < overall_best:
    #             overall_best = best_f

    # best_f, g = particle_swarm_optimization(problem,fitness, random_restart=True, simulated_annealing=True, T0=10000,beta=0.8,dim_learning=True, nb_update_before_dim=20)
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
    # for p in g.points:
    #     print(p)
    # display_environment(problem, path=g)

