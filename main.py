
from environment import *
from pso import *

if __name__ == "__main__":
    problem = load_problem("./scenarios/scenario3.txt")
    
    def fitness(path: Path) -> float:
        penalty = 3000.
        #return  (path.nb_pair_collision(problem.obstacles),path.length())
        return path.nb_pair_collision(problem.obstacles) * penalty + path.length()
    
    # overall_best = 100000000
    # for T in range(100, 1000, 100):
    #     for beta in range(5, 100, 2):
    #         print(f"{T} and {beta} #########################################")
    #         best_f, _ = particle_swarm_optimization(problem,fitness,max_iter=100,S = 500,random_restart=True, simulated_annealing=True, beta = beta/100,T0=T)
    #         if best_f < overall_best:
    #             overall_best = best_f

    best_f, g = particle_swarm_optimization(problem,fitness)
    
    
    # for p in g.points:
    #     print(p)
    # display_environment(problem, path=g)

