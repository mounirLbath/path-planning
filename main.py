
from environment import *
from pso import *

if __name__ == "__main__":
    problem = load_problem("./scenarios/scenario3.txt")
    
    def fitness(path: Path) -> float:
        # penalty = 10000000.
        return  (path.nb_pair_collision(problem.obstacles), path.length())
    
    g = particle_swarm_optimization(problem,fitness,  max_iter=100,S = 500)
    # for p in g.points:
    #     print(p)
    # display_environment(problem, path=g)

