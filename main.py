
from environment import *
from pso import *

if __name__ == "__main__":
    problem = load_problem("./scenarios/scenario4.txt")
    
    g = particle_swarm_optimization(problem, S = 200)
    # for p in g.points:
    #     print(p)
    # display_environment(problem, path=g)

