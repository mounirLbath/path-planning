
from environment import Problem, Point, Path, load_problem, display_environment
import random
import copy

problem = load_problem("./scenarios/scenario3.txt")

def particle_swarm_optimization(problem : Problem, max_iter: int = 500, S: int = 100,  c1: float = 1.5, c2: float = 1.5, w:float = 0.7) -> Path:
    """
    Returns the path found by the particle swarm optimization algorithm
    max_iter: maximum number of iterations before stopping the algorithm
    S: number of particles to simulate
    c1 and c2: acceleration coefficients
    w: inertia weight for velocity
    """
    def fitness(path: Path) -> float:
        # penalty = 10000000.
        return  (path.collision(problem.obstacles), path.length())

    nb_points_path = 10

    X = [Path([Point(random.random()*problem.xmax, random.random()*problem.ymax) for i in range(nb_points_path)], problem.start1, problem.goal1) for k in range(S)]
    print(X[0])
    
    V = [Path([Point(0,0) for i in range(nb_points_path)], problem.start1, problem.goal1) for k in range(S)]
    P = [copy.deepcopy(x) for x in X]
    g_index = min(range(S), key=lambda i: fitness(P[i])) # index of the particle with global best position
    best_f = fitness(P[g_index])
    display_environment(problem, path=P[g_index])

    for k in range(max_iter):
        for i in range(S): 
            r1 = random.random()
            r2 = random.random()
            V[i] = (V[i]*w + (P[i]-X[i])*c1*r1 + (P[g_index]-X[i])*c2*r2).clamp_norm(50)
            X[i] = (X[i] + V[i]).clamp(problem.xmax, problem.ymax)

            if fitness(X[i]) < fitness(P[i]):
                P[i] = copy.deepcopy(X[i])
        
        for i in range(S):
            curr_f = fitness(P[i])
            if  curr_f < best_f:
                g_index = i
                best_f = curr_f


        print(best_f)
        #display_environment(problem, path=p[g_index])

        # display_environment(problem, path=p[g_index])
    for p in P[g_index].points:
        print(p)
    display_environment(problem, path=P[g_index])
    return P[g_index]


g = particle_swarm_optimization(problem)
