
from environment import Problem, Point, Path, load_problem, display_environment
import random
import copy

random.seed(42)

def random_path(problem: Problem, nb_points_path:int) -> Path:
    sigma_sq = 200
    return Path([Point(random.gauss(0, sigma_sq)+(i+1)*problem.xmax/(nb_points_path+1), random.gauss(0, sigma_sq)+(i+1)*problem.ymax/(nb_points_path+1)).clamp(problem.xmax, problem.ymax)for i in range(nb_points_path)], problem.start1, problem.goal1)

def particle_swarm_optimization(problem : Problem, fitness, max_iter: int = 30, S: int = 20,  c1: float = 1.5, c2: float = 1.5, w:float = 0.7, random_period: int = 20) -> Path:
    """
    Returns the path found by the particle swarm optimization algorithm
    max_iter: maximum number of iterations before stopping the algorithm
    S: number of particles to simulate
    c1 and c2: acceleration coefficients
    w: inertia weight for velocity
    """

    nb_points_path = 10

    # X = [Path([Point(random.random()*problem.xmax, random.random()*problem.ymax) for i in range(nb_points_path)], problem.start1, problem.goal1) for k in range(S)]
    sigma_sq = 200
    X = [random_path(problem, nb_points_path) for k in range(S)]
    display_environment(problem, paths = X)
    
    V = [Path([Point(0,0) for i in range(nb_points_path)], problem.start1, problem.goal1) for k in range(S)]
    P = [x for x in X]
    g_index = min(range(S), key=lambda i: fitness(P[i])) # index of the particle with global best position
    best_f = fitness(P[g_index])

    for k in range(max_iter):
        for i in range(S): 
            r1 = random.random()
            r2 = random.random()
            if random_period > 0 and k%random_period == 0:
                X[i] = random_path(problem, nb_points_path)
            else:
                V[i] = (V[i]*w + (P[i]-X[i])*c1*r1 + (P[g_index]-X[i])*c2*r2).clamp_norm(50)
                X[i] = (X[i] + V[i]).clamp(problem.xmax, problem.ymax)

            if fitness(X[i]) < fitness(P[i]):
                P[i] = X[i]
        
        for i in range(S):
            curr_f = fitness(P[i])
            if  curr_f < best_f:
                g_index = i
                best_f = curr_f


        print(best_f)

    display_environment(problem, paths=P, path=P[g_index])
    return P[g_index]
