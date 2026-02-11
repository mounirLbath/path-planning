
from environment import Problem, Point, Path, load_problem, display_environment
import random
import copy
from math import exp

random.seed(42)

def random_path(problem: Problem, nb_points_path:int) -> Path:
    """random path"""
    return Path([Point(random.random()*problem.xmax, random.random()*problem.ymax) for i in range(nb_points_path)], problem.start1, problem.goal1)

def semi_random_path(problem: Problem, nb_points_path:int) -> Path:
    """random path sampled with gaussian around the line y = x"""
    sigma_sq = 400
    return Path([Point(random.gauss(0, sigma_sq)+(i+1)*problem.xmax/(nb_points_path+1), random.gauss(0, sigma_sq)+(i+1)*problem.ymax/(nb_points_path+1)).clamp(problem.xmax, problem.ymax)for i in range(nb_points_path)], problem.start1, problem.goal1)

def symmetric_prog_path(problem: Problem, nb_points_path: int, i: int, S: int) -> Path:
    """random paths sampled curves that roughly span the whole environment"""
    exponent = (i // 2 + 1)  / ((S + 1) // 2)
    points = []
    for k in range(nb_points_path):
        t = (k + 1) / (nb_points_path + 1)
        x = problem.xmax * t + random.gauss(0, 200)
        y = problem.xmax * (t ** exponent) + random.gauss(0, 200)

        # mirror every odd index
        if i % 2 == 1:
            x, y = y, x

        points.append(Point(x, y).clamp(problem.xmax, problem.ymax))

    return Path(points, problem.start1, problem.goal1)



def particle_swarm_optimization(problem : Problem, 
                                fitness, 
                                max_iter: int = 100, 
                                nb_points_path = 10,
                                S: int = 500, # nb of particles
                                c1: float = 1.5, 
                                c2: float = 1.5,
                                w:float = 0.7, 
                                random_restart : bool = True, 
                                random_period: int = 20, 
                                simulated_annealing: bool = False,
                                beta: float = 0.95,
                                T0: float = 10000,
                                dim_learning: bool = False,
                                nb_update_before_dim: int = 5, 
                                DEBUG: bool = True
                                ) -> Path:
    """
    Returns the path found by the particle swarm optimization algorithm
    max_iter: maximum number of iterations before stopping the algorithm
    nb_points_path: number of points on the discretized path
    fitness: the fitness function (takes a Path and returns its fitness value)
    S: number of particles to simulate
    c1 and c2: acceleration coefficients
    w: inertia weight for velocity
    random_restart: set to True to include random restart optimization
    random_period: set only if random_restart=True
    simulated_annealing: set to True to include simulated annealing
    beta and T0: parameters for simulated annealing
    dim_learning: set to True to include dimension learning optimization
    nb_update_before_dim: time to wait without any updates of P[i] before enforcing dimension learning 
    DEBUG: show logs if True
    """


    # initialize paths
    X = [symmetric_prog_path(problem, nb_points_path, k, S) for k in range(S)]

    display_environment(problem, path=X[150])

    V = [Path([Point(0,0) for i in range(nb_points_path)], problem.start1, problem.goal1) for k in range(S)]
    P = [x for x in X]
    g_index = min(range(S), key=lambda i: fitness(P[i])) # index of the particle with global best position
    best_f = fitness(P[g_index])

    # initialize temperature for simulated annealing
    T = T0

    # stores the last time P[i] was updated for each i
    last_updated = [0] * S

    for k in range(max_iter):
        for i in range(S): 
            r1 = random.random()
            r2 = random.random()
            # update particle
            if random_restart and random_period > 0 and k > 0 and k%random_period == 0:
                # reset to random path if random restart is activated
                X[i] = symmetric_prog_path(problem, nb_points_path, i, S)
                V[i] =Path([Point(0,0) for i in range(nb_points_path)], problem.start1, problem.goal1) 
            else:
                V[i] = (V[i]*w + (P[i]-X[i])*c1*r1 + (P[g_index]-X[i])*c2*r2).clamp_norm(60)
                X[i] = (X[i] + V[i]).clamp(problem.xmax, problem.ymax)

            curr_fitness = fitness(X[i])

            # dimension learning
            if dim_learning and k - last_updated[i] >= nb_update_before_dim:
                new_path = Path([x for x in X[i].points], X[i].start, X[i].end)
                for j in range(nb_points_path):
                    old_point = new_path.points[j]
                    new_path.points[j] = P[g_index].points[j]
                    new_fitness = fitness(new_path)
                    if new_fitness < curr_fitness:
                        curr_fitness = new_fitness
                    else:
                        new_path.points[j] = old_point
                X[i] = new_path
        
            # update particle's best solution
            if curr_fitness < fitness(P[i]):
                P[i] = X[i]
                last_updated[i] = k

        # update temperature
        if simulated_annealing:
            T *= beta

        # update global best solution
        for i in range(S):
            curr_f = fitness(P[i])
            if simulated_annealing and T >= 0.01:
                delta = curr_f - best_f
                if curr_f <= best_f:
                    prob = 1
                elif delta < T*10.:
                    prob = exp(-delta/T)
                else: # exp(-delta/T) ~ 0
                    prob = 0

                u = random.uniform(0,1)
                if u <= prob:
                    g_index = i
                    best_f = curr_f
            else:        
                if  curr_f < best_f:
                    g_index = i
                    best_f = curr_f

        if DEBUG:
            print(best_f)

    display_environment(problem, paths=P, path=P[g_index])
    return best_f, P[g_index]
