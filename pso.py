
import time
from environment import Problem, Point, Path, load_problem, display_environment
import random
import copy
from math import exp

def random_path(problem: Problem, nb_points_path:int) -> Path:
    """random path"""
    return Path([Point(random.random()*problem.xmax, random.random()*problem.ymax) for i in range(nb_points_path)], problem.start1, problem.goal1)

def semi_random_path(problem: Problem, nb_points_path:int) -> Path:
    """random path sampled with gaussian around the line y = x"""
    sigma_sq = 200
    points = []
    for i in range(nb_points_path):
        t = (i + 1) / (nb_points_path + 1)

        x = problem.start1.x * (1 - t) + problem.goal1.x * t + random.gauss(0, sigma_sq)
        y = problem.start1.y * (1 - t) + problem.goal1.y * t + random.gauss(0, sigma_sq)

        points.append(Point(x, y).clamp(problem.xmax, problem.ymax))

    return Path(points, problem.start1, problem.goal1)

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


def fitness(path: Path , problem: Problem) -> float:
    penalty = 10000.
    return path.nb_pair_collision(problem.obstacles) * penalty + path.length()

def particle_swarm_optimization(problem : Problem, 
                                fitness = fitness, 
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
                                DEBUG: bool = True, 
                                show_paths_start: bool = False
                                ) -> tuple[Path, list[Path]]:
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
    show_paths_start: display the initial paths of the particles if True
    """


    # initialize paths
    X = [symmetric_prog_path(problem, nb_points_path, i, S) for i in range(S)]
    
    if show_paths_start:
        display_environment(problem, paths=X)

    V = [Path([Point(0,0) for i in range(nb_points_path)], problem.start1, problem.goal1) for k in range(S)]
    P = [x for x in X]
    g_index = min(range(S), key=lambda i: fitness(P[i], problem)) # index of the particle with global best position
    fitness_g = fitness(P[g_index], problem)

    best_path = P[g_index]    # useful if simulated annealing is activated, to keep track of the best solution found even if g is not the best anymore
    best_fitness = fitness_g

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

            curr_fitness = fitness(X[i], problem)

            # dimension learning
            if dim_learning and k - last_updated[i] >= nb_update_before_dim:
                new_path = Path([x for x in X[i].points], X[i].start, X[i].end)
                for j in range(nb_points_path):
                    old_point = new_path.points[j]
                    new_path.points[j] = P[g_index].points[j]
                    new_fitness = fitness(new_path, problem)
                    if new_fitness < curr_fitness:
                        curr_fitness = new_fitness
                    else:
                        new_path.points[j] = old_point
                X[i] = new_path
        
            # update particle's best solution
            if curr_fitness < fitness(P[i], problem):
                P[i] = X[i]
                last_updated[i] = k

        # update temperature
        if simulated_annealing:
            T *= beta

        # update global best solution
        for i in range(S):
            curr_f = fitness(P[i], problem)
            if simulated_annealing and T >= 0.01:
                delta = curr_f - fitness_g
                if curr_f <= fitness_g:
                    prob = 1
                elif delta < T*10.:
                    prob = exp(-delta/T)
                else: # exp(-delta/T) is roughly 0
                    prob = 0

                u = random.uniform(0,1)
                if u <= prob:
                    g_index = i
                    fitness_g = curr_f
            else:        
                if  curr_f < fitness_g:
                    g_index = i
                    fitness_g = curr_f
            
            # independently of simulated annealing, keep track of the best solution found so far
            if curr_f < best_fitness:
                best_fitness = curr_f
                best_path = P[i]

        if DEBUG:
            print(fitness_g, "at iteration", k, "best fitness so far:", best_fitness)

    return best_path, P


if __name__ == "__main__":
    # set random seed
    random.seed(42)

    # Pick scenario from command line
    from sys import argv

    if len(argv) != 2:
        scenario_nb = 0
    else:
        scenario_nb = int(argv[1])
    timer = time.time()
    prob = load_problem(f"./scenarios/scenario{scenario_nb}.txt")
    print(f"Environment loaded in {time.time() - timer:.2f} seconds")
    timer = time.time()

    best_path, paths = particle_swarm_optimization(
        prob,
        fitness = fitness,
        max_iter=300,
        nb_points_path = 10,
        S=300,
        c1=2.0,
        c2=1.2,
        w = 0.8,
        random_restart=True,
        random_period=15,
        simulated_annealing=True,
        beta=0.95,
        T0=10000,
        dim_learning=True,
        nb_update_before_dim=15,
        DEBUG=True,
        show_paths_start=True
    )

    print(f"PSO completed in {time.time() - timer:.2f} seconds.")
    
    if best_path.nb_pair_collision(prob.obstacles) == 0:
        print(
            f"Path found with total length {best_path.length():.2f}"
        )
        display_environment(prob, best_path, paths)
    else:
        print("No path found")
        display_environment(prob, best_path, paths)