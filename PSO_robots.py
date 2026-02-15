
import time
from environment import Problem, Point, Path, load_problem, display_environment
import random
import copy
from math import exp
from utils import distance, dot

def random_path(problem: Problem, nb_points_path:int, robot: int) -> Path:
    """random path"""
    if robot == 1:
        start = problem.start1
        end = problem.goal1
    else:
        start = problem.start2
        end = problem.goal2
    return Path([Point(random.random()*problem.xmax, random.random()*problem.ymax) for i in range(nb_points_path)], start, end)

def semi_random_path(problem: Problem, nb_points_path:int, robot: int) -> Path:
    """random path sampled with gaussian around the line y = x"""
    if robot == 1:
        start = problem.start1
        end = problem.goal1
    else:
        start = problem.start2
        end = problem.goal2
    sigma_sq = 400
    points = []
    for i in range(nb_points_path):
        t = (i + 1) / (nb_points_path + 1)

        x = start.x * (1 - t) + end.x * t + random.gauss(0, sigma_sq)
        y = start.y * (1 - t) + end.y * t + random.gauss(0, sigma_sq)

        points.append(Point(x, y).clamp(problem.xmax, problem.ymax))

    return Path(points, start, end)

def symmetric_prog_path(problem: Problem, nb_points_path: int, i: int, S: int, robot: int) -> Path:
    """random paths sampled curves that roughly span the whole environment"""
    exponent = (i // 2 + 1)  / ((S + 1) // 2)
    if robot == 1:
        start = problem.start1
        end = problem.goal1
    else:
        start = problem.start2
        end = problem.goal2
    points = []
    for k in range(nb_points_path):
        t = (k + 1) / (nb_points_path + 1)
        
        x = start.x * (1 - t) + end.x * t + random.gauss(0, 200)
        y = start.y * (1 - t) + end.y * t + random.gauss(0, 200)

        # mirror every odd index
        if i % 2 == 1:
            x, y = y, x

        points.append(Point(x, y).clamp(problem.xmax, problem.ymax))

    return Path(points, start, end)


def collision(path1: Path, path2: Path, radius: float) -> int:
    """
    returns if there is a collision between two robots, given a safety radius and that the 2 robots move at 1m/s
    """

    points1 = [path1.start, *path1.points, path1.end]
    points2 = [path2.start, *path2.points, path2.end]


    i1 = 0
    i2 = 0
    collisions = 0
    while i1 < len(points1)-1 and i2 < len(points2)-1:
        a = points1[i1]
        b = points1[i1+1]
        c = points2[i2]
        d = points2[i2+1]

        d1 = distance(a, b)
        d2 = distance(c, d)

        if d1 == 0:
            i1 += 1
        elif d2 == 0:
            i2 += 1
        else:
            u = a-c
            v = (b-a)*(1/d1) - (d-c)*(1/d2)

            C = dot(u, u) - 4*radius**2
            B= 2*dot(u, v)
            A = dot(v, v)

            delta = B**2 - 4*A*C
            if delta >= 0:
                t1 = -B - delta**0.5
                t2 = -B + delta**0.5
                if t1 <= 2*A*min(d1,d2) and t2 >= 0:
                    return True

        if d1 < d2:
            points2[i2] = c + (d1/d2)*(d-c)
            i1 += 1
        else:
            points1[i1] = a + (d2/d1)*(b-a)
            i2 += 1

    return False

def fitness(path1: Path , path2: Path , problem: Problem) -> float:
    penalty = 1000000.
    penalty_radius = 1000000.
    radius = problem.safety_radius
    collision_count = collision(path1, path2, radius)


    return path1.nb_pair_collision(problem.obstacles) * penalty + path1.length() + path2.nb_pair_collision(problem.obstacles) * penalty + path2.length() + collision_count * penalty_radius

def particle_swarm_optimization_robots(problem : Problem, 
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
                                patience: int = 100,
                                DEBUG: bool = True, 
                                show_paths_start: bool = False
                                ) -> tuple[Path, list[Path]]:
    """
    Returns the paths found by the particle swarm optimization algorithm fpr 2 robots
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
    patience: number of iterations without improvement in the best fitness before stopping the algorithm
    DEBUG: show logs if True
    show_paths_start: display the initial paths of the particles if True
    """


    # initialize paths
    X1 = [semi_random_path(problem, nb_points_path, 1) for i in range(S)]
    X2 = [semi_random_path(problem, nb_points_path, 2) for i in range(S)]
    
    if show_paths_start:
        display_environment(problem, paths=X2)

    V1 = [Path([Point(0,0) for i in range(nb_points_path)], problem.start1, problem.goal1) for k in range(S)]
    V2 = [Path([Point(0,0) for i in range(nb_points_path)], problem.start2, problem.goal2) for k in range(S)]
    P1 = [x for x in X1]
    P2 = [x for x in X2]
    g_index = min(range(S), key=lambda i: fitness(P1[i], P2[i], problem)) # index of the particle with global best position
    fitness_g = fitness(P1[g_index], P2[g_index], problem)

    best_paths = P1[g_index], P2[g_index]    # useful if simulated annealing is activated, to keep track of the best solution found even if g is not the best anymore
    best_fitness = fitness_g
    last_update_best_fitness = 0

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
                X1[i] = semi_random_path(problem, nb_points_path, 1)
                X2[i] = semi_random_path(problem, nb_points_path, 2)
                V1[i] =Path([Point(0,0) for i in range(nb_points_path)], problem.start1, problem.goal1) 
                V2[i] =Path([Point(0,0) for i in range(nb_points_path)], problem.start2, problem.goal2) 
            else:
                V1[i] = (V1[i]*w + (P1[i]-X1[i])*c1*r1 + (P1[g_index]-X1[i])*c2*r2).clamp_norm(60)
                V2[i] = (V2[i]*w + (P2[i]-X2[i])*c1*r1 + (P2[g_index]-X2[i])*c2*r2).clamp_norm(60)
                X1[i] = (X1[i] + V1[i]).clamp(problem.xmax, problem.ymax)
                X2[i] = (X2[i] + V2[i]).clamp(problem.xmax, problem.ymax)

            curr_fitness = fitness(X1[i], X2[i], problem)

            # dimension learning
            if dim_learning and k - last_updated[i] >= nb_update_before_dim:
                new_path1 = Path([x for x in X1[i].points], X1[i].start, X1[i].end)
                new_path2 = Path([x for x in X2[i].points], X2[i].start, X2[i].end)
                for j in range(nb_points_path):
                    old_point1 = new_path1.points[j]
                    old_point2 = new_path2.points[j]
                    new_path1.points[j] = P1[g_index].points[j]
                    new_path2.points[j] = P2[g_index].points[j]
                    new_fitness = fitness(new_path1, new_path2, problem)
                    if new_fitness < curr_fitness:
                        curr_fitness = new_fitness
                    else:
                        new_path1.points[j] = old_point1
                        new_path2.points[j] = old_point2
                X1[i] = new_path1
                X2[i] = new_path2
        
            # update particle's best solution
            if curr_fitness < fitness(P1[i], P2[i], problem):
                P1[i] = X1[i]
                P2[i] = X2[i]
                last_updated[i] = k

        # update temperature
        if simulated_annealing:
            T *= beta

        # update global best solution
        for i in range(S):
            curr_f = fitness(P1[i], P2[i], problem)
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
                best_paths = P1[i], P2[i]
                last_update_best_fitness = k

        if k - last_update_best_fitness >= patience:
            print("No improvement in the best fitness for ", patience, " iterations, stopping the algorithm")
            break
                
        if DEBUG:
            print(fitness_g, "at iteration", k, "best fitness so far:", best_fitness)

    return best_paths, P1, P2


if __name__ == "__main__":
    # set random seed
    random.seed(42)

    # Pick scenario from command line
    from sys import argv

    if len(argv) != 2:
        scenario_nb = 0
    else:
        scenario_nb = int(argv[1])
        
    prob = load_problem(f"./scenarios/scenario{scenario_nb}.txt")


    t0 = time.time()
    best_paths, _, _ = particle_swarm_optimization_robots(
        prob,
        max_iter=5000,
        nb_points_path=15,
        S=40,
        c1=1.2,
        c2=1.2,
        w=0.9,
        random_restart=True,
        random_period=15,
        simulated_annealing=True,
        beta=0.95,
        T0=1e7,
        dim_learning=True,
        nb_update_before_dim=30,
        patience=100,
        DEBUG=True,
        show_paths_start=True,
    )
    print(f"PSO done in {time.time() - t0:.2f}s")
    
    animate_paths(prob, best_paths[0], best_paths[1], speed=1.0, playback_seconds=8.0, fps=30, trail=True)

