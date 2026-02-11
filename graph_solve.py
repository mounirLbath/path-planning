import random
import time
from queue import PriorityQueue

from environment import Path, Point, Problem, Rectangle, display_environment, load_problem
from utils import distance, segment_collision


def graph_solve(prob: Problem) -> tuple[float, list[Point] | None]:
    # We make the obstacles slightly larger to be sure to not have any unwanted path
    epsilon = 1e-6
    for i in range(len(prob.obstacles)):
        o = prob.obstacles[i]
        prob.obstacles[i] = Rectangle(o.x - epsilon, o.y - epsilon, o.width + 2 * epsilon, o.height + 2 * epsilon)
    # Add the limits of the environment as obstacles to avoid going outside

    prob.obstacles.append(Rectangle(-1, -1, prob.xmax + 2, 1))
    prob.obstacles.append(Rectangle(-1, -1, 1, prob.ymax + 2))
    prob.obstacles.append(Rectangle(prob.xmax, -1, 1, prob.ymax + 2))
    prob.obstacles.append(Rectangle(-1, prob.ymax, prob.xmax + 2, 1))

    # Dijkstra on obstacle vertices
    queue = PriorityQueue()
    queue.put((0, prob.start1, [prob.start1]))
    visited = set()

    while not queue.empty():
        cost, point, path = queue.get()
        if point in visited:
            continue
        if point == prob.goal1:
            return cost, path
        visited.add(point)

        if not segment_collision(point, prob.goal1, prob.obstacles, strict=True):
            queue.put((cost + distance(point, prob.goal1), prob.goal1, path + [prob.goal1]))

        for obs in prob.obstacles:
            for vertex in obs.vertices():
                if vertex in visited:
                    continue
                if not segment_collision(point, vertex, prob.obstacles, strict=True):
                    queue.put((cost + distance(point, vertex), vertex, path + [vertex]))

    return None, None


if __name__ == "__main__":
    # set random seed
    random.seed(1)

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
    cost, path = graph_solve(prob)
    print(f"Graph solve completed in {time.time() - timer:.2f} seconds. Decomposition of costs:")
    if path is not None:
        assert abs(cost - sum(distance(path[i], path[i + 1]) for i in range(len(path) - 1))) < 1e-6
        print(f"Path found with {len(path)} points and total length {cost:.2f}")
        display_environment(prob, Path(path, path[0], path[-1]))
    else:
        print("No path found")
