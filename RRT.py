import random

from environment import Point, Problem, display_environment, load_problem
from utils import distance, segment_collision


class Node:
    def __init__(self, point: Point, parent: int, cost: float):
        self.point = point
        self.parent = parent
        self.cost = cost
        self.children: set[int] = set()

    # TODO recompute cost each time?


def sample_random_point(problem: Problem) -> Point:
    return Point(random.uniform(0, problem.xmax), random.uniform(0, problem.ymax))


def nearest_node_index(nodes: list[Node], point: Point) -> int:
    i_near = 0
    d_best = float("inf")
    for i, node in enumerate(nodes):
        d = distance(node.point, point)
        if d < d_best:
            d_best = d
            i_near = i
    return i_near


def crop_vr(v_near: Point, v_rand: Point, delta_s: float) -> Point:
    dist = distance(v_near, v_rand)
    if dist <= delta_s:
        return v_rand
    scale = delta_s / dist
    return v_near + (v_rand - v_near) * scale


def reconstruct_path(nodes: list[Node], index: int) -> list[Point]:
    path: list[Point] = []
    while index != -1:
        path.append(nodes[index].point)
        index = nodes[index].parent
    path.reverse()
    return path


def nodes_around(grid_y_x: list[list[list[int]]], point: Point, delta_r: float) -> list[int]:
    indices = []
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            if 0 <= int(point.y // delta_r) + i < len(grid_y_x) and 0 <= int(point.x // delta_r) + j < len(grid_y_x[0]):
                indices.extend(grid_y_x[int(point.y // delta_r) + i][int(point.x // delta_r) + j])
    return indices


def rewire_nodes(
    nodes: list[Node], grid_y_x: list[list[list[int]]], problem: Problem, delta_r: float, rewire_from: int
) -> list[int]:
    rewired_nodes = []
    rewire_from_point = nodes[rewire_from].point
    for index in nodes_around(grid_y_x, rewire_from_point, delta_r):
        if index == rewire_from:
            continue
        if not segment_collision(rewire_from_point, nodes[index].point, problem.obstacles):
            new_cost = nodes[rewire_from].cost + distance(rewire_from_point, nodes[index].point)
            if new_cost < nodes[index].cost:
                nodes[index].cost = new_cost
                nodes[nodes[index].parent].children.discard(index)
                nodes[index].parent = rewire_from
                nodes[rewire_from].children.add(index)
                rewired_nodes.append(index)

                # update cost of descendants with a DFS
                stack = list(nodes[index].children)
                while stack:
                    child_index = stack.pop()
                    child_node = nodes[child_index]
                    new_cost = nodes[child_node.parent].cost + distance(
                        nodes[child_node.parent].point, child_node.point
                    )
                    if new_cost < child_node.cost:
                        child_node.cost = new_cost
                        stack.extend(child_node.children)
    return rewired_nodes


def rrt(
    problem: Problem,
    delta_s: float,
    delta_r: float,
    max_iters: int,
    recursive_rewire: bool = False,
    optimize_after_goal: bool = False,
) -> list[Point] | None:
    if delta_r < delta_s:
        raise ValueError("delta_r must be greater than or equal to delta_s")
    if delta_r > min(problem.xmax, problem.ymax):
        raise ValueError("delta_r is larger than one of the environment dimensions")

    if not segment_collision(problem.start1, problem.goal1, problem.obstacles):
        return [problem.start1, problem.goal1]
    # Warn that if delta_r is too small, the algorithm will start going slower instead of faster (too much memory used)
    if problem.xmax / delta_r * problem.ymax / delta_r > 1e6:
        print("Warning: delta_r is too small, the algorithm may run slowly due to high memory usage.")
        if problem.xmax / delta_r * problem.ymax / delta_r > 1e8:
            raise ValueError("delta_r is too small, the algorithm may run out of memory.")

    # Initialize nodes and grid for speeding up nearest neighbor search
    nodes: list[Node] = [Node(problem.start1, -1, 0.0)]
    goal = None

    grid_y_x: list[list[list[int]]] = [
        [[] for i in range(int(problem.xmax / delta_r) + 1)] for j in range(int(problem.ymax / delta_r) + 1)
    ]
    grid_y_x[int(problem.start1.y // delta_r)][int(problem.start1.x // delta_r)].append(0)

    for _ in range(max_iters):
        v_r = sample_random_point(problem)

        # nearest node
        i_n = nearest_node_index(nodes, v_r)
        v_n = nodes[i_n].point

        # crop random point within delta_s
        v_new = crop_vr(v_n, v_r, delta_s)
        if segment_collision(v_n, v_new, problem.obstacles):
            continue

        # choose best parent within delta_r for v_new
        # we check the 9 grid cells around v_new and that makes a sufficient condition (picking a parent further than delta_r with better total distance would still be optimal, and delta_r is here just to improve the efficiency of the algorithm)
        best_parent = i_n
        best_cost = nodes[i_n].cost + distance(v_n, v_new)
        for index in nodes_around(grid_y_x, v_new, delta_r):
            node = nodes[index]
            if not segment_collision(node.point, v_new, problem.obstacles):
                c = node.cost + distance(node.point, v_new)
                if c < best_cost:
                    best_cost = c
                    best_parent = index

        nodes.append(Node(v_new, best_parent, best_cost))
        nodes[best_parent].children.add(len(nodes) - 1)
        grid_y_x[int(v_new.y // delta_r)][int(v_new.x // delta_r)].append(len(nodes) - 1)
        i_new = len(nodes) - 1

        # rewiring nodes close enough to v_new similarly
        rewire_from = i_new
        rewired_nodes = rewire_nodes(nodes, grid_y_x, problem, delta_r, rewire_from)

        # Custom addition: rewire recursively if asked
        while recursive_rewire and len(rewired_nodes) > 0:
            rewire_from = rewired_nodes.pop()
            rewired_nodes.extend(rewire_nodes(nodes, grid_y_x, problem, delta_r, rewire_from))

        # goal check
        if not segment_collision(v_new, problem.goal1, problem.obstacles) and (
            goal is None or best_cost + distance(v_new, problem.goal1) < nodes[goal].cost
        ):
            nodes.append(Node(problem.goal1, i_new, best_cost + distance(v_new, problem.goal1)))
            nodes[i_new].children.add(len(nodes) - 1)
            # display_tree(problem, nodes)
            goal = len(nodes) - 1
            if not optimize_after_goal:
                break

    if goal is not None:
        # display_tree(problem, nodes)
        return reconstruct_path(nodes, goal)

    return None


def display_tree(problem: Problem, nodes: list[Node]) -> None:
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    ax.set_xlim(0, problem.xmax)
    ax.set_ylim(0, problem.ymax)
    ax.set_aspect("equal")

    # Obstacles
    for obs in problem.obstacles:
        ax.add_patch(plt.Rectangle((obs.x, obs.y), obs.width, obs.height, color="black"))

    # Start and goal points for 1st robot
    ax.plot(problem.start1.x, problem.start1.y, "ro", label="start", markersize=10, clip_on=False, zorder=3)
    ax.plot(problem.goal1.x, problem.goal1.y, "r*", label="goal", markersize=10, clip_on=False, zorder=3)

    # Start and goal points for 2nd robot
    ax.plot(problem.start2.x, problem.start2.y, "go", label="start2", markersize=10, clip_on=False, zorder=3)
    ax.plot(problem.goal2.x, problem.goal2.y, "g*", label="goal2", markersize=10, clip_on=False, zorder=3)

    # Tree
    for node in nodes:
        if node.parent != -1:
            parent_node = nodes[node.parent]
            ax.plot(
                [node.point.x, parent_node.point.x],
                [node.point.y, parent_node.point.y],
                "b-",
                linewidth=0.5,
            )

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.show()


# TODO authorize a longer run to continue improving the path; by recursively rewiring
if __name__ == "__main__":
    # set random seed
    random.seed(0)
    prob = load_problem("./scenarios/scenario1.txt")
    path = rrt(prob, delta_s=3.0, delta_r=10.0, max_iters=5000, recursive_rewire=False, optimize_after_goal=False)
    if path is not None:
        display_environment(prob, path)
        print(
            f"Path found with {len(path)} points and total length {sum(distance(path[i], path[i + 1]) for i in range(len(path) - 1)):.2f}"
        )
    else:
        print("No path found")
