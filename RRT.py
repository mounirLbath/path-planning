import random
import time
from math import cos, sin
from math import pi as PI

import numpy as np
from scipy.stats import truncnorm

from environment import Point, Problem, display_environment, load_problem
from utils import distance, segment_collision

COSTS = dict()


class Node:
    """Node in the RRT tree"""

    def __init__(self, point: Point, parent: int, cost: float):
        self.point = point
        self.parent = parent
        self.cost = cost
        self.children: set[int] = set()


def add_node(tree: list[Node], point: Point, parent: int, cost: float) -> None:
    tree.append(Node(point, parent, cost))
    tree[parent].children.add(len(tree) - 1)


def sample_random_point(problem: Problem, p_bias: float = 0, r_sampling: float = 0, path: list[Point] = None) -> Point:
    if path is None or len(path) == 0 or random.random() >= p_bias:
        return Point(random.uniform(0, problem.xmax), random.uniform(0, problem.ymax))
    else:
        v_path = random.choice(path[1:-1])  # random interior vertex on the path
        # Sample from truncated Gaussian around v_path
        radius = truncnorm.rvs(0, r_sampling, loc=0, scale=r_sampling / 2)
        angle = random.uniform(0, 2 * PI)
        x = v_path.x + radius * cos(angle)
        y = v_path.y + radius * sin(angle)
        x = max(0, min(problem.xmax, x))
        y = max(0, min(problem.ymax, y))
        return Point(x, y)


def nodes_around(
    grid_y_x: list[list[list[int]]],
    point: Point,
    delta_r: float,
    grid_zone: list[tuple[int, int]] = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)],
):
    for i, j in grid_zone:
        y_idx = int(point.y // delta_r) + i
        x_idx = int(point.x // delta_r) + j
        if 0 <= y_idx < len(grid_y_x) and 0 <= x_idx < len(grid_y_x[0]):
            yield from grid_y_x[y_idx][x_idx]


def nearest_node_index(nodes: list[Node], grid_y_x: list[list[list[int]]], delta_r: float, point: Point) -> int:
    i_best = 0
    d_best = float("inf")
    # Square approximation; at worst we look for nodes sqrt(2) times too far away
    for grid_distance in range(0, max(len(grid_y_x), len(grid_y_x[0]))):
        grid_zone = [
            (i, j)
            for i in range(-grid_distance, grid_distance + 1)
            for j in range(-grid_distance, grid_distance + 1)
            if max(abs(i), abs(j)) == grid_distance
        ]
        for node_index in nodes_around(grid_y_x, point, delta_r, grid_zone):
            d = distance(nodes[node_index].point, point)
            if d < d_best:
                d_best = d
                i_best = node_index
        # We may find a grid_distance 1 with distance 0, but grid_distance 2 will have dist > delta_r
        if d_best <= (grid_distance - 1) * delta_r:
            break
    return i_best


def crop_vr(v_near: Point, v_rand: Point, delta_s: float) -> Point:
    dist = distance(v_near, v_rand)
    if dist <= delta_s:
        return v_rand
    scale = delta_s / dist
    return v_near + (v_rand - v_near) * scale


def reconstruct_path(nodes: list[Node], index: int) -> list[Node]:
    path: list[Node] = []
    while index != -1:
        path.append(nodes[index])
        index = nodes[index].parent
    path.reverse()
    return path


def _switch_parent(nodes: list["Node"], node_index: int, new_parent: int):
    nodes[nodes[node_index].parent].children.discard(node_index)
    nodes[node_index].parent = new_parent
    nodes[new_parent].children.add(node_index)


def _propagate_costs(nodes: list[Node], index: int):
    stack = list(nodes[index].children)
    while stack:
        child_index = stack.pop()
        new_cost = nodes[nodes[child_index].parent].cost + distance(
            nodes[nodes[child_index].parent].point, nodes[child_index].point
        )
        if new_cost > nodes[child_index].cost:
            raise ValueError("Rewiring causes a higher cost")
        elif new_cost == nodes[child_index].cost:
            raise ValueError("Rewiting does not cause a lower cost")
        nodes[child_index].cost = new_cost
        stack.extend(nodes[child_index].children)


def switch_parent_and_propagate(nodes: list[Node], index: int, new_parent: int):
    new_cost = nodes[new_parent].cost + distance(nodes[new_parent].point, nodes[index].point)
    assert new_cost < nodes[index].cost, "New parent does not provide a lower cost"
    nodes[index].cost = new_cost
    _switch_parent(nodes, index, new_parent)
    _propagate_costs(nodes, index)


def rewire_nodes(
    nodes: list[Node], grid_y_x: list[list[list[int]]], problem: Problem, delta_r: float, rewire_from: int, goal: int
) -> list[int]:
    global COSTS
    rewired_nodes = []
    rewire_from_point = nodes[rewire_from].point
    updated_goal = False
    for index in nodes_around(grid_y_x, rewire_from_point, delta_r):
        if index == rewire_from:
            continue
        if not segment_collision(rewire_from_point, nodes[index].point, problem.obstacles):
            new_cost = nodes[rewire_from].cost + distance(rewire_from_point, nodes[index].point)
            if new_cost < nodes[index].cost:
                rewired_nodes.append(index)
                switch_parent_and_propagate(nodes, index, rewire_from)
    return rewired_nodes, updated_goal


def path_optimization(problem: Problem, tree: list[Node], index: int, k_rope: int) -> None:
    path: list[Node] = reconstruct_path(tree, index)
    goal = tree[index]
    if len(path) < 3:
        # One segment only: nothing to shortcut
        return
    # Verify if we can shortcut to the first element
    if not segment_collision(goal.point, path[0].point, problem.obstacles):
        switch_parent_and_propagate(tree, index, path[1].parent)
        return
    # Best shortcut for now (parent of index)
    best_i, best_k = len(path) - 1, k_rope
    for i in range(len(path) - 2, 0, -1):
        for k in range(1, k_rope + 1):
            v_short = ((k_rope - k) / k_rope) * path[i].point + ((k / k_rope) * path[i - 1].point)
            if not segment_collision(goal.point, v_short, problem.obstacles):
                # We consider this as the new best possible shortcut
                best_i, best_k = i, k
            else:
                if best_k != k_rope:
                    # We insert the node in the tree
                    v_short = ((k_rope - best_k) / k_rope) * path[best_i].point + (
                        (best_k / k_rope) * path[best_i - 1].point
                    )
                    add_node(
                        tree,
                        v_short,
                        path[best_i].parent,
                        tree[path[best_i].parent].cost + distance(tree[path[best_i].parent].point, v_short),
                    )
                    i_shortcut = len(tree) - 1
                else:
                    # We use the existing node
                    i_shortcut = path[best_i].parent  # Index of path[i-1]=v_short in the tree

                if best_i != len(path) - 1:
                    # Otherwise we have no shortcut
                    switch_parent_and_propagate(tree, index, i_shortcut)
                return path_optimization(problem, tree, i_shortcut, k_rope)
    raise ValueError("Should have found a shortcut to the first element, check the logic of path_optimization")


def rrt(
    problem: Problem,
    delta_s: float,
    delta_r: float,
    max_iters: int,
    recursive_rewire: bool = False,
    optimize_after_goal: bool = False,
    display_tree_end: bool = False,
    path_optimize: bool = False,
    k_rope=10,
    sample_optimize: bool = False,
    p_bias: float = 0.75,
    r_sampling: float = 1.0,
) -> list[Point] | None:
    global COSTS
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
    last_optimized_step = None

    timer = time.time()
    grid_y_x: list[list[list[int]]] = [
        [[] for i in range(int(problem.xmax / delta_r) + 1)] for j in range(int(problem.ymax / delta_r) + 1)
    ]
    grid_y_x[int(problem.start1.y // delta_r)][int(problem.start1.x // delta_r)].append(0)

    for _ in range(max_iters):
        check_infinite = 0
        while check_infinite < 1000:
            timer = time.time()
            if goal and sample_optimize:
                v_r = sample_random_point(
                    problem, p_bias=p_bias, r_sampling=r_sampling, path=[n.point for n in reconstruct_path(nodes, goal)]
                )
            else:
                v_r = sample_random_point(problem)
            # Only sample points that could improve the path to the goal; if we are not using the improved sampling
            while (
                not sample_optimize
                and goal
                and distance(problem.start1, v_r) + distance(v_r, problem.goal1) >= nodes[goal].cost
            ):
                v_r = sample_random_point(problem)
            COSTS["sampling"] = COSTS.get("sampling", 0) + time.time() - timer

            # nearest node
            timer = time.time()
            i_n = nearest_node_index(nodes, grid_y_x, delta_r, v_r)
            v_n = nodes[i_n].point
            COSTS["nearest"] = COSTS.get("nearest", 0) + time.time() - timer

            # crop random point within delta_s
            v_new = crop_vr(v_n, v_r, delta_s)
            if not segment_collision(v_n, v_new, problem.obstacles):
                break
            check_infinite += 1

        if check_infinite == 1000:
            raise ValueError(
                "Could not find a valid random point after 1000 attempts, consider decreasing delta_s or checking your environment"
            )

        # choose best parent within delta_r for v_new
        # we check the 9 grid cells around v_new and that makes a sufficient condition (picking a parent further than delta_r with better total distance would still be optimal, and delta_r is here just to improve the efficiency of the algorithm)
        timer = time.time()
        best_parent = i_n
        best_cost = nodes[i_n].cost + distance(v_n, v_new)
        for index in nodes_around(grid_y_x, v_new, delta_r):
            node = nodes[index]
            if not segment_collision(node.point, v_new, problem.obstacles):
                c = node.cost + distance(node.point, v_new)
                if c < best_cost:
                    best_cost = c
                    best_parent = index

        add_node(nodes, v_new, best_parent, best_cost)
        nodes[best_parent].children.add(len(nodes) - 1)
        grid_y_x[int(v_new.y // delta_r)][int(v_new.x // delta_r)].append(len(nodes) - 1)
        i_new = len(nodes) - 1
        COSTS["best_parent"] = COSTS.get("best_parent", 0) + time.time() - timer

        # rewiring nodes close enough to v_new similarly
        timer = time.time()
        rewire_from = i_new
        rewired_nodes, rewired_goal = rewire_nodes(nodes, grid_y_x, problem, delta_r, rewire_from, goal)
        if rewired_goal:
            last_optimized_step = len(nodes) - 1
        COSTS["rewire"] = COSTS.get("rewire", 0) + time.time() - timer

        # Custom addition: rewire recursively if asked
        if recursive_rewire:
            timer = time.time()
        while recursive_rewire and len(rewired_nodes) > 0:
            rewire_from = rewired_nodes.pop()
            rewired_nodes, rewired_goal = rewire_nodes(nodes, grid_y_x, problem, delta_r, rewire_from, goal)
            rewired_nodes.extend(rewired_nodes)
            if rewired_goal:
                last_optimized_step = len(nodes) - 1
        if recursive_rewire:
            COSTS["recursive_rewire"] = COSTS.get("recursive_rewire", 0) + time.time() - timer

        # check if we reached the goal
        if not segment_collision(v_new, problem.goal1, problem.obstacles) and (
            goal is None or best_cost + distance(v_new, problem.goal1) < nodes[goal].cost
        ):
            if goal is not None:
                # update goal instead
                switch_parent_and_propagate(nodes, goal, i_new)
                last_optimized_step = len(nodes) - 1
                # print("New node provides a better path to the goal, with cost ",nodes[goal].cost," at step ",len(nodes) - 1,)
                continue
            # add the goal node
            add_node(nodes, problem.goal1, i_new, best_cost + distance(v_new, problem.goal1))
            nodes[i_new].children.add(len(nodes) - 1)
            goal = len(nodes) - 1
            print("Goal found with cost ", nodes[goal].cost, " at step ", len(nodes) - 1)
            last_optimized_step = goal
            if not optimize_after_goal:
                if path_optimize:
                    timer = time.time()
                    path_optimization(problem, nodes, goal, k_rope)
                    COSTS["path_optimization"] = COSTS.get("path_optimization", 0) + time.time() - timer
                break

        # Path optimization
        if goal is not None and path_optimize:
            past_cost = nodes[goal].cost
            timer = time.time()
            path_optimization(problem, nodes, goal, k_rope)
            COSTS["path_optimization"] = COSTS.get("path_optimization", 0) + time.time() - timer
            if nodes[goal].cost < past_cost:
                last_optimized_step = len(nodes) - 1

    if display_tree_end:
        display_tree(problem, nodes)

    if goal is not None:
        # display_tree(problem, nodes)
        return nodes[goal].cost, [i.point for i in reconstruct_path(nodes, goal)], goal, last_optimized_step

    return None, None, None, None


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


if __name__ == "__main__":
    # set random seed
    random.seed(1)
    np.random.seed(1)

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
    cost, path, steps_taken, last_optimized_step = rrt(
        prob,
        delta_s=40.0,
        delta_r=150.0,
        max_iters=1000,
        recursive_rewire=False,
        optimize_after_goal=True,
        display_tree_end=False,
        path_optimize=True,
        k_rope=1000,
        sample_optimize=True,
        p_bias=0.8,
        r_sampling=20.0,
    )
    print(f"RRT completed in {time.time() - timer:.2f} seconds. Decomposition of costs:")
    for k, v in COSTS.items():
        print(f"  {k}: {v:.4f} seconds")
    print("Total accounted time: ", sum(COSTS.values()), " seconds")
    if path is not None:
        assert abs(cost - sum(distance(path[i], path[i + 1]) for i in range(len(path) - 1))) < 1e-6
        print(
            f"Path found with {len(path)} points and total length {cost:.2f}"
            + (
                f", found in {steps_taken} steps"
                if last_optimized_step == steps_taken
                else f", last optimized step: {last_optimized_step}"
            )
        )
        # display_environment(prob, path)
    else:
        print("No path found")
