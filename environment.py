import matplotlib.pyplot as plt
from utils import *
from geometry import Point, Rectangle
import numpy as np

class Path:
    """A path which linear by piece"""

    def __init__(self, points: list[Point], start: Point, end: Point) -> None:
        self.points = points
        self.start = start
        self.end = end

    def collision(self, obstacles: list[Rectangle]) -> bool:
        """Returns true if path collides with an obstacle"""
        for i in range(len(self.points)-1):
            if segment_collision(self.points[i], self.points[i+1], obstacles):
                return True
        return segment_collision(self.start, self.points[0], obstacles) or segment_collision(self.end, self.points[-1], obstacles)
    
    def nb_pair_collision(self, obstacles: list[Rectangle]) -> bool:
        """Returns the number of pairs (obstacle, segment) that collide (where segment is in the path)"""
        n = 0
        for ob in obstacles:
            if segment_intersects_rect(self.start, self.points[0], ob) or segment_intersects_rect(self.end, self.points[-1], ob):
                n += 1
            for i in range(len(self.points)-1):
                if segment_intersects_rect(self.points[i], self.points[i+1], ob):
                    n += 1
        return n

    def length(self):
        d = distance(self.start, self.points[0]) or distance(self.end, self.points[-1])
        for i in range(len(self.points) -1 ):
            d += distance(self.points[i], self.points[i+1])
        return d
    
    def update(self, other):
        if len(other.points) != len(self.points):
            raise ValueError("Update was tried but point lists not of same length.")
        
        for i in range(len(self.points)):
            self.points[i] = other.points[i]
    
    def __add__(self, other):
        """Adds the 2 path component wise"""
        if len(self.points) != len(other.points):
            raise ValueError("Trying to add 2 paths with different sizes.")
        result = [self.points[i] + other.points[i] for i in range(len(self.points))]
        return Path(result, self.start, self.end)
    
    def __sub__(self, other):
        if len(self.points) != len(other.points):
            raise ValueError("Trying to substract 2 paths with different sizes.")
        
        result = [self.points[i] - other.points[i] for i in range(len(self.points))]
        
        return Path(result, self.start, self.end)

    def __mul__(self, scalar: float):
        return Path([p*scalar for p in self.points], self.start, self.end)
    
    def __str__(self):
        s = str(self.start)+";"
        for p in self.points:
            s += str(p) + ";"
        s += str(self.end)
        return s
    
    def clamp(self, xmax:float, ymax:float):
        return Path([p.clamp(xmax, ymax) for p in self.points], self.start, self.end)

    def clamp_norm(self, max_norm:float):
        return Path([p.clamp_norm(max_norm) for p in self.points], self.start, self.end)
        
    

class Problem:
    """Represents a complete path-planning problem with environment bounds, start and goal points, safety radius, and obstacles."""

    def __init__(
        self,
        xmax: float,
        ymax: float,
        start1: Point,
        goal1: Point,
        start2: Point,
        goal2: Point,
        safety_radius: float,
        obstacles: list[Rectangle],
    ) -> None:
        self.xmax = xmax
        self.ymax = ymax
        self.start1 = start1
        self.goal1 = goal1
        self.start2 = start2
        self.goal2 = goal2
        self.safety_radius = safety_radius
        self.obstacles = obstacles


def load_problem(file_path: str) -> Problem:
    """Load and validate a path-planning problem from a scenario file.

    The format is: xmax, ymax, start1(x,y), goal1(x,y), start2(x,y), goal2(x,y), R,
    followed by zero or more obstacles as quadruples (xo, yo, width, height).
    """
    tokens: list[float] = []
    # read the file
    with open(file_path) as f:
        for line in f:
            for part in line.split():
                try:
                    tokens.append(float(part))
                except ValueError:
                    raise ValueError(f"Non-numeric value in scenario file: {part}")

    # Basic validity: need at least the 11 header values and obstacles in groups of four.
    if len(tokens) < 11:
        raise ValueError("Scenario file must contain at least 11 numeric values.")

    # Take first 11 values for bounds, starts, goals, and radius.
    header = tokens[:11]
    xmax, ymax, s1x, s1y, g1x, g1y, s2x, s2y, g2x, g2y, radius = header

    # Remaining tokens describe obstacles in groups of four.
    obstacle_tokens = tokens[11:]
    if len(obstacle_tokens) % 4 != 0:
        raise ValueError("Obstacle data must be in groups of four numeric values.")

    start1, goal1 = Point(s1x, s1y), Point(g1x, g1y)
    start2, goal2 = Point(s2x, s2y), Point(g2x, g2y)
    if not (
        start1.is_within_bounds(xmax, ymax)
        and goal1.is_within_bounds(xmax, ymax)
        and start2.is_within_bounds(xmax, ymax)
        and goal2.is_within_bounds(xmax, ymax)
    ):
        raise ValueError("Point out of bounds.")

    obstacles: list[Rectangle] = []
    for i in range(0, len(obstacle_tokens), 4):
        ob = Rectangle(*obstacle_tokens[i : i + 4])
        if not ob.is_within_bounds(xmax, ymax):
            raise ValueError(f"Obstacle {i // 4} is out of bound.")
        obstacles.append(ob)

    return Problem(
        xmax=xmax,
        ymax=ymax,
        start1=start1,
        goal1=goal1,
        start2=start2,
        goal2=goal2,
        safety_radius=radius,
        obstacles=obstacles,
    )


def display_environment(problem: Problem, path: Path = None, paths: list[Path] = None):
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

    # Path
    if paths:
        for pa in paths:
            ax.plot([pa.start.x,*[p.x for p in pa.points], pa.end.x], [pa.start.y,*[p.y for p in pa.points], pa.end.y], "b-", linewidth=0.5)

    if path:
        ax.plot([path.start.x,*[p.x for p in path.points], path.end.x], [path.start.y,*[p.y for p in path.points], path.end.y], "r-", label="path", linewidth=0.5)

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.show()
