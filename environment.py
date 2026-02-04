import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as MplRectangle


class Point:
    """Points in the environment."""

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    # Define basic vector operations
    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float):
        return Point(self.x * scalar, self.y * scalar)
    
    def is_within_bounds(self, xmax: float, ymax: float) -> bool:
        return 0 <= self.x <= xmax and 0 <= self.y <= ymax


class Rectangle:
    """Rectangles representing obstacles in the environment."""

    def __init__(self, x: float, y: float, width: float, height: float) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def contains_point(self, p: Point) -> bool:
        return self.x <= p.x <= self.x + self.width and self.y <= p.y <= self.y + self.height

    def is_within_bounds(self, xmax: float, ymax: float) -> bool:
        return self.x >= 0 and self.y >= 0 and self.x + self.width <= xmax and self.y + self.height <= ymax

    def edges(self):
        bottomLeft = Point(self.x, self.y)
        bottomRight = Point(self.x+self.width, self.y)
        topLeft = Point(self.x, self.y+self.height)
        topRight = Point(self.x+self.width, self.y+self.height)

        return [
            (bottomLeft, bottomRight),
            (bottomRight, topRight),
            (topRight, topLeft),
            (topLeft, bottomLeft),
        ]


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
    if not(start1.is_within_bounds(xmax, ymax) and goal1.is_within_bounds(xmax, ymax)\
           and start2.is_within_bounds(xmax, ymax) and goal2.is_within_bounds(xmax, ymax) ):
        raise ValueError(f"Point out of bounds.")

    obstacles: list[Rectangle] = []
    for i in range(0, len(obstacle_tokens), 4):
        ob = Rectangle(*obstacle_tokens[i : i + 4])
        if not ob.is_within_bounds(xmax, ymax):
            raise ValueError(f"Obstacle {i//4} is out of bound.")
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


def display_environment(problem: Problem, path: list[Point] = []):
    _, ax = plt.subplots()
    ax.set_xlim(0, problem.xmax)
    ax.set_ylim(0, problem.ymax)
    ax.set_aspect("equal")

    # Obstacles
    for obs in problem.obstacles:
        ax.add_patch(MplRectangle((obs.x, obs.y), obs.width, obs.height, color="black"))

    # Start and goal points for 1st robot
    ax.plot(problem.start1.x, problem.start1.y, "ro", label="start", markersize=10, clip_on=False, zorder=3)
    ax.plot(problem.goal1.x, problem.goal1.y, "r*", label="goal", markersize=10, clip_on=False, zorder=3)

    # Start and goal points for 2nd robot
    ax.plot(problem.start2.x, problem.start2.y, "go", label="start2", markersize=10, clip_on=False, zorder=3)
    ax.plot(problem.goal2.x, problem.goal2.y, "g*", label="goal2", markersize=10, clip_on=False, zorder=3)

    # Path
    if path:
        ax.plot([p.x for p in path], [p.y for p in path], "b-", label="path", linewidth=0.5)

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.show()
