
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
    
    def __str__(self):
        return f"Point({self.x},{self.y})"
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def clamp(self, xmax: float, ymax: float):
        return Point(max(0, min(self.x, xmax)), max(0, min(self.y, ymax)))
    
    def clamp_norm(self, max_norm : float):
        n = (self.x*self.x+self.y*self.y)**0.5
        if n > max_norm:
            return self * (max_norm/n)
        return self


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

