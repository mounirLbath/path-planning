from geometry import Point, Rectangle


def distance(a: Point, b: Point) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def dot(a: Point, b: Point) -> float:
    return a.x * b.x + a.y * b.y


def segment_intersects_rect(a: Point, b: Point, r: Rectangle) -> bool:
    if a == b:
        raise ValueError("Checking intersection for a segment of length 0")
    # quick reject by bounding boxes
    if max(a.x, b.x) < r.x or min(a.x, b.x) > r.x + r.width:
        return False
    if max(a.y, b.y) < r.y or min(a.y, b.y) > r.y + r.height:
        return False

    # quick accept if either endpoint is inside rectangle
    if r.contains_point(a) or r.contains_point(b):
        return True

    # For the case of colinearity between ab and edges :
    # If ab is axis-aligned: either it is outside the rectangle (rejected by bounding box) or it intersects the rectangle
    if a.x == b.x or a.y == b.y:
        return True

    # Normal vector to ab
    normal = Point(b.y - a.y, a.x - b.x)
    for p1, p2 in r.edges(): # p1 and p2 are the 2 ends of the edge
        # We do the full check now that we have no colinearity: a and b are opposite sides of edge p1p2 and p1 and p2 are opposite sides of ab
        normal_r = Point(p2.y - p1.y, p1.x - p2.x)
        if dot(normal, p1 - a) * dot(normal, p2 - a) <= 0 and dot(normal_r, a - p1) * dot(normal_r, b - p1) <= 0:
            return True

    return False


def segment_collision(a: Point, b: Point, obstacles: list[Rectangle]) -> bool:
    for obs in obstacles:
        if segment_intersects_rect(a, b, obs):
            return True
    return False

