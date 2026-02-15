from geometry import Point, Rectangle


def distance(a: Point, b: Point) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def dot(a: Point, b: Point) -> float:
    return a.x * b.x + a.y * b.y

def segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    """
    Check if the segments [a,b] and [c,d] intersect
    """
    normal = Point(b.y - a.y, a.x - b.x)
    normal_r = Point(d.y - c.y, c.x - d.x)
    # a and b are opposite sides of edge [c,d] and c and d are opposite sides of edge [a,b]
    return dot(normal, c - a) * dot(normal, d - a) <= 0 and dot(normal_r, a - c) * dot(normal_r, b - c) <= 0

def segment_intersects_rect(a: Point, b: Point, r: Rectangle, strict: bool = False) -> bool:
    # quick reject by bounding boxes
    if not strict:
        if max(a.x, b.x) < r.x or min(a.x, b.x) > r.x + r.width:
            return False
        if max(a.y, b.y) < r.y or min(a.y, b.y) > r.y + r.height:
            return False
    else:
        if max(a.x, b.x) <= r.x or min(a.x, b.x) >= r.x + r.width:
            return False
        if max(a.y, b.y) <= r.y or min(a.y, b.y) >= r.y + r.height:
            return False

    # quick accept if either endpoint is inside rectangle (probably likely to happen in the case of an intersection)
    if r.contains_point(a, strict=strict) or r.contains_point(b, strict=strict):
        return True

    # For the case of colinearity between ab and edges :
    # If ab is axis-aligned: either it is outside the rectangle (rejected by bounding box) or it intersects the rectangle
    if a.x == b.x or a.y == b.y:
        return True

    # We do the full check now that we have no colinearity: 
    for p1, p2 in r.edges(): # p1 and p2 are the 2 ends of the edge
        if segments_intersect(a, b, p1, p2):
            return True

    return False


def segment_collision(a: Point, b: Point, obstacles: list[Rectangle], strict: bool = False) -> bool:
    # If strict is True, we still reject points in 2 rectangles, since the path is of width 0 here.
    for obs in obstacles:
        if segment_intersects_rect(a, b, obs, strict=strict):
            return True
    return False