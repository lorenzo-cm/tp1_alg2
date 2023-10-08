import matplotlib.pyplot as plt

anchor:int

class Point:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return f'({self.x},{self.y})'
    
    def __eq__(self, p1) -> bool:
        return True if (self.x == p1.x and self.y == p1.y) else False
    

class Segment:
    def __init__(self, p1: Point, p2: Point):
        self.points = [p1, p2]


def orientation(p0: Point, p1: Point, p2: Point) -> int:
    val = (p1.x - p0.x) * (p2.y - p1.y) - (p1.y - p0.y) * (p2.x - p1.x)
    if val == 0: return 0  # colinear
    return 1 if val > 0 else -1  # counterclockwise or clockwise


def compare_points(p1: Point, p2: Point) -> int:
    o = orientation(anchor, p1, p2)
    if o == 0:
        # If colinear, sort by distance to anchor
        d1 = (p1.x - anchor.x)**2 + (p1.y - anchor.y)**2
        d2 = (p2.x - anchor.x)**2 + (p2.y - anchor.y)**2
        return -1 if d1 < d2 else 1
    return o


def plot_convex_hull(points: list[Point], hull_points: list[Point]):
    # Extract the x and y coordinates of the points
    x_coords = [p.x for p in points]
    y_coords = [p.y for p in points]

    hull_x = [p.x for p in hull_points]
    hull_y = [p.y for p in hull_points]

    # Plot the points
    plt.scatter(x_coords, y_coords, c='blue')

    # Plot the convex hull
    plt.plot(hull_x + [hull_x[0]], hull_y + [hull_y[0]], c='red')  # Adding the starting point to close the hull

    plt.show()


def plot_segments(segments: list[Segment]):
    for segment in segments:
        plt.plot([segment.points[0].x, segment.points[1].x], [segment.points[0].y, segment.points[1].y])

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Segments Plot')
    plt.grid(True)
    plt.show()