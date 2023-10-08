import numpy as np

from utils import *
from graham_scan import graham_scan
from sweep_line_intersection import sweep_line_intersection

if __name__ == '__main__':
    points = []

    x = np.round(np.random.rand(10) * 100, 3)
    y = np.round(np.log(np.random.rand(10)) * 10, 3)

    for i in range(len(x)):
        points.append(Point(x[i], y[i]))
    
    print(points)
    
    hull_points = graham_scan(points)
    print(hull_points)

    plot_convex_hull(points, hull_points)

    # SWEEP LINE INTERSECTION

    # generate random segments
    segments = []
    x_values = np.round(np.random.rand(10) * 100, 3)
    y_values = np.round(np.random.rand(10) * 100, 3)

    for i in range(0, len(x_values), 2):
        p1 = Point(x_values[i], y_values[i])
        p2 = Point(x_values[i+1], y_values[i+1])
        segments.append(Segment(p1, p2))

    print(sweep_line_intersection(segments))
    plot_segments(segments)
