import numpy as np

from utils import *
from graham_scan import graham_scan
from sweep_line_intersection import sweep_line_intersection
from dataset import get_2d_dataset
from find_linear_separation import find_linear_separation


if __name__ == '__main__':
    # circulo é um problema, pois tem o caso em que um
    # circulo está dentro do outro
    # desse modo não há interseção
    # mas os pontos coexistem no mesmo espaço
    X, y, X1, y1, X2, y2 = get_2d_dataset('blobs', n_samples=10000, noise=0.5)
    
    # making the hull for each
    hull1 = graham_scan([Point(x, y) for x, y in X1])
    hull2 = graham_scan([Point(x, y) for x, y in X2])

    # making the segments for each hull
    hull1_segments = hull_to_segments(hull1)
    hull2_segments = hull_to_segments(hull2)

    # make the sweep line intersection
    intersection = sweep_line_intersection(hull1_segments, hull2_segments)

    if intersection:
        print("Hull1 intersects with Hull2!")
    else:
        print("Hull1 does not intersect with Hull2!")

    abc = find_linear_separation(hull1, hull2)

    # Plotting the data
    plot_grid_hulls_separation(X, y, hull1, hull2, intersection, abc)

