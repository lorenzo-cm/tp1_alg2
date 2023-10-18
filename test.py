import numpy as np

from utils import *
from graham_scan import graham_scan
from sweep_line_intersection import sweep_line_intersection
from dataset import get_dataset
from linear_separation import linear_separation
from linear_classification import linear_classification
from svm import classify_points_svm

@timer
def time_get_dataset(*args, **kwargs):
    return get_dataset(*args, **kwargs)

@timer
def time_graham_scan(*args, **kwargs):
    return graham_scan(*args, **kwargs)

@timer
def time_hull_to_segments(*args, **kwargs):
    return hull_to_segments(*args, **kwargs)

@timer
def time_sweep_line_intersection(*args, **kwargs):
    return sweep_line_intersection(*args, **kwargs)

@timer
def time_linear_separation(*args, **kwargs):
    return linear_separation(*args, **kwargs)

@timer
def time_linear_classification(*args, **kwargs):
    return linear_classification(*args, **kwargs)


if __name__ == '__main__':
    # circulo é um problema, pois tem o caso em que um
    # circulo está dentro do outro
    # desse modo não há interseção
    # mas os pontos coexistem no mesmo espaço
    X, y, X1, y1, X2, y2 = get_dataset('blobs', n_samples=10000, noise=0.5)
    
    # making the hull for each
    hull1 = time_graham_scan([Point(x, y) for x, y in X1])
    hull2 = time_graham_scan([Point(x, y) for x, y in X2])

    # making the segments for each hull
    hull1_segments = time_hull_to_segments(hull1)
    hull2_segments = time_hull_to_segments(hull2)

    # make the sweep line intersection
    intersection = time_sweep_line_intersection(hull1_segments, hull2_segments)

    if intersection:
        print("intersects")
        exit(0)
    else:
        print("not intersects")


    # ax + by + c = 0

    a, b, c, mx, my = time_linear_separation(hull1, hull2)
    
    acc_svm, precision_svm, recall_svm, f1_svm, abc_svm = classify_points_svm(X, y, plot_boundary=False)

    acc, precision, recall, f1 = time_linear_classification(X, y, a, c, mx, my)

    print(f"METRICS: Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

    # Plotting the data
    if acc < 1 and not intersection:
        plot_grid_hulls_separation(X, y, hull1, hull2, intersection, (a,b,c), abc_svm)

