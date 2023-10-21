from utils import Point, timer
import numpy as np

# does not need to be super fast because it is used in hulls
def find_closest_points(hull1, hull2):
    min_distance = float('inf')
    closest_pair = (None, None)

    for p1 in hull1:
        for p2 in hull2:
            distance = np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
            if distance < min_distance:
                min_distance = distance
                closest_pair = (p1, p2)

    return closest_pair

@timer
def linear_separation(hull1, hull2):
    p1, p2 = find_closest_points(hull1, hull2)

    # midpoint
    mx = (p1.x + p2.x) / 2
    my = (p1.y + p2.y) / 2

    # achar reta perpendicular = -1/m da reta que passa por p1 e p2

    if p2.x != p1.x:
        # coeficiente angular da reta que passa por p1 e p2
        m = (p2.y - p1.y) / (p2.x - p1.x)

        # coeficiente angular da perpendicular
        m_perpendicular = -1/m
    else:
        # se o x for igual, a reta é vertical (coeficiente angular zero)
        m_perpendicular = 0

    # ax + by + c = 0
    # ax - y + c = 0
    # ax + c = y
    # y = ax + c
    a = m_perpendicular
    b = -1
    c = my - m_perpendicular * mx

    return a, b, c, mx, my
