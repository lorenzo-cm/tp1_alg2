import matplotlib.pyplot as plt
import seaborn as sns
import time

anchor:int

class Point:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return f'({self.x},{self.y})'
    
    def __eq__(self, p1) -> bool:
        return True if (self.x == p1.x and self.y == p1.y) else False
    
    def __hash__(self):
        # Calculate the hash based on the x and y attributes
        return hash((self.x, self.y))
    

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


def hull_to_segments(hull: list[Point]) -> list[Segment]:
    segments = []
    for i in range(len(hull)):
        segments.append(Segment(hull[i], hull[(i+1) % len(hull)]))
    return segments


def plot_grid_hulls_separation(X, y, 
                               hull1: list[Point], hull2: list[Point], 
                               intersection: bool, 
                               abc: tuple, abc_svm: tuple, 
                               save=False, filename=""):

    fig, axes = plt.subplots(1, 2, figsize=(10, 10))

    sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, ax=axes[0])
    axes[0].set_title("Dataset")

    # Plot the points of both hulls
    sns.scatterplot(x=[p.x for p in hull1], y=[p.y for p in hull1], ax=axes[1], label='Hull 1')
    sns.scatterplot(x=[p.x for p in hull2], y=[p.y for p in hull2], ax=axes[1], label='Hull 2')

    # Plot the lines of both hulls
    axes[1].plot([p.x for p in hull1] + [hull1[0].x], [p.y for p in hull1] + [hull1[0].y])  # Close the hull1
    axes[1].plot([p.x for p in hull2] + [hull2[0].x], [p.y for p in hull2] + [hull2[0].y])  # Close the hull2

    # Plot the line of separation
    if abc and not intersection:
        a, b, c = abc
        
        # Centralize the line of separation
        x_min = min(p.x for p in hull1 + hull2)
        x_max = max(p.x for p in hull1 + hull2)
        
        # Shorten the line of separation
        margin = 0.05 * (x_max - x_min)
        x_min -= margin
        x_max += margin
        
        # the X values for the line of separation
        x_vals = [x_min, x_max]
        
        if b != 0:
            y_vals = [(-a*xi - c) / b for xi in x_vals]
        else:
            y_vals = [(-c / a) for _ in x_vals]
        axes[1].plot(x_vals, y_vals, '-r', label='Requested model separating Line')
    
    # plot svm line of separation
    if abc and not intersection:
        a, b, c = abc_svm
        
        # Centralize the line of separation
        x_min = min(p.x for p in hull1 + hull2)
        x_max = max(p.x for p in hull1 + hull2)
        
        # Shorten the line of separation
        margin = 0.05 * (x_max - x_min)
        x_min -= margin
        x_max += margin
        
        # the X values for the line of separation
        x_vals = [x_min, x_max]
        
        if b != 0:
            y_vals = [(-a*xi - c) / b for xi in x_vals]
        else:
            y_vals = [(-c / a) for _ in x_vals]
        axes[1].plot(x_vals, y_vals, '-g', label='SVM best separating Line')

    axes[1].set_title("Both Hulls")
    axes[1].legend()

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    if save:
        plt.savefig(filename)


def timer(funct):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # Begin the timer
        result = funct(*args, **kwargs)  # Call the actual function
        end_time = time.perf_counter()  # End the timer
        elapsed_time = end_time - start_time
        return result, elapsed_time
    return wrapper