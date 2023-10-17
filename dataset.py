import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons, make_circles, make_blobs

def get_2d_dataset(dataset_name, n_samples=1000, noise=0.3):
    """
    Returns a 2D dataset based on the provided dataset name.

    Parameters:
    - dataset_name (str): Name of the dataset ('moons', 'circles', or 'blobs').
    - n_samples (int): Number of samples.
    - noise (float): Noise level.

    Returns:
    - X1 (numpy array): Data points for group 1.
    - y1 (numpy array): Labels for group 1 (all zeros).
    - X2 (numpy array): Data points for group 2.
    - y2 (numpy array): Labels for group 2 (all ones).
    """
    
    if dataset_name == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=noise)
    elif dataset_name == 'circles':
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5)
    elif dataset_name == 'blobs':
        X, y = make_blobs(n_samples=n_samples, centers=2)
    else:
        raise ValueError("Invalid dataset name. Choose from 'moons', 'circles', or 'blobs'.")
    
    # Separate the data points based on the labels
    X1 = X[y == 0]
    y1 = y[y == 0]
    X2 = X[y == 1]
    y2 = y[y == 1]
    
    return X, y, X1, y1, X2, y2


if __name__ == '__main__':
    X1, y1, X2, y2 = get_2d_dataset('moons')
    print(X1.shape, y1.shape, X2.shape, y2.shape)

    # Plotting the data
    sns.scatterplot(x=X1[:, 0], y=X1[:, 1], color='blue', label='Group 0')
    sns.scatterplot(x=X2[:, 0], y=X2[:, 1], color='red', label='Group 1')
    plt.legend()
    plt.show()
