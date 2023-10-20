import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_kddcup99, fetch_covtype, fetch_rcv1, fetch_lfw_people, fetch_olivetti_faces, fetch_20newsgroups_vectorized
from sklearn.preprocessing import LabelEncoder

def apply_pca(data, labels, target_names=None, visualize=True):
    """
    Apply PCA to reduce the dataset to 2 components.
    
    Parameters:
    - data: The dataset to be reduced.
    - labels: The labels or classes of the dataset.
    - target_names: Names of the classes for visualization.
    - visualize: Boolean indicating if the data should be visualized.
    
    Returns:
    - 2D reduced dataset.
    """

    # Apply PCA and reduce the dataset to 2 components
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)

    # Visualization
    if visualize:
        plt.figure(figsize=(10, 6))
        
        # If target names are not provided, use unique labels
        if target_names is None:
            target_names = np.unique(labels)
        
        colors = plt.cm.rainbow(np.linspace(0, 1, len(target_names)))
        
        for i, (color, target) in enumerate(zip(colors, target_names)):
            plt.scatter(data_pca[labels == i, 0], data_pca[labels == i, 1], color=color, label=target, edgecolor='k')
        
        plt.legend()
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('2D PCA of Dataset')
        plt.show()

    return data_pca

if __name__ == '__main__':
    data_obj = fetch_kddcup99(subset='SA')  # fetching a smaller subset for quicker execution
    labels_enc = LabelEncoder().fit_transform(data_obj.target)
    target_names = np.unique(data_obj.target).astype(str)
    reduced_data = apply_pca(data_obj.data, labels_enc, target_names, visualize=True)
