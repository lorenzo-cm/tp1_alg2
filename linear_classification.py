import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def classify_points(X, y, plot_boundary=False):
    """
    Classify points using a linear SVM and return accuracy, precision, recall, and F1 score.

    Parameters:
    - X (np.array): A dataset with x, y coordinates. Shape should be (n_samples, 2).
    - y (np.array): Labels corresponding to each data point in X.
    - plot_boundary (bool): If True, plot the decision boundary.

    Returns:
    - tuple: accuracy, precision, recall, F1 score
    """
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train a linear SVM
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)

    # Predict the labels for the test set
    y_pred = clf.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    if plot_boundary:
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow', alpha=0.7)
        
        h = .02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap='rainbow', alpha=0.5)
        plt.title('Decision Boundary')
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        plt.show()

    return clf, accuracy, precision, recall, f1

if __name__ == '__main__':
    from dataset import get_2d_dataset

    X, y, X1, y1, X2, y2 = get_2d_dataset('moons', n_samples=1000, noise=0.5)

    clf, accuracy, precision, recall, f1 = classify_points(X, y, plot_boundary=True)
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')
