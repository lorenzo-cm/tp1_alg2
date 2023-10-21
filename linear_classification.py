from utils import timer
import numpy as np

@timer
def linear_classification(X, y, a, b, class1, class2):
    # Classify the points based on the line equation
    predictions = []
    for point in X:
        x, y = point[0], point[1]
        y_line = a * x + b
        if y > y_line:
            predictions.append(class2)
        else:
            predictions.append(class1)

    # Evaluate the predictions using the true labels
    positive_class = class2  # Treating class2 as the 'positive' class for metrics calculation
    TP = sum((np.array(y) == positive_class) & (np.array(predictions) == positive_class))
    TN = sum((np.array(y) != positive_class) & (np.array(predictions) != positive_class))
    FP = sum((np.array(y) != positive_class) & (np.array(predictions) == positive_class))
    FN = sum((np.array(y) == positive_class) & (np.array(predictions) != positive_class))
    
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if TP + FP != 0 else 0
    recall = TP / (TP + FN) if TP + FN != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
    
    return accuracy, precision, recall, f1_score