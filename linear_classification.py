from utils import timer
import numpy as np
import matplotlib.pyplot as plt

@timer
def linear_classification(X, y, a, b, class1, class2, recursive=False):
    
    equation = lambda x: a*x + b
    labels = []
    TP, TN, FP, FN = 0, 0, 0, 0

    for i, (x1, x2) in enumerate(X):
        if x2 >= equation(x1):
            labels.append(class1)
        else:
            labels.append(class2)

        if labels[i] == y[i] and labels[i] == class1:
            TP += 1
        elif labels[i] == y[i] and labels[i] == class2:
            TN += 1
        elif labels[i] != y[i] and labels[i] == class1:
            FP += 1
        else:
            FN += 1

    accuracy = (TP + TN) / (TP + TN + FP + FN)

    if accuracy <= 0.5 and not recursive:
        return linear_classification(X, y, a, b, class2, class1, recursive=True, use_timer=False)

    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return accuracy, precision, recall, f1_score
