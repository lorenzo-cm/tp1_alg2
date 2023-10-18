def linear_classification(X, y, a, b, mx, my):
    """
    X: List of tuples containing points (x, y)
    y: List of labels (0 or 1) for each point in X
    a, b: Parameters of the line y = ax + b
    Returns: accuracy, precision, recall, f1_score
    """
    
    if X.shape[0] != y.shape[0]:
        raise Exception("X and y must have the same length")
    
    if X.shape[1] != 2:
        raise Exception("X must be a list of tuples containing points (x, y)")
    
    TP = 0  # True Positives
    FP = 0  # False Positives
    TN = 0  # True Negatives
    FN = 0  # False Negatives

    # direction such that the label is 1
    direction = ""
    if y[0] == 1 and X[0][1] < a * X[0][0] + b:
        direction = "below"
    elif y[0] == 0 and X[0][1] >= a * X[0][0] + b:
        direction = "below"
    else:
        direction = "above"


    if direction == 'above':
        for (x_val, y_val), label in zip(X, y):
            predicted_class = 1 if y_val > a * x_val + b else 0

            if y_val == a * x_val + b:
                if y_val > my:
                    predicted_class = 1
                else:
                    predicted_class = 0

            if predicted_class == 1 and label == 1:
                TP += 1
            elif predicted_class == 1 and label == 0:
                FP += 1
            elif predicted_class == 0 and label == 0:
                TN += 1
            elif predicted_class == 0 and label == 1:
                FN += 1
    else:
        for (x_val, y_val), label in zip(X, y):
            predicted_class = 1 if y_val < a * x_val + b else 0

            if y_val == a * x_val + b:
                if y_val > my:
                    predicted_class = 0
                else:
                    predicted_class = 1
                    
            if predicted_class == 1 and label == 1:
                TP += 1
            elif predicted_class == 1 and label == 0:
                FP += 1
            elif predicted_class == 0 and label == 0:
                TN += 1
            elif predicted_class == 0 and label == 1:
                FN += 1
            
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    # To avoid division by zero
    precision = TP / (TP + FP) if TP + FP != 0 else 0
    recall = TP / (TP + FN) if TP + FN != 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    
    return accuracy, precision, recall, f1_score