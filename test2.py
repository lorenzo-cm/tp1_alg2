import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_digits, load_breast_cancer


from utils import *
from graham_scan import graham_scan
from sweep_line_intersection import sweep_line_intersection
from dataset import get_dataset
from linear_separation import linear_separation
from linear_classification import linear_classification
from svm import classify_points_svm
from pca import apply_pca

def load_dataset(input_data):
    # Check if the input is a string (filename)
    if isinstance(input_data, str):
        with open(input_data, 'r') as file:
            data = file.read()
            return data
    
    # Check if the input is a function
    elif callable(input_data):
        return input_data()


if __name__ == '__main__':

    time_hull = time_intersection = time_linear_sep = time_linear_class = 0
    
    datasets = {
        "iris": load_iris,
        "wine": load_wine,
        "digits": load_digits,
        "breast_cancer": load_breast_cancer
    }

    for d_name, d in datasets.items():

        print(f"Dataset: {d_name}")

        data = load_dataset(d)

        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target

        # verify if it is a binary classification problem
        unique_classes = np.unique(df['target'])
        if unique_classes.size > 2:
            # randomly select 2 classes
            selected_classes = np.random.choice(unique_classes, 2, replace=False)

            filtered_df = df[df['target'].isin(selected_classes)]

        if df.shape[1] > 3: # 3 because of target
            # Extract data and labels from the filtered dataframe
            X_filtered = filtered_df.drop('target', axis=1).values
            y_filtered = filtered_df['target'].values

            # Apply PCA and visualize the result
            reduced_data = apply_pca(X_filtered, y_filtered, target_names=selected_classes)

        reduced_df = pd.DataFrame(reduced_data, columns=['x', 'y'])
        reduced_df['target'] = y_filtered

        reduced_df = reduced_df.sample(frac=0.2).reset_index(drop=True)

        X1 = reduced_df[reduced_df['target'] == selected_classes[0]][['x', 'y']]
        X2 = reduced_df[reduced_df['target'] == selected_classes[1]][['x', 'y']]

        # making the hull for each
        hull1, time1_ = graham_scan([Point(row['x'], row['y']) for _, row in X1.iterrows()])
        hull2, time2_ = graham_scan([Point(row['x'], row['y']) for _, row in X2.iterrows()])

        time_hull += time1_
        time_hull += time2_

        # making the segments for each hull
        hull1_segments = hull_to_segments(hull1)
        hull2_segments = hull_to_segments(hull2)

        # make the sweep line intersection
        intersection, time_ = sweep_line_intersection(hull1_segments, hull2_segments)

        time_intersection += time_

        if intersection:
            print("Intersects\n")
            continue
        else:
            print("Not intersects")

        # ax + by + c = 0
        (a, b, c, mx, my), time_ = linear_separation(hull1, hull2)

        time_linear_sep += time_
        
        X = reduced_df[['x', 'y']].values
        y = reduced_df['target'].values

        acc_svm, precision_svm, recall_svm, f1_svm, abc_svm = classify_points_svm(X, y, plot_boundary=False)

        (acc, precision, recall, f1), time_ = linear_classification(X, y, a, c, selected_classes[0], selected_classes[1])


        time_linear_class += time_

        print(f"METRICS: Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

        print(f"METRICS SVM: Accuracy: {acc_svm}, Precision: {precision_svm}, Recall: {recall_svm}, F1 Score: {f1_svm}")

        # Plotting the data
        if not intersection:
            plot_grid_hulls_separation(X, y, hull1, hull2, intersection, (a,b,c), abc_svm, save=True, filename=f"graphs/dataset_{d_name}.png")

        print()
        
    print(f"Time hull: {time_hull}")
    print(f"Time intersection: {time_intersection}")
    print(f"Time linear separation: {time_linear_sep}")
    print(f"Time linear classification: {time_linear_class}")
