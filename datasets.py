import numpy as np
import pandas as pd

from sklearn.datasets import load_iris, load_wine, load_digits, load_breast_cancer, load_linnerud, load_diabetes, fetch_california_housing

def load_iris_():
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df

def load_wine_():
    data = load_wine()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df

def load_digits_():
    data = load_digits()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df

def load_breast_cancer_():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df

def load_linnerud_():
    """1 if weight of the person is above the average of the dataset, 0 otherwise"""
    data = load_linnerud()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = pd.DataFrame(data.target[:,0], columns=[data.target_names[0]])
    df['target'] = np.where(df['target'] > 178.6, 1, 0)
    return df

def load_diabetes_():
    """1 if the patient's disease progressed, 0 otherwise"""
    data = load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    df['target'] = np.where(df['target'] > 200, 1, 0)
    return df

def load_california_housing_():
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    df['target'] = df['target'].apply(lambda x: 1 if x > 2.068 else 0)
    return df


datasets = {
    "iris": load_iris_,
    "wine": load_wine_,
    "digits": load_digits_,
    "breast_cancer": load_breast_cancer_,
    "linnerud": load_linnerud_,
    "diabetes": load_diabetes_,
    "california_housing": load_california_housing_
}


