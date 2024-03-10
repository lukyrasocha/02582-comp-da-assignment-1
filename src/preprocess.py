import pandas as pd
import numpy as np
from sklearn.externals._packaging.version import List
from sklearn.preprocessing import StandardScaler
from src.util import load_data


def preprocess(
    data: pd.DataFrame,
    template_columns: List[str] = [],
    impute_strategy: str = 'mean',
    standardize: bool = True,
    one_hot_encode: bool = True):
    """
    Preprocess the data
    :param data: pandas.DataFrame (original data format from load_data)
    :param impute_strategy: str (mean, median)
    :param standardize: bool
    :param one_hot_encode: bool
    """

    # Cast numerical values to float
    data.iloc[:, :-5] = data.iloc[:, :-5].astype(float)

    # Separate the target variable before preprocessing
    y = data['y']
    data = data.drop('y', axis=1)

    # Replace missing values with the numpy's missing value code
    pd.set_option('future.no_silent_downcasting', True)
    data.replace(to_replace=r'^\s*NaN\s*$', value=np.nan, regex=True, inplace=True)

    numerical_features = data.select_dtypes(include=['float64']).columns

    # Impute missing values with the mean or median of the column for numerical values
    if impute_strategy == 'mean':
        data[numerical_features] = data[numerical_features].apply(lambda x: x.fillna(x.mean()), axis=0)
    elif impute_strategy == 'median':
        data[numerical_features] = data[numerical_features].apply(lambda x: x.fillna(x.median()), axis=0)

    # Since C_ 2 has only 1 value, it does not help with the prediction, so we drop it
    data = data.drop(' C_ 2', axis=1)
    categorical_features = data.select_dtypes(include=['object']).columns

    # Impute missing values with the mode of the column for categorical values
    data[categorical_features] = data[categorical_features].apply(lambda x: x.fillna(x.mode()[0]), axis=0)

    # One-hot encode the categorical variables
    if one_hot_encode:
        data = pd.get_dummies(data, columns=categorical_features)

    if standardize:
        scaler = StandardScaler()
        data[numerical_features] = scaler.fit_transform(data[numerical_features])

    X = data.to_numpy()
    return X, y


if __name__ == '__main__':
    data = load_data()
    X, y = preprocess(data)
    print(X)
    print(y)
