import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from src.util import load_data


def preprocess(
    data: pd.DataFrame,
    impute_strategy: str = 'mean'):
    """
    Preprocess the data
    :param data: pandas.DataFrame (original data format from load_data)
    :param impute_strategy: str (mean, median)
    """

    pd.set_option('future.no_silent_downcasting', True)
    data.replace(to_replace=r'^\s*NaN\s*$', value=np.nan, regex=True, inplace=True)

    # Cast numerical values to float
    data.iloc[:, :-5] = data.iloc[:, :-5].astype(float)

    X = data.drop('y', axis=1)
    y = data['y']

    # Define numerical and categorical features
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'bool']).columns

    # Define preprocessing for numerical features: imputation + standardization
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=impute_strategy)),
        ('scaler', StandardScaler())])

    # Define preprocessing for categorical features: imputation + one-hot encoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)])


    return X, y, preprocessor

if __name__ == '__main__':
    data = load_data()
    X, y, preprocessor = preprocess(data)
    print(X)
    print(y)
