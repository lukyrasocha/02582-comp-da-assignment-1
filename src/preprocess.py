import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from util import load_data


def preprocess(
    data: pd.DataFrame,
    impute_strategy: str = 'mean'):
    """
    Preprocess the data
    :param data: pandas.DataFrame (original data format from load_data)
    :param impute_strategy: str (mean, median)
    """

    data.replace(to_replace=r'^\s*NaN\s*$', value=np.nan, regex=True, inplace=True)

    # Cast numerical values to float
    data.iloc[:, :-5] = data.iloc[:, :-5].astype(float)

    X = data.drop(['y', ' C_ 2'], axis=1)
    y = data['y']

    # Define numerical and categorical features
    numerical_features = X.iloc[:, :-4].columns.tolist()
    categorical_features = [" C_ 1", " C_ 3", " C_ 4", " C_ 5"]

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
    print(X.head())
    print(list(X.columns))
    print(y)
    # Try the preprocessor
    X = preprocessor.fit_transform(X)
    print(X.shape)
