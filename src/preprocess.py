import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer

from util import load_data


def preprocess(data: pd.DataFrame):
    """
    Preprocess the data
    :param data: pandas.DataFrame (original data format from load_data)
    """

    data.replace(to_replace=r'^\s*NaN\s*$', value=np.nan, regex=True, inplace=True)

    # Cast numerical values to float
    data.iloc[:, :-5] = data.iloc[:, :-5].astype(float)

    # Remove all white spaces from all column names even when there are multiple spaces between words
    data.columns = data.columns.str.replace(' ', '')

    if 'y' in data.columns:
        y = data['y']
        X = data.drop(['y', 'C_2'], axis=1)
    else:
        y = None
        X = data.drop(['C_2'], axis=1)

    # Replace the categorical missing values with a new category 'missing'
    X['C_1'] = X['C_1'].fillna('missing')
    X['C_3'] = X['C_3'].fillna('missing')
    X['C_4'] = X['C_4'].fillna('missing')
    X['C_5'] = X['C_5'].fillna('missing')

    # Define numerical and categorical features
    numerical_features = X.iloc[:, :-4].columns.tolist()
    categorical_features = ["C_1", "C_3", "C_4", "C_5"]

    # Define preprocessing for numerical features: imputation + standardization
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('imputer', KNNImputer(n_neighbors=5))])

    # Define preprocessing for categorical features: imputation + one-hot encoding
    categorical_transformer = Pipeline(steps=[
        #('imputer', SimpleImputer(strategy='most_frequent')),
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
