from util import load_data

from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer

from util import load_data


def preprocess(data: pd.DataFrame, include_target=True):
    """
    Preprocess the data.
    
    :param data: pandas.DataFrame (original data format from load_data)
    :param include_target: boolean, default True. If True, expects that the data includes the target column 'y'.
    """
    data.replace(to_replace=r'^\s*NaN\s*$', value=np.nan, regex=True, inplace=True)

    # Cast numerical values to float
    data.iloc[:, :-5] = data.iloc[:, :-5].astype(float)

    if include_target:
        X = data.drop(['y', ' C_ 2'], axis=1)
        y = data['y']
    else:
        X = data.drop([' C_ 2'], axis=1)
        y = None

    # Replace the categorical missing values with a new category 'missing'
    X[' C_ 1'] = X[' C_ 1'].fillna('missing')
    X[' C_ 3'] = X[' C_ 3'].fillna('missing')
    X[' C_ 4'] = X[' C_ 4'].fillna('missing')
    X[' C_ 5'] = X[' C_ 5'].fillna('missing')

    # Define numerical and categorical features
    numerical_features = X.iloc[:, :-4].columns.tolist()
    categorical_features = [" C_ 1", " C_ 3", " C_ 4", " C_ 5"]

    # Define preprocessing for numerical features: imputation + standardization
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('imputer', KNNImputer(n_neighbors=5))])

    # Define preprocessing for categorical features: imputation + one-hot encoding
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)])

    return X, y, preprocessor

def train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor, model):
    """
    Train a model and evaluate it on a test set.
    
    Args:
        X_train, X_test, y_train, y_test: Training and test sets.
        preprocessor: Preprocessor to transform the features.
        model: The model to be trained.
    """

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', model)])
    
    pipeline.fit(X_train, y_train)
    
    # Predict and evaluate on the test set
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"Test RMSE: {rmse}")

    return pipeline

def retrain_on_full_data(X, y, preprocessor, model):
    """
    Retrain the model on the full dataset.

    Args:
        X: Full feature set.
        y: Full target set.
        preprocessor: Preprocessor to transform the features.
        model: The model to be retrained.
    """
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', model)])
    pipeline.fit(X, y)
    return pipeline

def predict_new_data(pipeline, new_data_file, output_file):
    """
    Predict using the trained pipeline on new data and write to a file.
    
    Args:
        pipeline: The trained pipeline.
        new_data_file: File path for the new data.
        output_file: File path to write the predictions.
    """
    new_X = pd.read_csv(new_data_file)  # Assuming tab-separated values
    predictions = pipeline.predict(new_X)

    # Write the predictions to a text file
    with open(output_file, 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")

    print(f"Predictions written to {output_file}")

if __name__ == '__main__':
    data = load_data()
    X, y, preprocessor = preprocess(data)

    # Split the data into 90% training and 10% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Define and train the Lasso model, then evaluate it
    lasso_model = Lasso(alpha=1.78)
    train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor, lasso_model)
    lasso_model = Lasso(alpha=1.78)
    # Retrain the model on the full dataset
    trained_pipeline = retrain_on_full_data(X, y, preprocessor, lasso_model)

    # Predict on new data and write to file
    predict_new_data(trained_pipeline, "case1Data_Xnew.txt", "predictions.txt")
