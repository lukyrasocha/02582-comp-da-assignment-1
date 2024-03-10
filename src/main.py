from src.preprocess import preprocess
from src.util import load_data

# Linear Regression Models
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

def main():
    data = load_data()

    K = 10
    kfold = KFold(n_splits=K, shuffle=True)

    # Linear Regression
    models = {
        "LinearRegression": LinearRegression(),
        "Lasso": Lasso(alpha=1.0),
        "Ridge": Ridge(alpha=1.0)
    }

    # Dictionary to store the scores
    scores = {model_name: [] for model_name in models.keys()}

    for model_name, model in models.items():
        for train_index, test_index in kfold.split(data):
            # Split data (pre-preprocessing) into training and testing based on current fold
            train_data, test_data = data.iloc[train_index], data.iloc[test_index]

            # Apply preprocessing to the current fold's training and test data to avoid leakage
            X_train, y_train = preprocess(train_data)
            template_columns = list(X_train.columns)
            X_test, y_test = preprocess(test_data, template_columns=template_columns)

            X_train = X_train.to_numpy()
            X_test = X_test.to_numpy()
            y_train = y_train.to_numpy()
            y_test = y_test.to_numpy()

            # Fit model to training data
            model.fit(X_train, y_train)

            # Make predictions on test data
            predictions = model.predict(X_test)

            # Calculate RMSE and append to scores
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            scores[model_name].append(rmse)

        # Calculate and print average RMSE over all folds
        avg_rmse = np.mean(scores[model_name])
        print(f"{model_name}: Average RMSE across {K} folds: {avg_rmse}")

    if __name__ == '__main__':
        main()

if __name__ == '__main__':
    main()
