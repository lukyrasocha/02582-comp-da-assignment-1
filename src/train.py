from preprocess import preprocess
from util import load_data

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import  Lasso
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def train(
    X: pd.DataFrame,
    y,
    preprocessor: ColumnTransformer,
    models: dict,
    OUTER_K=10,
    INNER_K=5
):
    """
    Train models using nested cross-validation
    Args:
        X: pd.DataFrame
            Features (original format)
        y: pd.DataFrame
            Target (original format)
        preprocessor: ColumnTransformer
            Preprocessor to transform the features
        models: dict
            Dictionary with the models to be trained
        OUTER_K: int
            Number of folds for the outer cross-validation
        INNER_K: int
            Number of folds for the inner cross-validation
    """

    results = {method: {"scores": [], "r2": [], "model":[], "y_pred": [], "y_true":[]} for method in models}

    f = open("results.txt", "w")

    outer_kfold = KFold(n_splits=OUTER_K, shuffle=True, random_state=42)
    for method in models:
        print("="*80)
        print(f'Running nested cross-validation for {method}')

        f.write("="*80 + "\n")
        f.write(f'Running nested cross-validation for {method}\n')

        outer_counter = 0

        # Outer Loop
        for train_index, test_index in outer_kfold.split(X):
            outer_counter += 1
            # Split data into outer training and test sets
            X_outer_train, X_outer_test = X.iloc[train_index], X.iloc[test_index]
            y_outer_train, y_outer_test = y.iloc[train_index], y.iloc[test_index]

            best_score = np.inf
            best_model = models[method][0] # Initialize with the first model

            # Inner Loop: Iterate over parameter sets for the current model
            for model in models[method]:
                inner_scores = []

                # Define the inner K-fold split
                inner_kfold = KFold(n_splits=INNER_K, shuffle=True, random_state=42)

                # Inner CV for parameter tuning
                for inner_train_index, inner_test_index in inner_kfold.split(X_outer_train):
                    # Split data into inner training and validation sets
                    X_inner_train, X_inner_test = X_outer_train.iloc[inner_train_index], X_outer_train.iloc[inner_test_index]
                    y_inner_train, y_inner_test = y_outer_train.iloc[inner_train_index], y_outer_train.iloc[inner_test_index]

                    # Update model with current parameters
                    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                               ('regressor', model["model"])])

                    # Train and validate the model
                    pipeline.fit(X_inner_train, y_inner_train)
                    y_inner_pred = pipeline.predict(X_inner_test)
                    score = np.sqrt(mean_squared_error(y_inner_test, y_inner_pred))
                    inner_scores.append(score)

                # Average RMSE over the inner folds
                average_inner_score = np.mean(inner_scores)

                # If this parameter set is the best so far, save it
                if average_inner_score < best_score:
                    best_score = average_inner_score
                    best_model = model

            # Train the best model on the entire outer training set and evaluate on the test set
            best_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                            ('regressor', best_model["model"])])
            best_pipeline.fit(X_outer_train, y_outer_train)
            y_outer_pred = best_pipeline.predict(X_outer_test)

            # Calculate and store the scores for the outer loop
            outer_score = np.sqrt(mean_squared_error(y_outer_test, y_outer_pred))
            outer_r2_score = r2_score(y_outer_test, y_outer_pred)

            results[method]["scores"].append(outer_score)
            results[method]["r2"].append(outer_r2_score)
            results[method]["model"].append(best_model["model"])
            results[method]["y_pred"] += list(y_outer_pred)
            results[method]["y_true"] += list(y_outer_test)

            print(f"Best model for outer loop {outer_counter} for {method} has parameters: {best_model['parameters']}")
            print(f"RMSE for outer loop {outer_counter} for {method}: {outer_score}")
            print(f"R2 for outer loop {outer_counter} for {method}: {outer_r2_score}")

            # Write the results to the file
            f.write(f"Best model for outer loop {outer_counter} for {method} has parameters: {best_model['parameters']}\n")
            f.write(f"RMSE for outer loop {outer_counter} for {method}: {outer_score}\n")
            f.write(f"R2 for outer loop {outer_counter} for {method}: {outer_r2_score}\n")


        # Print the average RMSE and R2 across all outer folds for the current model
        print(f'Nested CV Average RMSE (Generalization error) for {method}: {np.mean(results[method]["scores"])}')
        print(f'Nested CV Average R2 for {method} with best parameters: {np.mean(results[method]["r2"])}')

        # Write the results to the file
        f.write(f'Nested CV Average RMSE (Generalization error) for {method}: {np.mean(results[method]["scores"])}\n')
        f.write(f'Nested CV Average R2 for {method} with best parameters: {np.mean(results[method]["r2"])}\n')

    f.close()
    return results

if __name__ == '__main__':
    data = load_data()
    X, y, preprocessor = preprocess(data)

    models = {}
    models['LinearRegression'] = [{'parameters':{}, 'model': LinearRegression(), 'scores': [], 'r2': []}]
    models['Ridge'] = []
    models['Lasso'] = []
    models['ElasticNet'] = []
    models['RandomForest'] = []

    # Ridge
    lambda_values = np.logspace(-3, 3, 13)
    for lambda_value in lambda_values:
        models['Ridge'].append({'parameters': {'regressor_alpha': lambda_value},
                                'model': Ridge(alpha=lambda_value),
                                'scores': [],
                                'r2': []})

    # Lasso
    lambda_values = np.logspace(0, 3, 13)
    for lambda_value in lambda_values:
        models['Lasso'].append({'parameters': {'regressor_alpha': lambda_value},
                                'model': Lasso(alpha=lambda_value),
                                'scores': [],
                                'r2': []})

    # Random Forest
    n_estimators = [10, 50, 100, 200]
    max_depth = [5, 10, 20, 50, 200]
    for n in n_estimators:
        for d in max_depth:
            models['RandomForest'].append({'parameters': {'n_estimators': n, 'max_depth': d},
                                           'model': RandomForestRegressor(n_estimators=n, max_depth=d),
                                           'scores': [],
                                            'r2': []})
    # Elastic net
    alpha_values = np.logspace(0, 3, 13)
    l1_ratio_values = np.linspace(0.001, 1, 15)
    for alpha in alpha_values:
        for l1_ratio in l1_ratio_values:
            models['ElasticNet'].append({'parameters': {'regressor_alpha': alpha, 'l1_ratio': l1_ratio},
                                        'model': ElasticNet(alpha=alpha, l1_ratio=l1_ratio),
                                        'scores': [],
                                        'r2': []})
    # Nested cross-validation
    OUTER_K = 10
    INNER_K = 5

    results = train(X, y, preprocessor, models, OUTER_K, INNER_K)

    print(results)
