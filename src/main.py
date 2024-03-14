from preprocess import preprocess
from util import load_data

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    # Load and preprocessor
    data = load_data()
    X, y, preprocessor = preprocess(data)

    models = {}
    models['LinearRegression'] = [{'parameters':{}, 'model': LinearRegression(), 'scores': [], 'r2': []}]
    models['Ridge'] = []
    models['Lasso'] = []
    models['ElasticNet'] = []
    models['DecisionTree'] = []
    models['RandomForest'] = []
    models['AdaBoost'] = []
    models['LDA'] = []
    models['QDA'] = []
    models['KNN'] = []

    # Ridge
    lambda_values = np.logspace(-3, 3, 13)
    for lambda_value in lambda_values:
        models['Ridge'].append({'parameters': {'regressor_alpha': lambda_value},
                                'model': Ridge(alpha=lambda_value),
                                'scores': [],
                                'r2': []})
    # Lasso
    lambda_values = np.logspace(-1, 3, 5)
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

    models["RandomForest"] = []

    # Elastic net
    #alpha_values = np.logspace(0, 3, 13)
    alpha_values =[10] #np.arange(0, 1, 0.1)
    l1_ratio_values = np.linspace(0, 1, 11)
    for alpha in alpha_values:
        for l1_ratio in l1_ratio_values:
            models['ElasticNet'].append({'parameters': {'regressor_alpha': alpha, 'l1_ratio': l1_ratio},
                                        'model': ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000),
                                        'scores': [],
                                        'r2': []})

    K = 10
    kfold = KFold(n_splits=K, shuffle=True, random_state=42)

    # Cross Validation for each model
    for method in models:
        print("="*80)
        print(f'Running cross-validation for {method}')
        for model in models[method]:
            pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('regressor', model["model"])])
            for train_index, test_index in kfold.split(X):
                # Split data into training and validation sets for this fold
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                # Fit the model pipeline on the training data
                pipeline.fit(X_train, y_train)

                # Predict on the validation set
                y_pred = pipeline.predict(X_test)

                # Calculate RMSE for this fold and append to scores
                score = np.sqrt(mean_squared_error(y_test, y_pred))
                model["scores"].append(score)
                model["r2"].append(pipeline.score(X_test, y_test))

            # Print the average RMSE across all folds
            print(f'Average RMSE for {method} with parameters: {model["parameters"]}: {np.mean(model["scores"])}')
            print(f'Average R2 for {method} with parameters: {model["parameters"]}: {np.mean(model["r2"])}')

    # Select the best model for each method
    best_models = {}
    for method in models:
        if not models[method]:
            continue
        best_model = min(models[method], key=lambda x: np.mean(x["scores"]))
        best_models[method] = best_model
        print(f'Best model for {method}: {best_model["model"]} with parameters: {best_model["parameters"]}')


    for method in best_models:
        print(method)
        print("Parameters:", best_models[method]["parameters"])
        print(f"RMSE: {np.mean(best_models[method]['scores'])}")

    # Barchart of RMSE for the best model of each method
    fig, ax = plt.subplots()
    ax.bar(best_models.keys(),
           [np.mean(best_models[method]["scores"]) for method in best_models])
    ax.set_ylabel('RMSE')
    ax.set_title('RMSE for the best model of each method')
    plt.show()

    # Barchart of R2 for the best model of each method
    fig, ax = plt.subplots()
    ax.bar(best_models.keys(),
              [np.mean(best_models[method]["r2"]) for method in best_models])
    ax.set_ylabel('R2')
    ax.set_title('R2 for the best model of each method')
    plt.show()


if __name__ == '__main__':
    main()
