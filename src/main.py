from sklearn.pipeline import Pipeline
from preprocess import preprocess
from util import load_data
from train import train
from visualize import visualize_results, visualize_coefficients, visualize_residuals

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    # Load data and preprocessor
    data = load_data()
    X, y, preprocessor = preprocess(data)

    # Define models to be trained
    models = {}
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
    OUTER_K = 10
    INNER_K = 5

    results = train(X, y, preprocessor, models, OUTER_K, INNER_K)

    visualize_results(results)
    visualize_coefficients(results)

    # Take the best model for each method and visualize the residuals
    for method in results:
        y_pred = results[method]["y_pred"] # Predictions from all folds together
        y_true = results[method]["y_true"] # True values from all folds together
        visualize_residuals(y_true, y_pred, method)




if __name__ == '__main__':
    main()
