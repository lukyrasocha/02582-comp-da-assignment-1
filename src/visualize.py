from util import load_data
from preprocess import preprocess

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def visualize_correlation_matrix(X):

    # Create a correlation matrix
    X = X.drop([' C_ 1', ' C_ 3', ' C_ 4', ' C_ 5'], axis=1)
    corr = X.corr()

    # Draw the correlation matrix
    plt.figure(figsize=(8, 8))
    sns.heatmap(corr, annot=False, cmap='mako', fmt='.1f', square=True, cbar=True)
    plt.title('Correlation matrix')
    plt.savefig('../figures/correlation_matrix.png')

def visualize_results(results):
    # Calculate means and standard errors for RMSE and R^2
    data = []
    for method, values in results.items():
        rmse_mean = np.mean(values['scores'])
        rmse_se = np.std(values['scores'], ddof=1) / np.sqrt(len(values['scores']))
        r2_mean = np.mean(values['r2'])
        r2_se = np.std(values['r2'], ddof=1) / np.sqrt(len(values['r2']))
        data.append((method, rmse_mean, rmse_se, r2_mean, r2_se))

    # Create DataFrame
    df = pd.DataFrame(data, columns=['Method', 'RMSE Mean', 'RMSE SE', 'R2 Mean', 'R2 SE'])

    # Initialize Seaborn
    sns.set(style="whitegrid")

    # RMSE plot
    plt.figure(figsize=(10, 6))
    barplot = sns.barplot(x='RMSE Mean', y='Method',legend=False, data=df, palette='mako', hue='Method')
    for index, row in df.iterrows():
        barplot.errorbar(row['RMSE Mean'], index, xerr=row['RMSE SE'], fmt='none', c='black', capsize=5)
    plt.title('Mean RMSE Scores by Method with Standard Error')
    plt.tight_layout()
    plt.savefig('../figures/rmse.png')

    # R2 plot
    plt.figure(figsize=(10, 6))
    barplot = sns.barplot(x='R2 Mean', y='Method',legend=False, data=df, palette='mako', hue='Method')
    for index, row in df.iterrows():
        barplot.errorbar(row['R2 Mean'], index, xerr=row['R2 SE'], fmt='none', c='black', capsize=5)
    plt.title('Mean R^2 Scores by Method with Standard Error')
    plt.tight_layout()
    plt.savefig('../figures/r2.png')


def visualize_residuals(y_true, y_pred, method):
    # Calculate residuals
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    residuals = y_true - y_pred

    # Create a residual plot
    plt.figure(figsize=(8, 8))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='black', linestyle='--')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.title(f'Residual plot from all outer folds for {method} model')
    plt.savefig(f'../figures/residuals_{method}.png')

def visualize_coefficients(results):
    for method, info in results.items():
        if method in ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet']:  # RandomForest does not have .coef_
            coefs = [model.coef_ for model in info['model']]
            coef_df = pd.DataFrame(coefs, index=[f'{method} Model {i+1}' for i in range(len(coefs))])

            plt.figure(figsize=(10, 7))
            sns.heatmap(coef_df, annot=False, cmap='Spectral', center=0)
            plt.title(f'{method} Coefficients')
            plt.xlabel('Feature')
            plt.ylabel('Model')
            plt.savefig(f'../figures/{method}_coefficients.png')

def visualize_c1():
    data = load_data()

    data.replace(to_replace=r'^\s*NaN\s*$', value=np.nan, regex=True, inplace=True)
    sns.set_theme(style='darkgrid', palette='pastel', font='sans-serif', font_scale=1, color_codes=True, rc=None)

    # Create a barplot so that categories are on the x-axis and counts are on the y-axis
    plt.figure(figsize=(8, 8))
    sns.countplot(data[' C_ 1'], color='black')
    plt.title('Distribution of C_1')
    plt.savefig('../figures/c1_histogram_nan.png')

    # Impute missing values with most frequent


    data[' C_ 1'] = data[' C_ 1'].fillna(data[' C_ 1'].mode()[0])

    # Create a histogram of C_1
    plt.figure(figsize=(8, 8))
    sns.countplot(data[' C_ 1'], color='black')
    plt.title('Distribution of C_1 after imputation')
    plt.savefig('../figures/c1_histogram_imputed.png')


if __name__ == '__main__':
    data = load_data()
    X, y, preprocessor = preprocess(data)
    visualize_correlation_matrix(X)
    visualize_c1()
