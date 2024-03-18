from preprocess import preprocess
from util import load_data

from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
new_X = pd.read_csv("case1Data_Xnew.txt", sep='\t')
data = load_data()
X, y, preprocessor = preprocess(data)
new_X = preprocessor.transform(new_X)
