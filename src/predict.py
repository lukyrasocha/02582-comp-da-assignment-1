from preprocess import preprocess
from util import load_data

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.pipeline import Pipeline
import pandas as pd

newx = pd.read_csv('../case1Data_Xnew.txt')
oldx = load_data()

oldX, y, preprocessor = preprocess(oldx)
newX, _, _ = preprocess(newx)

pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', Lasso(alpha=1.7782794100389228))])
trained_model = pipeline.fit(oldX, y)

predictions = trained_model.predict(newX)

print(predictions.shape)

import numpy as np

# Save predictions
np.savetxt('predictions.txt', predictions, fmt='%f', newline='\n')
