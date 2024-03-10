import pandas as pd
import numpy as np

def load_data():
    data = pd.read_csv('case1Data.txt')

    pd.set_option('future.no_silent_downcasting', True)
    data.replace(to_replace=r'^\s*NaN\s*$', value=np.nan, regex=True, inplace=True)

    # Cast numerical values to float
    data.iloc[:, :-5] = data.iloc[:, :-5].astype(float)

    return data

if __name__ == '__main__':
    data = load_data()
    print(data.head())
