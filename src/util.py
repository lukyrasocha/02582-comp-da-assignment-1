import pandas as pd
import numpy as np

def load_data():
    data = pd.read_csv('../case1Data.txt')

    return data

if __name__ == '__main__':
    data = load_data()
    print(data.head())
