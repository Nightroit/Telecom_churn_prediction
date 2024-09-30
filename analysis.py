import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('data.csv')

# Data Cleaning

for col in df.columns:
    if len(df[col].unique()) < 100: 
        df[col] = pd.factorize(df[col])[0]

