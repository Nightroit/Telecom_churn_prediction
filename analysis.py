import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('data.csv')

# Data Cleaning
def data_cleaning(df):
    for col in df.columns:
        if len(df[col].unique()) < 100: 
            df[col] = pd.factorize(df[col])[0]
    
    for i in range(len(df["TotalCharges"])):
        if df["TotalCharges"][i].strip() == "":
            df["TotalCharges"][i] = 0
        else: 
            float(df["TotalCharges"][i])
            df["TotalCharges"][i] = float(df["TotalCharges"][i])
    
    df['MonthlyCharges'] = df['MonthlyCharges'].astype(float) 

data_cleaning(df)
print(type(df["TotalCharges"][0]))
