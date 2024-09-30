import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('data.csv')

# Data Cleaning
def data_cleaning(df):
    for col in df.columns:
        if len(df[col].unique()) < 100: 
            df[col] = pd.factorize(df[col])[0]

    df['TotalCharges'] = df['TotalCharges'].replace(" ", 0).astype(float)  
    df['MonthlyCharges'] = df['MonthlyCharges'].astype(float)

    return df

data_cleaning(df)

X = df[df.columns[1:len(df.columns)-1]]
y = df[df.columns[len(df.columns)-1]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Mean Squared Error:', mse)
print('R-squared:', r2)



