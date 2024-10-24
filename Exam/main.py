# C 2 (Cel 1)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import *

import pandas as pd

def main():
    # C 3
    TARGET = 'Rest of World'
    data = pd.read_csv("Video Games Sales.csv")
    data = data[['North America', 'Europe', 'Japan', 'Rest of World']]
    data.fillna(value=0, inplace=True)
    
    xs = data.drop(columns=[TARGET])
    ys = data[TARGET]
    
    x_train, x_test, y_train, y_test = train_test_split(xs, ys, train_size=0.7)

    print(x_train)
    print(x_test)
    print(y_train)
    print(y_test)

    # C 4
    steps = [
        ('scale', MinMaxScaler()),
        ('predict', LinearRegression(n_jobs=-1))
    ]
    pipe = Pipeline(steps)
    pipe.fit(x_train, y_train)
    
    
    # C 5
    predicted_ys = pipe.predict(x_test)
    print('r2', r2_score(y_test, predicted_ys))

if __name__ == '__main__':
    main()