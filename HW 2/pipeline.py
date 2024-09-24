"""
Project 2
CS 437
Elijah Delavar
"""

from sklearn.model_selection import GridSearchCV
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

TARGET = 'SalePrice'

class SelectColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, xs, ys, **params):
        return self
    
    def transform(self, xs):
        return xs[self.columns]


def main():
    data = pd.read_csv('AmesHousing.csv')

    data = data.fillna(0)

    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    numeric_columns = numeric_columns.drop([TARGET, 'Order', 'PID', 'BsmtFin SF 2', 'Low Qual Fin SF', '3Ssn Porch', 'Pool Area', 'Misc Val', 'Mo Sold', 'Yr Sold'])

    grid = {
        'column_select__columns': [
            numeric_columns
        ],
        'linear_regression': [
            TransformedTargetRegressor(
                LinearRegression(n_jobs=-1),
                func = lambda x: x,
                inverse_func = lambda x: x
            ),
            TransformedTargetRegressor(
                LinearRegression(n_jobs=-1),
                func = np.sqrt,
                inverse_func = np.square
            ),
            TransformedTargetRegressor(
                LinearRegression(n_jobs=-1),
                func = np.log,
                inverse_func = np.exp
            )
        ]
    }


    steps = [
        ('column_select', SelectColumns(['Gr Liv Area', 'Overall Qual']))
        , ('linear_regression', TransformedTargetRegressor(
                LinearRegression(n_jobs=-1),
                func = lambda x: x,
                inverse_func = lambda x: x
            ))
    ]
    pipe = Pipeline(steps)
    search = GridSearchCV(pipe, grid, scoring='r2', n_jobs=-1)

    xs = data.drop(columns=[TARGET])
    ys = data[TARGET]

    search.fit(xs, ys)

    print(search.best_score_)
    print(search.best_params_)

def data_corr():
    data = pd.read_csv('AmesHousing.csv')
    
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    numeric_columns = numeric_columns.drop(TARGET)

    for col in numeric_columns:
        plt.figure(figsize=(6,4))
        plt.scatter(data[col], data[TARGET], alpha=0.5)
        plt.title(f'{col} vs {TARGET}')
        plt.xlabel(col)
        plt.ylabel(TARGET)
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    main()
    # data_corr()