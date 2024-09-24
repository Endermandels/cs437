"""
Project 2
CS 437
Elijah Delavar
"""

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

class SelectColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, xs, ys, **params):
        return self
    
    def transform(self, xs):
        return xs[self.columns]


def main():
    data = pd.read_csv('AmesHousing.csv')

    grid = {
        'column_select__columns': [
            ['Gr Liv Area']
            , ['Overall Qual']
            , ['Gr Liv Area', 'Overall Qual']
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

    xs = data.drop(columns=['SalePrice'])
    ys = data['SalePrice']

    search.fit(xs, ys)
    r_squared = search.best_score_

    print(f'R-Squared: {r_squared}')

if __name__ == '__main__':
    main()