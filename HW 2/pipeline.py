"""
Project 2
CS 437
Elijah Delavar
"""

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.linear_model import LinearRegression
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

    steps = [
        ('column_select', SelectColumns(['Gr Liv Area', 'Overall Qual']))
        , ('linear_regression', LinearRegression(n_jobs=-1))
    ]
    pipe = Pipeline(steps)

    xs = data.drop(columns=['SalePrice'])
    ys = data['SalePrice']
    train_x, test_x, train_y, test_y = train_test_split(xs, ys, train_size=0.7)

    pipe.fit(train_x, train_y)

    r_squared = pipe.score(test_x, test_y)

    print(f'R-Squared: {r_squared}')

if __name__ == '__main__':
    main()