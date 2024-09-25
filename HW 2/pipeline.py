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
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sys

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

    categorical_columns = data.select_dtypes(include=['object', 'category']).columns

    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    numeric_columns = numeric_columns.drop([TARGET, 'Order', 'PID', 'BsmtFin SF 2',
                                            'Low Qual Fin SF', '3Ssn Porch', 'Pool Area',
                                            'Misc Val', 'Mo Sold', 'Yr Sold'])

    data[categorical_columns] = data[categorical_columns].fillna('Missing')
    data[numeric_columns] = data[numeric_columns].fillna(0)
    
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_columns = encoder.fit_transform(data[categorical_columns])
    encoded_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(categorical_columns))

    data = data.drop(categorical_columns, axis=1)
    data = pd.concat([data, encoded_df], axis=1)

    categorical_columns = encoder.get_feature_names_out(categorical_columns)

    grid = {
        'column_select__columns': [
            list(numeric_columns) + list(categorical_columns)
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
    
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns

    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    numeric_columns = numeric_columns.drop([TARGET, 'Order', 'PID', 'BsmtFin SF 2',
                                            'Low Qual Fin SF', '3Ssn Porch', 'Pool Area',
                                            'Misc Val', 'Mo Sold', 'Yr Sold'])

    data[categorical_columns] = data[categorical_columns].fillna('Missing')
    data[numeric_columns] = data[numeric_columns].fillna(0)
    
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_columns = encoder.fit_transform(data[categorical_columns])
    encoded_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(categorical_columns))

    data = data.drop(categorical_columns, axis=1)
    data = pd.concat([data, encoded_df], axis=1)
    
    categorical_columns = encoder.get_feature_names_out(categorical_columns)
    
    for col in categorical_columns:
        plt.figure(figsize=(6,4))
        plt.scatter(data[col], data[TARGET], alpha=0.5)
        plt.title(f'{col} vs {TARGET}')
        plt.xlabel(col)
        plt.ylabel(TARGET)
        plt.grid(True)
        plt.show()
    
    # correlation_matrix = data[numeric_columns].corr()

    # plt.figure(figsize=(12,8))
    # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    # plt.show()

if __name__ == '__main__':
    main()
    # data_corr()