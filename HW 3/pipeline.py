"""
Project 3
CS 437
Elijah Delavar
"""

from sklearn.model_selection import GridSearchCV
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
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

class TransformData(BaseEstimator, TransformerMixin):
    def __init__(self, func):
        self.func = func

    def fit(self, xs, ys, **params):
        return self
    
    def transform(self, xs):
        result = xs.apply(self.func)
        return result

def myLinearRegression(numeric_columns, categorical_columns, data):
    grid = {
        'column_select__columns': [
            list(numeric_columns) + list(categorical_columns)
        ],
        'transform_data__func': [
            lambda x: np.where(x > 0, np.sqrt(x), 0)
            , lambda x: x
            , np.square
        ],
        'linear_regression': [
            TransformedTargetRegressor(
                LinearRegression(n_jobs=-1),
                func = np.sqrt,
                inverse_func = np.square
            )
            , TransformedTargetRegressor(
                LinearRegression(n_jobs=-1),
                func = lambda x: x,
                inverse_func = lambda x: x
            )
            , TransformedTargetRegressor(
                LinearRegression(n_jobs=-1),
                func = np.log,
                inverse_func = np.exp
            )
        ]
    }
    
    steps = [
        ('column_select', SelectColumns(['Gr Liv Area', 'Overall Qual']))
        , ('transform_data', TransformData(lambda x: x))
        , ('linear_regression', TransformedTargetRegressor(
                LinearRegression(n_jobs=-1),
                func = lambda x: x,
                inverse_func = lambda x: x
            ))
    ]
    pipe = Pipeline(steps)
    search = GridSearchCV(pipe, grid, scoring='r2', n_jobs=-1, cv=5)

    xs = data.drop(columns=[TARGET])
    ys = data[TARGET]

    search.fit(xs, ys)

    print('Linear regression:')
    print('R-squared:', search.best_score_) # Usually gets 0.83 r2
    print('Best params:', search.best_params_)
    
def myDecisionTreeRegression(numeric_columns, categorical_columns, data):
    grid = {
        'column_select__columns': [
            list(numeric_columns) + list(categorical_columns)
        ],
        'transform_data__func': [
            lambda x: np.where(x > 0, np.sqrt(x), 0)
            , lambda x: x
            , np.square
        ],
        'regression__max_depth': [
            1,2,3,4,5,6,7,8,9,10,11,12,13,14
        ]
    }
    
    steps = [
        ('column_select', SelectColumns(['Gr Liv Area', 'Overall Qual']))
        , ('transform_data', TransformData(lambda x: x))
        , ('regression', DecisionTreeRegressor(max_depth=2))
    ]
    pipe = Pipeline(steps)
    search = GridSearchCV(pipe, grid, scoring='r2', n_jobs=-1, cv=5)

    xs = data.drop(columns=[TARGET])
    ys = data[TARGET]

    search.fit(xs, ys)

    print('Decision tree:')
    print('R-squared:', search.best_score_) # Usually gets around 0.78 r2
    print('Best params:', search.best_params_)

def myRandomForestRegression(numeric_columns, categorical_columns, data):
    grid = {
        'column_select__columns': [
            list(numeric_columns) + list(categorical_columns)
        ],
        'transform_data__func': [
            lambda x: np.where(x > 0, np.sqrt(x), 0)
            , lambda x: x
            # , np.square
        ],
        'regression__max_depth': [
            # 2,3,4,5,6,7,8,9,10,11,12,13,14
            6,8,12,13
        ]
    }
    
    steps = [
        ('column_select', SelectColumns(['Gr Liv Area', 'Overall Qual']))
        , ('transform_data', TransformData(lambda x: x))
        , ('regression', RandomForestRegressor(max_depth=2))
    ]
    pipe = Pipeline(steps)
    search = GridSearchCV(pipe, grid, scoring='r2', n_jobs=-1, cv=5)

    xs = data.drop(columns=[TARGET])
    ys = data[TARGET]

    search.fit(xs, ys)

    print('Random forest:')
    print('R-squared:', search.best_score_) # Usually gets around 0.88 r2
    print('Best params:', search.best_params_)

def myGradientBoostingRegression(numeric_columns, categorical_columns, data):
    grid = {
        'column_select__columns': [
            list(numeric_columns) + list(categorical_columns)
        ],
        'transform_data__func': [
            lambda x: np.where(x > 0, np.sqrt(x), 0)
            , lambda x: x
            # , np.square
        ],
        'regression__max_depth': [
            # 2,3,4,5,6,7,8,9
            3
        ],
        'regression__learning_rate': [
            # 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
            0.1
        ]
    }
    
    steps = [
        ('column_select', SelectColumns(['Gr Liv Area', 'Overall Qual']))
        , ('transform_data', TransformData(lambda x: x))
        , ('regression', GradientBoostingRegressor(max_depth=2))
    ]
    pipe = Pipeline(steps)
    search = GridSearchCV(pipe, grid, scoring='r2', n_jobs=-1, cv=5)

    xs = data.drop(columns=[TARGET])
    ys = data[TARGET]

    search.fit(xs, ys)

    print('Gradient boosting:')
    print('R-squared:', search.best_score_) # Usually gets around 0.90 r2
    print('Best params:', search.best_params_)



def main():
    data = pd.read_csv('AmesHousing.csv')
    data = data.drop(columns=['Neighborhood'])

    categorical_columns = data.select_dtypes(include=['object', 'category']).columns

    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    numeric_columns = numeric_columns.drop([TARGET, 'Order', 'PID', 'BsmtFin SF 2',
                                            'Low Qual Fin SF', '3Ssn Porch', 'Pool Area',
                                            'Misc Val', 'Mo Sold', 'Yr Sold'])

    data[categorical_columns] = data[categorical_columns].fillna('Missing')
    data[numeric_columns] = data[numeric_columns].fillna(0)
    
    # Encode categorical columns
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_columns = encoder.fit_transform(data[categorical_columns])
    encoded_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(categorical_columns))

    # Drop the original categorical data and add the encoded data
    data = data.drop(categorical_columns, axis=1)
    data = pd.concat([data, encoded_df], axis=1)

    # Get the correct feature names from the encoder
    categorical_columns = encoder.get_feature_names_out(categorical_columns)


    myLinearRegression(numeric_columns, categorical_columns, data)
    myDecisionTreeRegression(numeric_columns, categorical_columns, data)
    myRandomForestRegression(numeric_columns, categorical_columns, data)
    myGradientBoostingRegression(numeric_columns, categorical_columns, data)

if __name__ == '__main__':
    main()