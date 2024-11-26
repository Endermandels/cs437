# CEL 2 #
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import pandas as pd
import numpy as np

# CEL 3 #
TARGET = 'is_legendary'

data = pd.read_csv('./updated_data.csv')
data = data.drop(columns=['classfication', 'japanese_name', 'name', 'generation'])

# Make capture_rate a numeric data type column
data['capture_rate'] = data['capture_rate'].str.extract('(\\d+)').astype(np.int64)

ability_column = 'abilities'
categorical_columns = \
    data.select_dtypes(include=['object']).columns.drop(labels=[ability_column])
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns

data[categorical_columns] = data[categorical_columns].fillna('Missing')
data[ability_column] = data[ability_column].fillna('[]')
data[numeric_columns] = data[numeric_columns].fillna(0)

# Encode categorical columns
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_df = pd.DataFrame(  encoder.fit_transform(data[categorical_columns])
                        , columns=encoder.get_feature_names_out(categorical_columns))

# Drop the original categorical data and add the encoded data
data = data.drop(categorical_columns, axis=1)
data = pd.concat([data, encoded_df], axis=1)

# Get the correct feature names from the encoder
categorical_columns = encoder.get_feature_names_out(categorical_columns)

# Convert stringified lists to actual lists
data['abilities'] = data['abilities'].apply(lambda x: eval(x))

# Convert abilities to multiple one-hot-encoded columns
mlb = MultiLabelBinarizer()
abilities_data = pd.DataFrame(  mlb.fit_transform(data['abilities'])
                            , columns=mlb.classes_
                            , index=data.index)

# Drop abilities column and add encoded data
data = data.drop(ability_column, axis=1)
data = pd.concat([data, abilities_data], axis=1)

xs = data.drop(columns=TARGET)
ys = data[TARGET]

# CEL 4 #
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

steps = [
    ('column_select', SelectColumns(['capture_rate']))
    , ('transform_data', TransformData(lambda x: x))
    , ('classify', GradientBoostingClassifier(max_depth=2))
]
pipe =  Pipeline(steps)

grid = {
    'column_select__columns': [
        list(xs.columns)
    ],
    'transform_data__func': [
        lambda x: np.where(x > 0, np.sqrt(x), 0)
        , lambda x: x
        , np.square
    ],
    'classify__max_depth': [
        2,3,4,5,6,7,8,9
    ],
    'classify__learning_rate': [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
    ]
}
search = GridSearchCV(pipe, grid, scoring='f1', n_jobs=-1, cv=5)

# CEL 5 #
search.fit(xs, ys)

print('Gradient boosting:')
print('F1-Score:', search.best_score_)
print('Best params:', search.best_params_)
