# CC 1 (Code Cel 1)
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np

# CC 2
data = pd.read_csv("Video Games Sales.csv")
# Only want platform, year, genre and publisher to determine how many sales it generated globally
data = data.drop(columns=['index', 'Rank', 'Game Title', 'North America', 'Europe'
                          , 'Japan', 'Rest of World', 'Review'])

categorical_columns = data.select_dtypes(include=['object', 'category']).columns
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
# data[categorical_columns] = data[categorical_columns].fillna('Missing')
# data[numeric_columns] = data[numeric_columns].fillna(0)

categorical_data = pd.get_dummies(data[categorical_columns], dtype=np.int64)

data = data.drop(categorical_columns, axis=1)
data = pd.concat([data, categorical_data], axis=1)

data.fillna(value=0, inplace=True)

print (data)
