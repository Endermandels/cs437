# CC 1 (Code Cel 1)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

# CC 2
data = pd.read_csv("Video Games Sales.csv")
# Only want platform, year, genre and publisher to determine how many sales it generated globally
data = data.drop(columns=['index', 'Rank', 'Game Title', 'North America', 'Europe'
                          , 'Japan', 'Rest of World', 'Review'])

categorical_columns = data.select_dtypes(include=['object', 'category']).columns
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
categorical_data = pd.get_dummies(data[categorical_columns], dtype=np.int64)

data = data.drop(categorical_columns, axis=1)
data = pd.concat([data, categorical_data], axis=1)
data.fillna(value=0, inplace=True)

xs = data.drop(columns=['Global'])
ys = data['Global']

x_train, x_test, y_train, y_test = train_test_split(xs, ys, train_size=0.7)

print(x_train)
print(x_test)
print(y_train)
print(y_test)


