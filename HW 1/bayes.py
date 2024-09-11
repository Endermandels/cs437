import pandas as pd
import math

"""
Bayes Theorem
P(c) * P(F1 | c) * P(F2 | c) * ...
----------------------------------
            P(F1 ^ F2 ^ ...)
"""

class BayesianClassifier:
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)
        self.train = self.data.sample(frac=0.7)
        self.test = self.data.drop(self.train.index)
        
    def compute_probability_table(self):
        pass

def main():
    bc = BayesianClassifier('zoo.csv')

if __name__ == '__main__':
    main()