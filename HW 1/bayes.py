from collections import defaultdict
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
        self.LABEL_COL = 'class_type'
        self.data = pd.read_csv(data_file)
        self.class_feat_prob = self.compute_probability_table(self.data)
        
        self.train = self.data.sample(frac=0.7)
        self.test = self.data.drop(self.train.index)
        
    def compute_probability_table(self, data):
        """
        class_feat_prob: key = class, val = (prob, feat_prob: key = feat, val = prob)
        class_feat_counts: key = class, val = [count, feat_count: key = feat, val = count]
        """
        class_feat_prob = {}
        class_feat_counts = defaultdict(lambda: [0, defaultdict(lambda: 0)])
        
        for i, row in data.iterrows():
            class_type = row[self.LABEL_COL]
            class_feat_counts[class_type][0] += 1
                
            for feat_name, feat_val in row[1:-1].items(): # ignore animal_name and class_type
                if feat_name == 'legs':
                    class_feat_counts[class_type][1][f'{feat_name}_{feat_val}'] += 1
                else:
                    class_feat_counts[class_type][1][feat_name] += feat_val
                
        num_rows = data.shape[0]
        
        for class_type, class_stats in class_feat_counts.items():
            class_count = class_stats[0]
            feat_prob = {}
            for feat_name, feat_count in class_stats[1].items():
                feat_prob[feat_name] = feat_count / class_count
            class_feat_prob[class_type] = (class_count / num_rows, feat_prob)
            
        return class_feat_prob

def main():
    bc = BayesianClassifier('zoo.csv')

if __name__ == '__main__':
    main()