from collections import defaultdict
import pandas as pd
import math

"""
Bayes Theorem

"""

LABEL_COL = 'class_type'
CATEGORICAL_COLS = ['legs']

def compute_probability_table(data):
    """
    class_feat_prob: key = class, val = (prob, feat_prob: key = feat, val = prob)
    class_feat_counts: key = class, val = [count, feat_count: key = feat, val = count]
    """
    class_feat_prob = {}
    class_feat_counts = defaultdict(lambda: [0, defaultdict(lambda: 0)])
    
    for i, row in data.iterrows():
        class_type = row[LABEL_COL]
        class_feat_counts[class_type][0] += 1
            
        for feat_name, feat_val in row[1:-1].items(): # ignore animal_name and class_type
            if feat_name in CATEGORICAL_COLS:
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

def run_test(test, class_feat_prob):
    pass

def main():
    LABEL_COL = 'class_type'
    data = pd.read_csv('zoo.csv')
    
    train = data.sample(frac=0.7)
    test = data.drop(train.index)
    class_feat_prob = compute_probability_table(train)
    print(class_feat_prob)
    run_test(test, class_feat_prob)

if __name__ == '__main__':
    main()