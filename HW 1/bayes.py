from collections import defaultdict
import pandas as pd
import math

LABEL_COL = 'class_type'
CATEGORICAL_COLS = ['legs']
DEBUG = False

def compute_probability_table(data):
    """
    class_feat_prob: key = class, val = (prob, default_leg_prob, feat_prob: key = feat, val = prob)
    default_leg_prob is when the specified leg number is not present in the class
    
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
    
    a = 0.01
    d = 1
    
    for class_type, class_stats in class_feat_counts.items():
        class_count = class_stats[0]
        feat_prob = {}
        for feat_name, feat_count in class_stats[1].items():
            feat_prob[feat_name] = (feat_count + a) / (class_count + a * d)
        class_feat_prob[class_type] = ((class_count + a) / (num_rows + a * d)
                                       , a / (class_count + a * d)
                                       , feat_prob)
        
    return class_feat_prob

def calc_nb(class_type, feat_name, class_stats, row):
    nb = math.log2(class_stats[0])
            
    for feat_name, feat_val in row[1:-1].items():
        feat_probs = class_stats[2]
        if feat_name in CATEGORICAL_COLS:
            feat_name = f'{feat_name}_{feat_val}'
            if feat_name in feat_probs:
                nb += math.log2(feat_probs[feat_name])
            else:
                nb += math.log2(class_stats[1]) # Default leg number probability
        elif feat_val > 0:
            nb += math.log2(feat_probs[feat_name])
            
    return pow(2, nb)

def run_test(test, class_feat_prob):
    """
    Bayes Theorem
    nb(c | f1, f2, ..., fn) = p(c) * MUL from i = 1 to n (p(xi | c))
    denom = SUM for all c in C (nb(c | f1, f2, ..., fn))
    p(x in c) = nb(c | f1, f2, ..., fn) / denom
    
    returns num_correct
    """
    
    for feat_name in test.columns:
        print(feat_name, end=',')
    print('predicted,probability,correct?')
    
    num_correct = 0
    
    for i, row in test.iterrows():

        # Calculate Naiive Bayes odds for each class
        highest_nb = (0, None)
        sum_of_nbs = 0

        for class_type, class_stats in class_feat_prob.items():
            nb = calc_nb(class_type,feat_name,class_stats,row)
            sum_of_nbs += nb
            
            if nb > highest_nb[0]:
                highest_nb = (nb, class_type)
                
        for val in row.to_list():
            print(val, end=',')
        print(highest_nb[1], end=',')
        print(highest_nb[0] / sum_of_nbs, end=',')
        if highest_nb[1] == row[LABEL_COL]:
            print('CORRECT')
            num_correct += 1
        else:
            print('wrong')
            
    if DEBUG:
        print('Percent Correct:', num_correct / test.shape[0])
    
    return num_correct

def run_multiple_tests(data, num_tests=100):
    
    train = data.sample(frac=0.7)
    test = data.drop(train.index)
    class_feat_prob = compute_probability_table(train)
    
    num_per_test = test.shape[0]
    num_correct = 0
    for _ in range(num_tests):
        num_correct += run_test(test, class_feat_prob)
        train = data.sample(frac=0.7)
        test = data.drop(train.index)
        class_feat_prob = compute_probability_table(train)
    
    if DEBUG:
        print(f'Percent correct over {num_tests} tests: {num_correct / (num_per_test * num_tests)}')

def main():
    data = pd.read_csv('zoo.csv')
    run_multiple_tests(data, num_tests=1)

if __name__ == '__main__':
    main()