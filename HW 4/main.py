'''
Project 4
Elijah Delavar
CS 437
'''

from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import pandas as pd

def printResults(df: pd.DataFrame, pipe: Pipeline):
    for type in set(df['Type 1']):
        print(type)
        print('-----------')
        data = df[df['Type 1'] == type].drop(columns=['Type 1'])
        best_choice, best_score = optimize_n_clusters(range(2,min(data.shape[0], 15)), data, pipe)
        print('Best number of clusters:', best_choice)
        print('Best score:', best_score)
        print()

def optimize_n_clusters(r: range, data: pd.DataFrame, pipe: Pipeline):
    best_choice = r.start
    best_silhouette = float('-inf')
    
    for n_clusters in r:
        pipe.set_params(cluster__n_clusters=n_clusters)
        score = silhouette_score(data, pipe.fit_predict(data))
        
        print(f'{n_clusters} clusters: {score}')
        
        if score > best_silhouette:
            best_silhouette = score
            best_choice = n_clusters
    return best_choice, best_silhouette # we didn't have to return the score

def createPipeline(df: pd.DataFrame):
    steps = [
        ('scale', MinMaxScaler())
        , ('cluster', KMeans())
    ]
    
    return Pipeline(steps)

def getData(fn: str):
    df = pd.read_csv(fn)
    df = df.drop(columns=['#', 'Name', 'Total', 'Generation', 'Legendary', 'Type 2'])
    return df

def main():
    df = getData('./_files/Pokemon.csv')
    pipe = createPipeline(df)
    printResults(df, pipe)

if __name__ == '__main__':
    main()