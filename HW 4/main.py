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
    types = set(df['Type 1'])
    clusters = {}
    
    # Show cluster scores
    for type in types:
        print(type)
        print('-----------')
        
        data = df[df['Type 1'] == type].drop(columns=['Type 1', 'Name'])
        
        best_choice, best_score, best_prediction = \
            optimize_n_clusters(range(2,min(data.shape[0], 15)), data, pipe)
            
        clusters[type] = best_prediction
        
        print('Best number of clusters:', best_choice)
        print('Best score:', best_score)
        print()

    # Show cluster elements and means
    for type in types:
        print(type)
        print('-----------')
        
        data = df[df['Type 1'] == type].drop(columns=['Type 1'])
        
        # Put cluster into data
        data['Cluster'] = clusters[type]
        
        # Print data grouped by cluster
        groups = data.groupby('Cluster')
        for cluster_num, cluster_data in groups:
            print(f"Cluster {cluster_num}")
            print(cluster_data.drop(columns=['Cluster']).to_string())
            print('Mean HP:', cluster_data['HP'].mean())
            print('Mean Attack:', cluster_data['Attack'].mean())
            print('Mean Defend:', cluster_data['Defense'].mean())
            print('Mean Sp. Attack:', cluster_data['Sp. Atk'].mean())
            print('Mean Sp. Defense:', cluster_data['Sp. Def'].mean())
            print('Mean Speed:', cluster_data['Speed'].mean())
            print()
        print()

def optimize_n_clusters(r: range, data: pd.DataFrame, pipe: Pipeline):
    best_choice = r.start
    best_silhouette = float('-inf')
    best_prediction = None
    
    for n_clusters in r:
        pipe.set_params(cluster__n_clusters=n_clusters)
        prediction = pipe.fit_predict(data)
        score = silhouette_score(data, prediction)
        
        print(f'{n_clusters} clusters: {score}')
        
        if score > best_silhouette:
            best_silhouette = score
            best_choice = n_clusters
            best_prediction = prediction
    return best_choice, best_silhouette, best_prediction

def createPipeline(df: pd.DataFrame):
    steps = [
        ('scale', MinMaxScaler())
        , ('cluster', KMeans())
    ]
    
    return Pipeline(steps)

def getData(fn: str):
    df = pd.read_csv(fn)
    df = df.drop(columns=['#', 'Total', 'Generation', 'Legendary', 'Type 2'])
    return df

def main():
    df = getData('./_files/Pokemon.csv')
    pipe = createPipeline(df)
    printResults(df, pipe)

if __name__ == '__main__':
    main()