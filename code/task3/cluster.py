import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    df = pd.read_csv('..\\..\\data\\task3\\DC_Crime_Preprocessed.csv')
    features = df.drop('NEIGHBORHOOD_CLUSTER', axis=1)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features.iloc[:, 1:])

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(scaled_features)

    df['cluster'] = kmeans.labels_
    print(df)
    df.to_csv('..\\..\\data\\task3\\KMeans.csv', index=True)
