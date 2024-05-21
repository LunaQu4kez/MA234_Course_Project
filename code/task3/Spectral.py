import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel


if __name__ == '__main__':
    df = pd.read_csv('..\\..\\data\\task3\\DC_Crime_Preprocessed.csv')
    features = df.drop('NEIGHBORHOOD_CLUSTER', axis=1)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features.iloc[:, 1:])

    gamma = 0.1
    affinity_matrix = rbf_kernel(scaled_features, gamma=gamma)

    spectral = SpectralClustering(n_clusters=5, affinity='precomputed', random_state=42)
    spectral.fit(affinity_matrix)

    df['cluster'] = spectral.labels_
    print(df)
    df.to_csv('..\\..\\data\\task3\\Spectral.csv', index=True)
