import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt


if __name__ == '__main__':
    df = pd.read_csv('..\\..\\data\\task3\\DC_Crime_Preprocessed.csv')
    df_without_first_column = df.iloc[:, 1:]

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_without_first_column)

    distance_matrix = pdist(df_scaled)
    Z = linkage(distance_matrix, method='ward')

    plt.figure(figsize=(10, 7))
    dendrogram(Z)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Index')
    plt.ylabel('Distance')
    plt.show()
