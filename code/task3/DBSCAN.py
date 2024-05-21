import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


if __name__ == '__main__':
    df = pd.read_csv('..\\..\\data\\task3\\DC_Crime_Preprocessed.csv')
    df_without_first_column = df.iloc[:, 1:]

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_without_first_column)

    # eps是邻域大小，min_samples是形成聚类所需的最小样本数
    dbscan = DBSCAN(eps=0.9, min_samples=3)
    dbscan.fit(df_scaled)

    df['cluster'] = dbscan.labels_
    print(df)
    df.to_csv('..\\..\\data\\task3\\DBSCAN.csv', index=True)
