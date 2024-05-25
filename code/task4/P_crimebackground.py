import pandas as pd
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



df_C = pd.read_csv('..\\..\\data\\task4\\Process_Properties_condominium.csv')
df_G = pd.read_csv('..\\..\\data\\task4\\Process_Properties_Residential.csv')
df_G = df_G.drop(df_G.columns[0], axis=1)
df_C = df_C.drop(df_C.columns[0], axis=1)

df = pd.read_csv('..\\..\\data\\task2\\DC_Crime_Preprocessed.csv')

X = df[['LONGITUDE', 'LATITUDE']]  
y = df['NEIGHBORHOOD_CLUSTER']  

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

knn = KNeighborsClassifier(n_neighbors=5)  

knn.fit(X, y_encoded)

X_C = df_C[['LONGITUDE', 'LATITUDE']]
predicted_clusters = knn.predict(X_C)

X_G = df_G[['LONGITUDE', 'LATITUDE']]
predicted_clusters_G = knn.predict(X_G)

predicted_clusters_decoded = label_encoder.inverse_transform(predicted_clusters)
predicted_clusters_decoded_G = label_encoder.inverse_transform(predicted_clusters_G)

df_C['NEIGHBORHOOD_CLUSTER'] = predicted_clusters_decoded
df_G['NEIGHBORHOOD_CLUSTER'] = predicted_clusters_decoded_G

value_counts = df['NEIGHBORHOOD_CLUSTER'].value_counts()
df_G['crime background'] = df_G['NEIGHBORHOOD_CLUSTER'].map(value_counts)
df_C['crime background'] = df_C['NEIGHBORHOOD_CLUSTER'].map(value_counts)

mean_val_G = df_G['crime background'].mean()
std_val_G = df_G['crime background'].std()
df_G['crime background'] = (df_G['crime background'] - mean_val_G) / std_val_G

mean_val_C = df_C['crime background'].mean()
std_val_C = df_C['crime background'].std()
df_C['crime background'] = (df_C['crime background'] - mean_val_C) / std_val_C

df_C.to_csv('..\\..\\data\\task4\\Process_Properties_condominium.csv', index=True)
df_G.to_csv('..\\..\\data\\task4\\Process_Properties_Residential.csv', index=True)
