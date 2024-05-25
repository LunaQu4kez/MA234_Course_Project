import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

df_C = pd.read_csv('..\\..\\data\\task4\\Process_Properties_Residential.csv')
df_C = df_G.drop(df_G.columns[0], axis=1)

bool_columns = df_.select_dtypes(include=[bool]).columns

df_C[bool_columns] = df_C[bool_columns].astype(int)

scaler = MinMaxScaler()

df_non_bool = df_C.select_dtypes(exclude=[bool, object])
scaled_data = scaler.fit_transform(df_non_bool)
df_scaled = pd.DataFrame(scaled_data, columns=df_non_bool.columns, index=df_non_bool.index)

for column in df_scaled.columns:
    df_C[column] = df_scaled[column]

df_C[bool_columns] = df_C[bool_columns].astype(bool)


X = df_C.drop('PRICE', axis=1)
y = df_C['PRICE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

importances = rf.feature_importances_

feature_names = X_train.columns

sorted_indices = importances.argsort()

plt.barh(range(len(sorted_indices)), importances[sorted_indices], align='center')
plt.yticks(range(len(sorted_indices)), [feature_names[i] for i in sorted_indices])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importances')
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)

filename = '..\\..\\pic\\task4\\4.3_Feature Importances_Condominium.png'
plt.savefig(filename, bbox_inches='tight', dpi=300)