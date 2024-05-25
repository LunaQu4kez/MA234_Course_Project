import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df11 = pd.read_csv('..\\..\\data\\task4\\DC_Properties_condominium_G.csv')
df12 = pd.read_csv('..\\..\\data\\task4\\DC_Properties_residential_G.csv')

df21 = pd.read_csv('..\\..\\data\\task4\\DC_Properties_condominium_T.csv')
df22 = pd.read_csv('..\\..\\data\\task4\\DC_Properties_residential_T.csv')

df31 = pd.read_csv('..\\..\\data\\task4\\DC_Properties_condominium_F.csv')
df32 = pd.read_csv('..\\..\\data\\task4\\DC_Properties_residential_F.csv')

df11=df11[['LONGITUDE', 'LATITUDE', 'WARD', 'QUADRANT']]
df11['WARD'] = df11['WARD'].str.extract('(\d+)')
quadrant_dummies = pd.get_dummies(df11['QUADRANT'], prefix='QUADRANT')
df11 = pd.concat([df11, quadrant_dummies], axis=1)
df11 = df11.drop('QUADRANT', axis=1)

df12=df12[['LONGITUDE', 'LATITUDE', 'WARD', 'QUADRANT']]
df12['WARD'] = df12['WARD'].str.extract('(\d+)')
quadrant_dummies = pd.get_dummies(df12['QUADRANT'], prefix='QUADRANT')
df12 = pd.concat([df12, quadrant_dummies], axis=1)
df12 = df12.drop('QUADRANT', axis=1)



df21['SALEDATE'] = df21['SALEDATE'].str[:4]
df21 = df21.drop('GIS_LAST_MOD_DTTM', axis=1)
df21 = df21.drop(df21.columns[0], axis=1)

df22['SALEDATE'] = df22['SALEDATE'].str[:4]
df22 = df22.drop('GIS_LAST_MOD_DTTM', axis=1)
df22 = df22.drop(df22.columns[0], axis=1)



mapping_AC = {'Y': 1, 'N': 0}
mapping_QU = {'Q': 1, 'U': 0}
mapping_GD = {'Exceptional-D ':0, 'Exceptional-C':0, 'Exceptional-B':0, 'Exceptional-A':0,
               'Fair Quality':1,'Low Quality':2, 'Average':3, 'Above Average':4 ,
               'Good Quality':5 ,'Very Good':6 ,'Excellent':7,'Superior':8 }
mapping_CD = {'Default':0, 'Poor':2, 'Fair':1, 
               'Average':3,'Good':4, 'Very Good':5, 'Excellent':6 }


missing_ratio_31 = df31.isna().mean()
columns_to_drop_31 = missing_ratio_31[missing_ratio_31 > 0.5].index
df31 = df31.drop(columns=columns_to_drop_31)
df31 = df31.drop(df31.columns[0], axis=1)
df31 = df31.drop('SOURCE', axis=1)
df31['AC'] = df31['AC'].map(mapping_AC)
df31['QUALIFIED'] = df31['QUALIFIED'].map(mapping_QU)
df31 = df31.drop('HEAT', axis=1)
df31 = df31.fillna(method='ffill')

corr_31 = df31.corr()
corr_31 = corr_31.round(2)
plt.figure(figsize=(10, 8))
Heatmap=sns.heatmap(corr_31, annot=True, cmap='coolwarm', vmin=-1, vmax=1,annot_kws={'fontsize': 6})
plt.title('Feature Info Correlation Matrix Heatmap_Condominium')
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)

filename = '..\\..\\pic\\task4\\4.1_Corr_Matrix_Heatmap_Condominium.png'
plt.savefig(filename, bbox_inches='tight', dpi=300)



missing_ratio_32 = df32.isna().mean()
columns_to_drop_32 = missing_ratio_32[missing_ratio_32 > 0.5].index
df32 = df32.drop(columns=columns_to_drop_32)
df32 = df32.drop(df32.columns[0], axis=1)
df32 = df32.drop('SOURCE', axis=1)
df32['AC'] = df32['AC'].map(mapping_AC)
df32['QUALIFIED'] = df32['QUALIFIED'].map(mapping_QU)
df32['GRADE'] = df32['GRADE'].map(mapping_GD)
df32['CNDTN'] = df32['CNDTN'].map(mapping_CD)
df32 = df32.drop('HEAT', axis=1)
df32 = df32.drop('EXTWALL', axis=1)
df32 = df32.drop('ROOF', axis=1)
df32 = df32.drop('INTWALL', axis=1)
df32 = df32.drop('STRUCT', axis=1)
df32['FLOOR'] = df32['STYLE'].str.extract('(\d+)') 
df32['STYLE_FIN'] = df32['STYLE'].str.replace('Fin', '1', regex=True)
df32['STYLE_FIN'] = df32['STYLE'].str.replace('Unfin', '0.5', regex=True)
df32['STYLE_FIN'] = df32['STYLE'].str.replace(r'[^0]', '0', regex=True)
df32 = df32.drop('STYLE', axis=1)

corr_32 = df32.corr()
corr_32 = corr_32.round(2)
plt.figure(figsize=(10, 8))
Heatmap=sns.heatmap(corr_32, annot=True, cmap='coolwarm', vmin=-1, vmax=1,annot_kws={'fontsize': 6})
plt.title('Feature Info Correlation Matrix Heatmap_Residential')
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)

filename = '..\\..\\pic\\task4\\4.1_Corr_Matrix_Heatmap_Residential.png'
plt.savefig(filename, bbox_inches='tight', dpi=300)

Processeddf1=pd.concat([df11, df21,df31], axis=1)
Processeddf1 = Processeddf1.fillna(method='ffill')
Processeddf1.to_csv('..\\..\\data\\task4\\Process_Properties_condominium.csv', index=True)

Processeddf2=pd.concat([df12, df22,df32], axis=1)
Processeddf2 = Processeddf2.fillna(method='ffill')
Processeddf2.to_csv('..\\..\\data\\task4\\Process_Properties_Residential.csv', index=True)
