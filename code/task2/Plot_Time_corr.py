import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df = pd.read_csv('..\\..\\data\\task2\\DC_Crime_Preprocessed.csv')
    index = df.columns.get_loc('SHIFT_day')

    grouped = df.groupby('YEAR')
    
    unique_values = df['YEAR'].unique()

    grouped_dataframes = {}

    for value in unique_values:
        grouped_dataframes[value] = df[df['YEAR'] == value]

    for value in range(2008,2022):
        df=grouped_dataframes[value]
        CA_df = df.iloc[:, index :]

        correlation_matrix = CA_df.corr()
        correlation_matrix=correlation_matrix.round(2)

        plt.figure(figsize=(10, 8))
        Heatmap=sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1,annot_kws={'fontsize': 6})
        plt.title('Crime Info Correlation Matrix Heatmap shift-offense-method in %s'%value)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)

        filename = '..\\..\\pic\\task2\\2.2_Corr_crime_Matrix_Heatmap in %s.png'%value
        plt.savefig(filename, bbox_inches='tight', dpi=300)


    
