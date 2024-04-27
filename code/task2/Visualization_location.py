import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    correlation_matrix = pd.read_csv('..\\..\\data\\task2\\corr_G.csv')
    correlation_matrix.set_index(correlation_matrix.columns[0], inplace=True)
    correlation_matrix = correlation_matrix.reset_index(drop=True)
    column_names = correlation_matrix.columns.tolist()
    new_index = [None] * len(correlation_matrix)

    for i, column_name in enumerate(column_names):
        new_index[i] = column_name

    correlation_matrix.index = new_index
    
    correlation_matrix=correlation_matrix.abs()

    
    Heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='Reds', vmin=0, vmax=1)
    plt.title("Correlation Matrix Heatmap")
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)

    filename = '..\\..\\data\\task2\\Correlation Matrix Heatmap.png'
    plt.savefig(filename, bbox_inches='tight', dpi=300)


