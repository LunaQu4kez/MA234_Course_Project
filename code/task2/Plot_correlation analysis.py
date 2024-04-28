import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == '__main__':
    df = pd.read_csv('..\\..\\data\\task2\\DC_Crime_Preprocessed.csv')

    index = df.columns.get_loc('SHIFT_day')
    CA_df = df.iloc[:, index :]

    correlation_matrix = CA_df.corr()
    correlation_matrix=correlation_matrix.abs()
    correlation_matrix=correlation_matrix.round(2)

    Heatmap=sns.heatmap(correlation_matrix, annot=True, cmap='Reds', vmin=0, vmax=1,annot_kws={'fontsize': 6})
    plt.title('Crime Info Correlation Matrix Heatmap  shift-offense-method')
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.show()

    

