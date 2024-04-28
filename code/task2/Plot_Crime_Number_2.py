import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df = pd.read_csv('DC_Crime_Preprocessed.csv')

    year_counts = df['NEIGHBORHOOD_CLUSTER'].value_counts()
    year_counts = year_counts.sort_index()
    year_counts.plot(kind='bar')

    for index, value in enumerate(year_counts):
        plt.text(index, value, str(value), ha='center', va='bottom', size=6)

    plt.title('Number of Crime of Clusters')
    plt.xlabel('Neighborhood Cluster')
    plt.ylabel('Number of Crime')

    plt.show()
