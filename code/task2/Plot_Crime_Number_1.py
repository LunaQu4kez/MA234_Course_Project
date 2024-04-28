import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df = pd.read_csv('DC_Crime_Preprocessed.csv')

    df = df[(df['YEAR'] >= 2008) & (df['YEAR'] <= 2021)]
    year_counts = df['YEAR'].value_counts()
    year_counts = year_counts.sort_index()
    year_counts.plot(kind='bar')

    plt.title('Number of Crime per Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Crime')

    plt.show()
