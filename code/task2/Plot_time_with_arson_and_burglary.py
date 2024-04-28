import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == '__main__':
    df = pd.read_csv('..\\..\\data\\task2\\DC_Crime_Preprocessed.csv')

    index = df.columns.get_loc('SHIFT_day')
    df = df.iloc[:, index :]

    Arson=df[df['OFFENSE_arson']==1]
    Bur=df[df['OFFENSE_burglary']==1]
    
    Arson=Arson[['SHIFT_day','SHIFT_evening','SHIFT_midnight',]]
    Bur=Bur[['SHIFT_day','SHIFT_evening','SHIFT_midnight',]]

    Ar_count=Arson.sum()
    Ar_count.plot(kind='bar', color=sns.color_palette("husl", 3))
    plt.title('The time of arson occurrence')
    plt.xlabel('Time')
    plt.ylabel('The number of arsons')
    filename = '..\\..\\pic\\task2\\2.2_arson_bar.png'
    plt.savefig(filename, bbox_inches='tight', dpi=300)

    Bur_count=Bur.sum()
    Bur_count.plot(kind='bar', color=sns.color_palette("husl", 3))
    plt.title('The time of burglary occurrence')
    plt.xlabel('Time')
    plt.ylabel('The number of burglarys')
    filename = '..\\..\\pic\\task2\\2.2_Burglary_bar.png'
    plt.savefig(filename, bbox_inches='tight', dpi=300)