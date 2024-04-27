import random

import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    y = 2021   # 2008 to 2021
    colors = ['yellow', 'blue', 'red', 'grey', 'black', 'blue', 'pink', 'lime', 'salmon', 'skyblue', 'purple']

    df = pd.read_csv('DC_Crime_Preprocessed.csv')

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    for index, row in df.iterrows():
        if row['YEAR'] == y:
            if random.randint(0, 9) == 0:
                ax.plot(row['LONGITUDE'], row['LATITUDE'], '.',
                        color=colors[int(row['NEIGHBORHOOD_CLUSTER']) % len(colors)])

    ax.set_title('Crime of Year ' + str(y))
    ax.set_xlabel('LONGITUDE')
    ax.set_ylabel('LATITUDE')

    plt.show()
