import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import random

if __name__ == '__main__':
    y = 2021  # 2008 to 2021
    colors = ['yellow', 'blue', 'red', 'grey', 'black', 'blue', 'pink', 'lime', 'salmon', 'skyblue', 'purple']

    shapefile_path = '..\\..\\data\\WDC_shp'
    us_states = gpd.read_file(shapefile_path)

    df = pd.read_csv('DC_Crime_Preprocessed.csv')

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    us_states.plot(ax=ax, color='lightgrey', edgecolor='0.8')

    for index, row in df.iterrows():
        if row['YEAR'] == y:
            if random.randint(0, 9) == 0:
                color = colors[int(row['NEIGHBORHOOD_CLUSTER']) % len(colors)]
                ax.scatter(row['LONGITUDE'], row['LATITUDE'], color=color, s=10)

    ax.set_title('Crime of Year ' + str(y))
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    plt.show()
