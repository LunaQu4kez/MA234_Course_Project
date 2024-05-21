import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import random


if __name__ == '__main__':
    colors = ['blue', 'red', 'yellow']

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    shapefile_path = '..\\..\\data\\WDC_shp'
    us_states = gpd.read_file(shapefile_path)
    us_states.plot(ax=ax, color='lightgrey', edgecolor='0.8')

    df = pd.read_csv('..\\..\\data\\task3\\DC_Crime.csv')
    cl = pd.read_csv('..\\..\\data\\task3\\Hierarchical.csv')

    color_idx = [-1 for _ in range(50)]

    for index, row in cl.iterrows():
        color_idx[int(row['NEIGHBORHOOD_CLUSTER'])] = int(row['cluster'])

    print(color_idx)
    for index, row in df.iterrows():
        if row['YEAR'] == 2017:
            if random.randint(0, 9) == 0:
                color = colors[color_idx[int(row['NEIGHBORHOOD_CLUSTER'])]]
                ax.scatter(row['LONGITUDE'], row['LATITUDE'], color=color, s=10)

    ax.set_title('K-Means Cluster')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    plt.show()
