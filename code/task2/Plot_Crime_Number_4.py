import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    colors = ['blue', 'red', 'black', 'pink', 'lime', 'salmon', 'purple']

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    wdc_land = gpd.read_file('..\\..\\data\\WDC_land_shp')
    wdc_land = wdc_land.to_crs(epsg=4326)
    wdc_land.plot(ax=ax, color='lightgrey', edgecolor='0.6')

    wdc_water = gpd.read_file('..\\..\\data\\WDC_waterarea_shp')
    wdc_water = wdc_water.to_crs(epsg=4326)
    wdc_water.plot(ax=ax, color='skyblue', edgecolor='0.8')

    df = pd.read_csv('DC_Crime_Preprocessed.csv')

    filtered_df = df[(df['NEIGHBORHOOD_CLUSTER'] >= 40) & (df['NEIGHBORHOOD_CLUSTER'] <= 46)]

    for index, row in filtered_df.iterrows():
        color = colors[int(row['NEIGHBORHOOD_CLUSTER']) - 40]
        ax.scatter(row['LONGITUDE'], row['LATITUDE'], color=color, s=10)

    ax.set_xlim([wdc_land.geometry.total_bounds[0], wdc_land.geometry.total_bounds[2]])
    ax.set_ylim([wdc_land.geometry.total_bounds[1], wdc_land.geometry.total_bounds[3]])

    ax.set_title('Crime of Edge')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    plt.show()
