import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import random


def type2int(s) -> int:
    x = -1
    if s == 'OFFENSE_arson':
        x = 0
    elif s == 'OFFENSE_assault w/dangerous weapon':
        x = 1
    elif s == 'OFFENSE_burglary':
        x = 2
    elif s == 'OFFENSE_homicide':
        x = 3
    elif s == 'OFFENSE_motor vehicle theft':
        x = 4
    elif s == 'OFFENSE_robbery':
        x = 5
    elif s == 'OFFENSE_sex abuse':
        x = 6
    elif s == 'OFFENSE_theft f/auto':
        x = 7
    elif s == 'OFFENSE_theft/other':
        x = 8
    return x


if __name__ == '__main__':
    t = 'SHIFT_midnight'
    colors = ['yellow', 'blue', 'red', 'black', 'blue', 'pink', 'lime', 'salmon', 'purple']

    df = pd.read_csv('DC_Crime_Preprocessed.csv')
    df = df[df[t] == 1]
    crime_columns = [col for col in df.columns if col in ['OFFENSE_arson', 'OFFENSE_assault w/dangerous weapon',
                                                          'OFFENSE_burglary', 'OFFENSE_homicide',
                                                          'OFFENSE_motor vehicle theft', 'OFFENSE_robbery',
                                                          'OFFENSE_sex abuse', 'OFFENSE_theft f/auto',
                                                          'OFFENSE_theft/other']]
    crime_counts = df.groupby('NEIGHBORHOOD_CLUSTER')[crime_columns].sum()
    most_common_crime = crime_counts.apply(lambda x: x.idxmax(), axis=1)

    shapefile_path = '..\\..\\data\\WDC_shp'
    us_states = gpd.read_file(shapefile_path)

    df = pd.read_csv('DC_Crime_Preprocessed.csv')

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    us_states.plot(ax=ax, color='lightgrey', edgecolor='0.8')

    for index, row in df.iterrows():
        if int(row['NEIGHBORHOOD_CLUSTER']) <= 40 and random.randint(0, 99) == 0:
            ty = str(most_common_crime[int(row['NEIGHBORHOOD_CLUSTER'])])
            color = colors[type2int(ty)]
            ax.scatter(row['LONGITUDE'], row['LATITUDE'], color=color, s=10)

    ax.set_title('Crime Type at Midnight')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    ax.plot([], [], 'o', color=colors[0], label='OFFENSE_arson')
    ax.plot([], [], 'o', color=colors[1], label='OFFENSE_assault w/dangerous weapon')
    ax.plot([], [], 'o', color=colors[2], label='OFFENSE_burglary')
    ax.plot([], [], 'o', color=colors[3], label='OFFENSE_homicide')
    ax.plot([], [], 'o', color=colors[4], label='OFFENSE_motor vehicle theft')
    ax.plot([], [], 'o', color=colors[5], label='OFFENSE_robbery')
    ax.plot([], [], 'o', color=colors[6], label='OFFENSE_sex abuse')
    ax.plot([], [], 'o', color=colors[7], label='OFFENSE_theft f/auto')
    ax.plot([], [], 'o', color=colors[8], label='OFFENSE_theft/other')
    ax.legend(loc='best')

    ax.set_xlim([-77.20, -76.90])
    ax.set_ylim([38.700, 39.000])

    plt.show()
