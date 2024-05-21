import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('..\\..\\data\\task3\\DC_Crime.csv')
    grouped_stats = df.groupby('NEIGHBORHOOD_CLUSTER').agg({
        'ucr-rank': 'sum',
        'NEIGHBORHOOD_CLUSTER': 'count',
        'SHIFT_day': 'sum',
        'SHIFT_evening': 'sum',
        'SHIFT_midnight': 'sum',
        'OFFENSE_GROUP_property': 'sum',
        'OFFENSE_GROUP_violent': 'sum',
        'METHOD_gun': 'sum',
        'METHOD_knife': 'sum',
        'METHOD_others': 'sum'
    })
    grouped_stats.rename(columns={'ucr-rank': 'ucr-rank_sum'}, inplace=True)
    grouped_stats.rename(columns={'NEIGHBORHOOD_CLUSTER': 'crime_cnt'}, inplace=True)
    grouped_stats['ucr-rank_avg'] = grouped_stats['ucr-rank_sum'] / grouped_stats['crime_cnt']
    grouped_stats['SHIFT_day_rate'] = grouped_stats['SHIFT_day'] / grouped_stats['crime_cnt']
    grouped_stats['SHIFT_evening_rate'] = grouped_stats['SHIFT_evening'] / grouped_stats['crime_cnt']
    grouped_stats['SHIFT_midnight_rate'] = grouped_stats['SHIFT_midnight'] / grouped_stats['crime_cnt']
    grouped_stats['OFFENSE_GROUP_property_rate'] = grouped_stats['OFFENSE_GROUP_property'] / grouped_stats['crime_cnt']
    grouped_stats['OFFENSE_GROUP_violent_rate'] = grouped_stats['OFFENSE_GROUP_violent'] / grouped_stats['crime_cnt']
    grouped_stats['METHOD_gun_rate'] = grouped_stats['METHOD_gun'] / grouped_stats['crime_cnt']
    grouped_stats['METHOD_knife_rate'] = grouped_stats['METHOD_knife'] / grouped_stats['crime_cnt']
    grouped_stats['METHOD_others_rate'] = grouped_stats['METHOD_others'] / grouped_stats['crime_cnt']
    cols = ['crime_cnt', 'ucr-rank_avg', 'SHIFT_day_rate', 'SHIFT_evening_rate',
            'SHIFT_midnight_rate', 'OFFENSE_GROUP_property_rate', 'OFFENSE_GROUP_violent_rate', 'METHOD_gun_rate',
            'METHOD_knife_rate', 'METHOD_others_rate']
    grouped_stats = grouped_stats.reindex(columns=cols)
    print(grouped_stats)
    grouped_stats.to_csv('..\\..\\data\\task3\\DC_Crime_Preprocessed.csv', index=True)
