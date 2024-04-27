import pandas as pd


def NEIGHBORHOOD_CLUSTER_convert(s):
    return int(s[8:])


def VOTING_PRECINCT_convert(s):
    return int(s[9:])


def anc1(s):
    return int(s[0])


def anc2(s):
    return ord(s[1]) - ord('A')


def year_cal(s):
    return int(s[0:4])


def month_cal(s):
    return int(s[5:7])


def day_cal(s):
    return int(s[8:10])


if __name__ == '__main__':
    df = pd.read_csv('..\\..\\data\\task2\\DC_Crime_Screen.csv')
    df = df.dropna()
    df = df.rename(columns={'offensegroup': 'OFFENSE_GROUP'})

    df['NEIGHBORHOOD_CLUSTER'] = df['NEIGHBORHOOD_CLUSTER'].astype(str)
    df['NEIGHBORHOOD_CLUSTER'] = df['NEIGHBORHOOD_CLUSTER'].apply(NEIGHBORHOOD_CLUSTER_convert)

    df['VOTING_PRECINCT'] = df['VOTING_PRECINCT'].astype(str)
    df['VOTING_PRECINCT'] = df['VOTING_PRECINCT'].apply(VOTING_PRECINCT_convert)

    df['ANC'] = df['ANC'].astype(str)
    df['ANC1'] = df['ANC'].apply(anc1)
    df['ANC2'] = df['ANC'].apply(anc2)

    df['START_DATE'] = df['START_DATE'].astype(str)
    df['YEAR'] = df['START_DATE'].apply(year_cal)
    df['MONTH'] = df['START_DATE'].apply(month_cal)
    df['DAY'] = df['START_DATE'].apply(day_cal)

    cols = ['NEIGHBORHOOD_CLUSTER', 'CENSUS_TRACT', 'LONGITUDE', 'LATITUDE', 'PSA', 'VOTING_PRECINCT', 'ANC1', 'ANC2',
            'YEAR', 'MONTH', 'DAY', 'SHIFT', 'OFFENSE_GROUP', 'OFFENSE', 'METHOD', 'ucr-rank', 'CCN']
    df = df.reindex(columns=cols)

    df = pd.get_dummies(df, columns=['SHIFT', 'OFFENSE_GROUP', 'OFFENSE', 'METHOD'])

    df.to_csv('..\\..\\data\\task2\\DC_Crime_Preprocessed.csv', index=True)
