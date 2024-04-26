import pandas as pd


if __name__ == '__main__':
    df1 = pd.read_csv('..\\..\\data\\task2\\DC_Crime_G.csv')
    df2 = pd.read_csv('..\\..\\data\\task2\\DC_Crime_T.csv')
    df3 = pd.read_csv('..\\..\\data\\task2\\DC_Crime_C.csv')

    col1 = ['NEIGHBORHOOD_CLUSTER', 'CENSUS_TRACT', 'XBLOCK', 'YBLOCK', 'PSA', 'VOTING_PRECINCT', 'ANC']
    col2 = ['START_DATE', 'SHIFT']
    col3 = ['offensegroup', 'OFFENSE', 'METHOD', 'ucr-rank', 'CCN']

    df = df1.reindex(columns=col1)
    df['START_DATE'] = df2['START_DATE']
    df['SHIFT'] = df2['SHIFT']
    df['offensegroup'] = df3['offensegroup']
    df['OFFENSE'] = df3['OFFENSE']
    df['METHOD'] = df3['METHOD']
    df['ucr-rank'] = df3['ucr-rank']
    df['CCN'] = df3['CCN']

    df.to_csv('..\\..\\data\\task2\\DC_Crime_Screen.csv', index=True)

