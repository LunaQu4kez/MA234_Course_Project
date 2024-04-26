import pandas as pd


df = pd.read_csv('..\\..\\origin_material\\DC_Crime.csv')

new_order = ['NEIGHBORHOOD_CLUSTER', 'CENSUS_TRACT', 'LONGITUDE', 'YBLOCK', 'DISTRICT', 'WARD', 'BID', 'sector',
             'PSA', 'BLOCK_GROUP', 'VOTING_PRECINCT', 'XBLOCK', 'BLOCK', 'ANC', 'location', 'LATITUDE']
df1 = df.reindex(columns=new_order)
df1.to_csv('..\\..\\data\\task2\\DC_Crime_G.csv', index=True)

new_order = ['END_DATE', 'SHIFT', 'YEAR', 'START_DATE', 'REPORT_DAT']
df2 = df.reindex(columns=new_order)
df2.to_csv('..\\..\\data\\task2\\DC_Crime_T.csv', index=True)

new_order = ['offensegroup', 'offense-text', 'offensekey', 'ucr-rank', 'CCN', 'OFFENSE', 'OCTO_RECORD_ID', 'METHOD']
df3 = df.reindex(columns=new_order)
df3.to_csv('..\\..\\data\\task2\\DC_Crime_C.csv', index=True)
