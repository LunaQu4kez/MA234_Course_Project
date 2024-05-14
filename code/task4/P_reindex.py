import pandas as pd


df = pd.read_csv('..\\..\\origin_material\\DC_Properties.csv')
df = df.dropna(subset=['PRICE'])

new_order = ['FULLADDRESS','CITY','STATE','ZIPCODE', 'NATIONALGRID', 'LONGITUDE', 'ASSESSMENT_NBHD', 'ASSESSMENT_SUBNBHD', 'CENSUS_TRACT', 'CENSUS_BLOCK', 'WARD','SQUARE','X','Y','QUADRANT']
df1 = df.reindex(columns=new_order)
df1.to_csv('..\\..\\data\\task4\\DC_Properties_G.csv', index=True)

new_order = ['AYB', 'YR_RMDL', 'EYB', 'SALEDATE', 'GIS_LAST_MOD_DTTM']
df2 = df.reindex(columns=new_order)
df2.to_csv('..\\..\\data\\task4\\DC_Properties_T.csv', index=True)

new_order = ['BATHRM', 'HF_BATHRM', 'HEAT', 'AC', 'NUM_UNITS', 'ROOMS', 'BEDRM', 'STORIES','PRICE','QUALIFIED','SALE_NUM',
             'GBA','BLDG_NUM','STYLE','STRUCT','GRADE','CNDTN','EXTWALL','ROOF','INTWALL','KITCHENS',' FIREPLACES','USECODE','LANDAREA','SOURCE','CMPLX_NUM','LIVING_GBA']
df3 = df.reindex(columns=new_order)
df3.to_csv('..\\..\\data\\task4\\DC_Properties_F.csv', index=True)
