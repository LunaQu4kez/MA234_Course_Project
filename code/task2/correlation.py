import pandas as pd


def NEIGHBORHOOD_CLUSTER_convert(s):
    try:
        return int(s[8:])
    except ValueError:
        return 0


def sector_convert(s):
    try:
        ascii_val = [ord(char) for char in s]
        return 256 * 256 * ascii_val[0] + 256 * ascii_val[1] + ascii_val[2]
    except ValueError:
        return 0


def BLOCK_GROUP_convert(s):
    try:
        return int(s[7:])
    except ValueError:
        return 0


def VOTING_PRECINCT_convert(s):
    try:
        return int(s[9:])
    except ValueError:
        return 0


def ANC_convert(s):
    try:
        ascii_val = [ord(char) for char in s]
        return 256 * ascii_val[0] + ascii_val[1]
    except ValueError:
        return 0


if __name__ == '__main__':
    path = '..\\..\\data\\task2\\DC_Crime_G.csv'
    df = pd.read_csv(path)

    if path == '..\\..\\data\\task2\\DC_Crime_G.csv':
        df['NEIGHBORHOOD_CLUSTER'] = df['NEIGHBORHOOD_CLUSTER'].astype(str)
        df['NEIGHBORHOOD_CLUSTER'] = df['NEIGHBORHOOD_CLUSTER'].apply(NEIGHBORHOOD_CLUSTER_convert)
        df['sector'] = df['sector'].astype(str)
        df['sector'] = df['sector'].apply(sector_convert)
        df['BLOCK_GROUP'] = df['BLOCK_GROUP'].astype(str)
        df['BLOCK_GROUP'] = df['BLOCK_GROUP'].apply(BLOCK_GROUP_convert)
        df['VOTING_PRECINCT'] = df['VOTING_PRECINCT'].astype(str)
        df['VOTING_PRECINCT'] = df['VOTING_PRECINCT'].apply(VOTING_PRECINCT_convert)
        df['ANC'] = df['ANC'].astype(str)
        df['ANC'] = df['ANC'].apply(ANC_convert)

        df = df.drop('Unnamed: 0', axis=1)
        correlation_matrix = df.corr(method='pearson')
        print(correlation_matrix)
        correlation_matrix.to_csv('..\\..\\data\\task2\\corr_G.csv')
