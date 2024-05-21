import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('..\\..\\data\\task3\\DC_Crime_Preprocessed.csv')
    df_normalized = (df - df.mean()) / df.std()
    df_normalized.to_csv('..\\..\\data\\task3\\DC_Crime_Normalized.csv', index=True)
