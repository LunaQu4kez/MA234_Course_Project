import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def extract_corr(s):
    index = s.columns.get_loc('SHIFT_day')
    s= s.iloc[:, index :]
    correlation_matrix = s.corr()
    correlation_matrix=correlation_matrix.round(2)
    return correlation_matrix

if __name__ == '__main__':
    df = pd.read_csv('..\\..\\data\\task2\\DC_Crime_Preprocessed.csv')
    grouped = df.groupby('YEAR').apply(extract_corr)

    dfs = {}
    for name, group in grouped:
        dfs[name] = group
