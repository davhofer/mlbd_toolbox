"""
aggregate features
TODO: look at lectures to see what else is relevant for this
"""


def flattening(df, id_col, time_col):
    pass 

def aggregation(df, id_col, time_col, method):
    assert method in ['sum', 'avg']
    if method == 'sum':
        df = df.groupby(id_col).sum()
    else:
        df = df.groupby(id_col).mean()

    df = df.reset_index()
    df = df.drop(columns=[time_col])
    return df
