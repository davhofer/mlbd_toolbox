"""
data cleaning
imputing missing values
standardization
"""
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def remove_nan(df, columns=None, nanval=None):
    if nanval:
        def condition(data):
            return data != nanval 
    else:
        def condition(data):
            return ~data.isna()

    if columns:
        for c in columns:
            df = df[condition(df[c])]

    else:
        df = df[condition(df).all(axis=1)]
    return df

def standardize_features(df, columns=[]):
    if columns:
        for c in columns:
            df[c] = (df[c] - df[c].mean())/df[c].std()

        return df 
    return (df - df.mean())/df.std()

def min_max_scaling(df, columns: list):
    for c in columns:
        scaler = MinMaxScaler()
        df[c] = scaler.fit_transform(df[c].to_numpy().reshape(-1,1))
    return df




def remove_inactive_students(df, ts):
    df = df.fillna('NaN')
    
    #find all users weeks with 0 clicks on weekends and 0 clicks on weekdays during the first weeks of the semester
    df_first = ts[ts.week < 5]
    rows = np.where(np.logical_and(df_first.ch_total_clicks_weekend==0, df_first.ch_total_clicks_weekday == 0).to_numpy())[0]
    df_zero = df_first.iloc[rows,:]
    dropusers = np.unique(df_zero.user)

    ts = ts[ts.user.isin(dropusers)==False]
    df = df[df.user.isin(dropusers)==False]
    return df, ts

df_lq, ts = remove_inactive_students(df_lq, ts)


# Example for binning
def binning(df):
    learn_maps = {0: 'less than 10s', 1: 'less than 20s', 2: 'less than 30s', 3: 'less than 40s', 4: 'less than 50s'}
    df['bin_s_first_response'] = (df['ms_first_response'] // (10 * 1000)).map(learn_maps).fillna('other')
    return df


"""
column transformer:

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler,  OneHotEncoder

preprocessor = ColumnTransformer([
    ('categorical', OneHotEncoder(handle_unknown='ignore', drop = 'first'), ['group']),
    ('numerical', MinMaxScaler(),['studying_hours'])
])

preprocessor.fit_transform(X_train)

"""
