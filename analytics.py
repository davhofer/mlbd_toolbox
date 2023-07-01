"""
univariate and multivariate analsis,
correlation, mutual information
time series

visualizations?
"""
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
# import scipy as sp
from scipy import stats
# from scipy.stats import skewnorm
import seaborn as sns
import numpy as np
# from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
# from sklearn.preprocessing import LabelEncoder 


"""
Feature exploration and visualization
"""


def twodata_barplot(title, array1, array2, labels, name1, name2, ylabel=None, xlabel=None):
        
    X_ticks = np.arange(len(labels))

    plt.bar(X_ticks - 0.2, array1, 0.4, label=name1)


    plt.bar(X_ticks + 0.2, array2, 0.4, label=name2)
    plt.xticks(X_ticks, labels)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()


def single_barplot(title, data, labels, legend_name):
        
    X_ticks = np.arange(len(labels))

    plt.bar(X_ticks, data, 0.4, label=legend_name)


    plt.xticks(X_ticks, labels)
    plt.ylabel(title)
    plt.legend()


def simple_plot(title, data, labels, ylabel=None, xlabel=None):
    plt.bar(labels, data)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)






def make_barplot(df):
    sns.barplot(x='Method', y='Accuracy', data=df, errorbar='ci')
    # does the same work with standard deviation?
    # sns.barplot(x='Method', y='Accuracy', data=df, errorbar='sd')

def plot_features(df, hue = None):
    continuous_cols = list(df._get_numeric_data().columns)
    categorical_cols =  list(df.select_dtypes(include=['O']).columns.values)

    rows = np.ceil(len(df.columns)/3).astype(int)
    fig, axes = plt.subplots(rows, 3, figsize=(15,5*rows))
    for i, col in enumerate(df.columns):
        ax = axes[i // 3, i % 3]
        if col in continuous_cols:
            sns.histplot(data=df, x = col, ax=ax,  kde=True, hue= hue) 
        elif col in categorical_cols:
            sns.countplot(data=df, x=col, ax=ax, hue = hue)
        else:
            print(col)
        ax.set(xlabel=col, ylabel='Count', title= 'Distribution {}'.format(col))

    fig.tight_layout()
    plt.show()

    
def plot_time_series(df, time_attr, hue=None):
    continuous_cols = list(df._get_numeric_data().columns)

    rows = np.ceil(len(continuous_cols)/3).astype(int)
    fig, axes = plt.subplots(rows, 3, figsize=(15,5*rows))
    for i, col in enumerate(continuous_cols):
        ax = axes[i // 3, i % 3]
        sns.lineplot(data=df, x=time_attr, y=col, ax = ax, errorbar='sd', hue=hue)
        ax.set(xlabel=time_attr, ylabel=col, title= 'Time series {}'.format(col))

    fig.tight_layout()
    plt.show()



def plot_time_series_single(df, x, y, hue=None):
    ax = sns.lineplot(data=df, x=x, y=y, errorbar='sd', hue=hue)
    plt.show()


def get_feature_stats(df):
    numerical = df.describe(include= ['float64'])
    categorical = df.describe(include= ['object'])
    stats = pd.concat([numerical, categorical])
    #stats = df.describe(include= 'all') # alternative
    
    # Select the desired statistics
    stats = stats.loc[['mean', 'std', 'min', '25%', '50%', '75%', 'max', 'unique', 'top', 'freq']]

    # maybe isna() instead of isnull()
    percentage = df.isnull().sum(axis = 0)*100 / len(df)
    stats.loc['missing_values'] = np.array(percentage)
    return stats

def plot_correlation(df, correlation_method='pearson'):
    corr = np.round(df.corr(method=correlation_method), 3)
    mask = np.tril(corr)
    ax = plt.axes()
    heatmap = sns.heatmap(corr, annot=True, mask=mask, cmap='RdBu_r')
    ax.set_title('Correlation between variables')
    plt.show()


def plot_feature_by_category(df, feature, category, ylim, bins):
    df = df[~df[category].isna()]
    n = df[category].nunique()
    values = list(df[category].unique())

    xlim = (df[feature].min(), df[feature].max())

    f, axarr = plt.subplots(nrows=1,ncols=(n+1), figsize=(20,5))

    sns.histplot(data= df, x=feature, hue=category, ax = axarr[0], bins=bins, binrange = xlim)
    axarr[0].set(title='All', ylim=(0,ylim),  xlim=xlim)

    print(axarr[0])
    
    for i, val in enumerate(values):

        sns.histplot(data= df[df[category]==val], x  = feature,  ax = axarr[i+1], bins=bins, binrange = xlim)
        axarr[i+1].set(title=str(val), ylim=(0, ylim),  xlim=xlim)

    plt.show()
    
# plot_by_category(df, 'time_in_problem', 'gender', ylim = 50, bins =9)


# NOTE: z should be categorical
def three_features_plot(df, x, y, z):
    sns.scatterplot(data = df, y = y, x = x, hue = z)



"""
Univariate analysis
"""
def visualize_value_counts(df, category):
    counts = df[category].value_counts(dropna=False)
    ax = sns.countplot(data=df, x=category)
    ax.set(xlabel=category, ylabel='Count')
    plt.show()
    return pd.DataFrame({"Category": counts.index, "Count": counts.tolist(), "Count %": counts.tolist()/np.sum(counts.tolist()) })


def test_normality(data):
    k2, p = stats.normaltest(data)
    alpha = 0.01
    print("p = {:g}".format(p))
    if p < alpha:  # null hypothesis: x comes from a normal distribution
        print("The null hypothesis can be rejected")
    else:
        print("The null hypothesis cannot be rejected")


"""
Multivariate analsis
"""
def make_scatterplot(df, x, y, hue=None):
    sns.scatterplot(data=df, y = y, x = x, hue=hue)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()

def compute_correlation(df, x, y, method='pearson'):
    df = df[[x, y]]
    return df.corr(method=method)

def compute_pearsonr(df, x, y):
    return stats.pearsonr(df[x], df[y])


def mutual_information_discrete(df, categorical_x, categorical_y):

    counts_xy = df[[categorical_x, categorical_y]].value_counts()
    d_xy = pd.DataFrame((counts_xy/ (counts_xy.sum()))).reset_index()
    d_xy.columns = [categorical_x, categorical_y, 'pxy']

    counts_x = df[categorical_x].value_counts()
    dx = pd.DataFrame((counts_x/ (counts_x.sum()))).reset_index()
    dx.columns = [categorical_x, 'px']

    counts_y = df[categorical_y].value_counts()
    dy = pd.DataFrame(counts_y/ (counts_y.sum())).reset_index()
    dy.columns = [categorical_y, 'py']

    d_mi = d_xy.merge(dx, on=categorical_x, how='left').merge(dy, on=categorical_y, how='left')

    mi = 0
    for i, row in d_mi.iterrows():
        if row['pxy']>0:
            mi += row['pxy']*np.log(row['pxy']/(row['px']*row['py']))

    return mi

def mutual_information_continous(df, x,y, bins=None):
    if not bins:
        bins = np.floor(np.sqrt(df.shape[0]/5))
        bins = int(bins)

    c_xy = np.histogram2d(df[x], df[y], bins)[0]
    g, p, dof, expected = chi2_contingency(c_xy, lambda_="log-likelihood")
    mi = 0.5 * g / c_xy.sum()
    return mi




