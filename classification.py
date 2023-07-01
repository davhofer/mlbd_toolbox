"""
WEEK 4 notebooks

decision trees -> greedy heursitic with entropy, information gain
random forest -> ensemble method
k-nearest neighbor
logistic regression

TODO: how to actually make classificatio npredictions with them???
"""
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

import statsmodels.api as sm
import statsmodels.formula.api as smf

from scipy.spatial.distance import pdist, cdist, squareform

# construct tree by hand: lecture 4 slide 15
def decision_tree():
    clf = tree.DecisionTreeClassifier(max_depth=2, random_state=0, criterion='entropy')
    # clf.fit(X_train, y_train)
    # tree.plot_tree(clf, feature_names=X_train.columns)
    # clf.predict(X_test)
    
    return clf

def random_forest():
    rf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0, criterion='entropy')
    return rf

def knn_classif():
    knn = KNeighborsClassifier(n_neighbors=5, metric='precomputed')
    return knn

def knn_manual(df, x, label: str, k: int, dist_func=None):
    if not dist_func:
        # euclidean distance function
        def dist_func(a, b):
            return np.sqrt(((np.array(a) - np.array(b))**2).sum())
    out = df[[label]]
    X = df.drop(columns=[label])
    # numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    # X = X.select_dtypes(include=numerics)


    out['dist'] = X.apply(lambda row: dist_func(row, x), axis=1)

    top_k = out.sort_values('dist', ascending=True)[:k]

    return top_k[label].mode()[0]


def logistic_regression():
    clf = LogisticRegression(random_state=0)
    return clf


def compute_scores(clf, X_train, y_train, X_test, y_test, roundnum = 3):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy =  balanced_accuracy_score(y_test, y_pred)
    
    y_pred_proba = clf.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, y_pred_proba)
    
    return round(accuracy,roundnum), round(auc,roundnum)
