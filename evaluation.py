"""
lecture 3
regression metrics, 
r-squared
MAE, RMSE

lecture 4+
evaluation metrics,
confusion matrix
specifity, sensitivity
ROC, AUROC

lecture 5: information criteria
AIC, BIC



visualization of results, error bars
"""

import numpy as np
import math

def r_squared(y_true, y_hat):
    y_true = np.array(y_true) 
    y_hat = np.array(y_hat)

    SS_res = ((y_true - y_hat)**2).sum()

    y_bar = y_true.mean()
    SS_tot = ((y_true - y_bar)**2).sum()

    return 1 - (SS_res/SS_tot)

def mae(y_true, y_hat):
    y_true = np.array(y_true) 
    y_hat = np.array(y_hat)

    return np.abs(y_true - y_hat).mean()


def mse(y_true, y_hat):
    y_true = np.array(y_true) 
    y_hat = np.array(y_hat)

    return ((y_true - y_hat)**2).mean()

def rmse(y_true, y_hat):
    return np.sqrt(mse(y_true, y_hat))


# from sklearn.metrics import confusion_matrix
# in the binary case (two classes):
# tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()



def accuracy(y_true, y_hat):
    y_true = np.array(y_true) 
    y_hat = np.array(y_hat)

    return (y_true == y_hat).sum()/len(y_true)


def specificity(y_true, y_hat):
    pass 

def sensitivity(y_true, y_hat):
    pass 

# akaike information criterion
def aic(loglik, N, d):
    return -(2/N) * loglik + 2 * d / N 

# bayesian information criterion
def bic(loglik, N, d):
    return -2 * loglik + math.log(N) * d


# Kullback-Leibler Divergence
def kl_divergence(P, Q, X):
    return sum([P(x) * np.log(P(x)/Q(x)) for x in X])



"""
compute ROC (lecture 04 slide 70):
    given classifier f(x)
    for threshold values t = [0, 0.1, ..., 1.0]:
        make predictions using threshold t (f(x) >= t -> 1, f(x) < t -> 0)
        compute TPR and FPR

    plot the curve of TPR on y axis vs FPR on x axis





from sklearn.metrics import roc_auc_score, balanced_accuracy_score

"""

