"""
Pipelines, cross validation, bootstrapping

"""
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, auc
from sklearn.utils import resample

# With the cross_validate function, the SciKit Learn library automatically uses stratification across folds with the "cv" argument. 
# In the background, it's using the StratifiedKFold function with 10 folds.
# We pass in our desired metrics ("accuracy", "roc_auc") for evaluation in the "scoring" argument.

""" cross validation """
scores = cross_validate(clf, X, y, cv=5, scoring=['accuracy', 'roc_auc'])


# We compute a grid search across the following parameter space
parameters = {
    'n_estimators': [20, 50, 100],
    'criterion': ['entropy', 'gini'],
    'max_depth': np.arange(3, 7),
    'min_samples_split': [2],
    'min_samples_leaf': [1],
}
# This cell takes ~3 minutes to run

# Inner cross validation loop
clf = GridSearchCV(RandomForestClassifier(random_state=42), parameters, cv=10)

# Outer cross validation loop
scores_nested_cv = cross_validate(clf, X, y, cv=3, scoring=['accuracy', 'roc_auc'])



""" bootstrapping """
# It is important to sample with:
# (1) the same size as the initial df (df_size) 
# (2) with replacement (replace=True)
# for the bootstrap samples to be representative.
df_size = len(df_lq)
B = 100

# Generate B bootstrap samples of the dataset with replacement.
samples = [resample(X, y, replace=True, n_samples=df_size) for b in range(B)]

# Train a random forest classifier for each bootstrap set
clfs = [RandomForestClassifier(random_state=42).fit(X_b, y_b) for X_b, y_b in samples]
