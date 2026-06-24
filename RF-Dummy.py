# Random Forest: cancer (C) vs non-cancer (NC) from X_1–X_9
# 70:30 train/test split + 5-fold NESTED cross-validation

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     StratifiedKFold)
from sklearn.ensemble import RandomForestClassifier
import random

def getFeatures(data: pd.DataFrame, indices: list, features = []): # "give me the inputs for these rows"
    if features == []:# Features = x_col, we upgrade this to make sure that we can select the features we want out of the real spectrum
        features = [col for col in list(data.columns) if col[:2] == 'X_'] # The column name of the features we choose for this dummy dataset
    return data.loc[indices, features]

def getTarget(data: pd.DataFrame, indices: list, target: str = 'cnc'):
    return data.loc[indices, target]

# 1. Shuffling and splitting patients by patients ID (using all the data points per patient) instead of using the mean value (previous versino of code)
# Create a function to collapse one patient ID into a list of [[a ,b ,c], [d, e, f], ... [x, y, z]]
def split(data: pd.DataFrame, training_size: float = 0.7, shuffle: bool = True, random_state: int = 42):
    # Shuffle:bool = True is the setting for the split function (Should I shuffle the patients before splitting them into train and test?)
    patients = {}  # Starts an empty dictionary
    # The following code block explain the patient-grouping block
    # Go through every row, one at a time where index is a row number (0, 1, 2 ...)
    for index in range(len(data)):
        # .loc = location, this look up which patient that row belongs to
        pid = data.loc[index, 'pid']
        if pid not in patients:  # If it's the first time you've seen this patient, give them an empty list to hold their rows
            # Create an empty list to store the things as the value for the key "pid" in the dictionary
            patients[pid] = []
        # Add this row's number to that patient's list
        patients[pid].append(index)

    patients = list(patients.values())
    # It throws away the patient-id keys and keeps just the lists of row numbers
    # Make the list of "patients = [[0, 1, 2], [3, 4, 5], ...]"
    num_patients = len(patients)
    training_num = int(num_patients * training_size)
    testing_num = int(num_patients - training_num)

    # Splitting the patients
    if shuffle:
        # If shuffling is turned on (it is, by default), randomly reorders the list of patients
        random.shuffle(patients)
    training_patients = patients[:training_num]
    testing_patients = patients[training_num:]

    training_indices = []  # Start an empty list to collect all the training row numbers
    for patient in training_patients:
        # Go through each patient bundle and pours the patient's row numbers into a flat list
        training_indices += patient
    testing_indices = []
    for patient in testing_patients:
        testing_indices += patient

    return training_indices, testing_indices


# 2. Using the features information to predict C/NC
df = pd.read_csv("Dummy.csv")
features = [f"X_{i}" for i in range(1, 10)]


# 3. Split by patient: get the train/test row numbers, then pick out those rows
training_indices, testing_indices = split(df)  # Previous function in step 1.
# Training data
# Pick out a set of rows from the location of training_indices


# 4. Base model + the hyperparameter grid to search
rf = RandomForestClassifier(
    criterion='gini', class_weight="balanced", random_state=42, n_jobs=-1)
# class_weight="balanced" for there being more NC than C, n_jobs = -1 means using all cpu cores
param_grid = {
    "n_estimators":     [500],
    "max_depth":        [3, 5, 7],
    "min_samples_leaf": [1, 3, 5],
    "max_features": ["sqrt", "log2", None]
}

# 5. Nested cross-validation (run on the TRAINING set) ## ** wrtie our own K fold function so there will be no data leakage among the same pid **
# Inner 5-fold: grid searches and picks the best hyperparameters

# Build the inputs from the previous steps
X_train = getFeatures(df, training_indices) # DataFrame of X_1–X_9, index = training_indices
Y_train = getTarget(df, training_indices) # Series of 'cnc', index = training_indices
groups  = df.loc[training_indices, 'pid'] # Needed because the function needs to know which patient each row beongs to

def inner_k_fold_cv(X: pd.DataFrame, Y: pd.Series, groups: pd.Series, k: int = 5, shuffle=True, random_state=42):
    patients = {} # creating an empty dictionary that maps each patient's ID to a list of patient's row
    for idx in X.index: # looping over every row in X, one row at a time   # FIX: X not X_train (use the argument)
        pid = groups.loc[idx] # look up which patient this row belongs to   # FIX: do the lookup once
        if pid not in patients: # if it has not yet seen the patient (new patient), open an empty list for it
            patients[pid] = [] # create the empty list here
        patients[pid].append(idx) # add the current row to that patient's list
    patients = list(patients.values()) # should come out like: [[idx 0, 1, 2], [idx 3, 4, 5], ...] where .values() drops the pid keys

    # setting up the positions to shuffle and slice
    n_patients = len(patients) # count the number of patients
    indices = np.arange(n_patients) # make an array of the patient's position on the list

    # shuffle the position of patients in the list
    if shuffle:
        np.random.default_rng(random_state).shuffle(indices) # shuffle with reproducible random seed
    # deciding the fold size
    folds = np.array_split(indices, k) # split the position (indices), not patients

    splits = [] # prepare an empty list to collect training/testing set per fold
    for fold_num in range(k):
        test_patient  = folds[fold_num] # only one fold for test/validation
        train_patient = [] # create the loop for the rest of the folds to be training set
        for other_fold in range(k):
            if other_fold != fold_num: # if it is not the test fold, make them a training set
                for patient in folds[other_fold]:
                    train_patient.append(patient)

        # making the folds into list of row numbers
        test_indices  = []
        for patient in test_patient:
            test_indices += patients[patient] # add all of the patient's rows   # FIX: patient is a position → look up patients[patient]
        train_indices = []
        for patient in train_patient:
            train_indices += patients[patient] # same lookup for the training side
        splits.append((train_indices, test_indices))

    return splits

grid = GridSearchCV(rf, param_grid, cv=inner_k_fold_cv,
                    scoring="roc_auc", n_jobs=1, refit=True)
# rf = the estimator, param_grid = the hyperparameters to search, cv = the cross-validation strategy used to score each candidate,
# scoring="roc_auc" to decide which hyperparameter combination is the best, n_jobs=1 do one job at a time,
# refit=True to refit the best model on the whole training set after grid search.

# 6. Tune on the full training set, then test on the 30% testing set
# inner CV picks the best params (from training set)
grid.fit(getFeatures(df, training_indices), getTarget(df, training_indices))
print("="*30)
print()
print("Best hyperparameters:", grid.best_params_)
predicted = grid.best_estimator_.predict(getFeatures(df, testing_indices))

# 7. Report performance on the test set
print()

# Gini importance
best_rf = grid.best_estimator_
importance_table = pd.DataFrame(
    {"Importance": best_rf.feature_importances_}, index=features
)
print("\nGini importance:")
print(importance_table.round(3))
print()

# 8. Confusion matrix on the test set
cm = metrics.confusion_matrix(getTarget(df, testing_indices), predicted, labels=["NC", "C"])
cm_df = pd.DataFrame(cm,
                     index=["True NC", "True C"],
                     columns=["Predicted NC", "Predicted C"])
print("\nConfusion matrix:")
print(cm_df)
print()

# 9. Dxcover clinical matrics
# With labels=["NC","C"], the matrix is [[TN, FP], [FN, TP]].
tn, fp, fn, tp = cm.ravel()
print("Dxcover clinical matrics:")
print(f"Sensitivity (recall, C):  {tp/(tp+fn):.3f}")
print(f"Specificity (recall, NC): {tn/(tn+fp):.3f}")
print(f"PPV:                      {tp/(tp+fp):.3f}")
print(f"NPV:                      {tn/(tn+fn):.3f}")
print()
print("="*30)
