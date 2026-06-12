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


def split(data: pd.DataFrame, training_size: float = 0.7, shuffle: bool = True, random_state: int = 42):
    patients = {}
    for index in range(len(data)):
        pid = data.loc[index, 'pid']
        if pid not in patients:
            patients[pid] = []
        patients[pid].append(index)

    patients = list(patients.values())
    num_patients = len(patients)
    training_num = int(num_patients * training_size)
    testing_num = int(num_patients - training_num)

    # splitting
    if shuffle:
        random.shuffle(patients)
    training_patients = patients[:training_num]
    testing_patients = patients[training_num:]

    training_indices = []
    for patient in training_patients:
        training_indices += patient
    testing_indices = []
    for patient in testing_patients:
        testing_indices += patient

    return training_indices, testing_indices


# 1. Load data and average the 3 replicates per patient
#     Each patient has 3 repeated rows. Averaging gives ONE row per patient,
#     so replicates of the same person can't leak across the split / folds
#     (which would otherwise make the scores look better than they are).
df = pd.read_csv("Dummy.csv", dtype={'pid': str})
features = [f"X_{i}" for i in range(1, 10)]

training_indices, testing_indices = split(df)
print(len(training_indices), len(testing_indices))
exit()


# 2. Define the features (X) and the label (y), and patient ID (groups)
X = df[features]
y = df["cnc"]  # "C" or "NC"
groups = df["pid"]  # patient ID for grouping in cross-validation

# 3. 70:30 train/test split (split by patients)
patient_label = df.groupby("pid")["cnc"].first()  # force one label per patient
train_pids, test_pids = train_test_split(patient_label.index, test_size=0.3,
                                         stratify=patient_label.values, random_state=42)

# 4. Base model + the hyperparameter grid to search
# class_weight="balanced" for there being more NC than C.
rf = RandomForestClassifier(class_weight="balanced", random_state=42)
param_grid = {
    "n_estimators":     [300, 500],
    "max_depth":        [3, 5, 7],
    "min_samples_leaf": [1, 3, 5],
    "max_features": ["sqrt", "log2", None]
}

# 5. Nested cross-validation (run on the TRAINING set)
#    Inner 5-fold: grid searches and picks the best hyperparameters
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(rf, param_grid, cv=cv,
                    scoring="f1_weighted", n_jobs=1, refit=True)
# rf = the estimator, param_grid = the hyperparameters to search, cv = the cross-validation strategy used to score each candidate,
# scoring="f1_weighted" to decide whuch hyperparameter combination is the best using F1 score, n_jobs=-1 to use all the cores,
# refit=True to refit the best model on the whole training set after grid search, verbose=0 for no output during fitting, pre_dispatch='2*n_jobs'
# to control how many jobs get dispatched during parallel execution, error_score=np.nan to assign NaN if an error occurs during fitting, return_train_score=False to not return training scores.

# 6. Tune on the full training set, then test on the 30% testing set
# inner CV picks the best params (from training set)
grid.fit(X_train, y_train)
print("Best hyperparameters:", grid.best_params_)
predicted = grid.best_estimator_.predict(X_test)

# 7. Report performance on the test set
print(metrics.classification_report(y_test, predicted))
print(f"Test accuracy: {metrics.accuracy_score(y_test, predicted):.3f}")

# 8. Confusion matrix on the test set
cm = metrics.confusion_matrix(y_test, predicted, labels=["NC", "C"])
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm",
            xticklabels=["NC", "C"], yticklabels=["NC", "C"])
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# 9. Dxcover clinical matrics
# With labels=["NC","C"], the matrix is [[TN, FP], [FN, TP]].
tn, fp, fn, tp = cm.ravel()
print(f"Sensitivity (recall, C):  {tp/(tp+fn):.3f}")
print(f"Specificity (recall, NC): {tn/(tn+fp):.3f}")
print(f"PPV (precision, C):       {tp/(tp+fp):.3f}")
print(f"NPV:                      {tn/(tn+fn):.3f}")
