# Random Forest: cancer (C) vs non-cancer (NC) from X_1–X_9
# 70:30 train/test split + 5-fold NESTED cross-validation

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     StratifiedKFold, cross_val_score)
from sklearn.ensemble import RandomForestClassifier

# 1. Load data and average the 3 replicates per patient
#     Each patient has 3 repeated rows. Averaging gives ONE row per patient,
#     so replicates of the same person can't leak across the split / folds
#     (which would otherwise make the scores look better than they are).
df = pd.read_csv("Dummy.csv")
features = [f"X_{i}" for i in range(1, 10)]
patients = df.groupby(["pid", "cnc"], as_index=False)[features].mean()

# 2. Define the features (X) and the label (y)
X = patients[features]
y = patients["cnc"]  # "C" or "NC"

# 3. 70:30 train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)  # stratify to keep class balance in train/test

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

grid = GridSearchCV(rf, param_grid, cv=cv, scoring="f1_weighted", n_jobs=1, refit=True,
                    verbose=0, pre_dispatch='2*n_jobs', error_score=np.nan, return_train_score=False)
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
