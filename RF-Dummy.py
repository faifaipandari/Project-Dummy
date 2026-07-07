# Random Forest: cancer (C) vs non-cancer (NC) from X_1–X_9
# 70:30 train/test split + 5-fold NESTED cross-validation

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, auc, make_scorer
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, PredefinedSplit)
from sklearn.ensemble import RandomForestClassifier
import random
from copy import deepcopy
import json

# ===========================================
# Single demonstration run (one split, with 5-fold inner CV inside GridSearchCV)
# ===========================================

# "Give me the inputs for these rows"


def getFeatures(data: pd.DataFrame, indices: list, features=[]):
    if features == []:  # Features = x_col, we upgrade this to make sure that we can select the features we want out of the real spectrum
        # The column name of the features we choose for this dummy dataset
        features = [col for col in list(data.columns) if col[:2] == 'X_']
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
        random.seed(random_state)
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

# 5. Nested cross-validation (run on the TRAINING set)
# Inner 5-fold: grid searches and picks the best hyperparameters

# Build the inputs from the previous steps (for cross validation, only pull 70% of the dataset from "training_indices")
X_train = getFeatures(df, training_indices)  # dataFrame of X_1–X_9
Y_train = getTarget(df, training_indices)  # series of 'cnc'
# keep the patients with the same ID together
groups = df.loc[training_indices, 'pid']


def inner_k_fold_cv(df: pd.DataFrame, indices: list, k: int = 5, shuffle: bool = True, random_state: int = 42):
    # groupping the roq by patients ID (pid)
    patients = []  # creating an empty dictionary that maps each patient's ID to a list of patient's row
    # looping over every position in X, one row at a time (need to be position, not the orginal df label)
    for position, idx in enumerate(getFeatures(df, indices).index):
        # look up which patient this row belongs to   # FIX: do the lookup once
        pid = df.loc[training_indices, 'pid'].loc[idx]
        # if it has not yet seen the patient (new patient), open an empty list for it
        if pid not in patients:
            patients.append(pid)

    # setting up the positions to shuffle and slice
    n_patients = len(patients)  # count the number of patients

    # shuffle the position of patients in the list, not the rows
    if shuffle:
        random.seed(random_state)  # shuffle with reproducible random seed
        random.shuffle(patients)

    # divide the patients into 5 buckets
    pid_folds = {}
    for fold in range(k):  # we set k = 5
        # check ** ends at 20, stats at 20 ** is it duplicated?
        pid_folds[fold] = patients[int(
            (n_patients / k) * fold):int((n_patients / k) * (fold + 1))]
        # 5-folds are 0 (patient pid starts at 0: ends at 20), 1 (patient pid starts at 20: ends at 40), ...

    # turn patient-buckets into a per-row label list as 'PredefinedSplit' as it doesn't want patient buckets, it wants one fold number per row, in the same order as the data
    folds = []
    for index in indices:
        pid = df.loc[index, 'pid']
        for fold in pid_folds:
            if pid in pid_folds[fold]:
                folds.append(fold)

    # SANITY CHECK: patient's label in the folds - only printed when mismatch
    if len(indices) != len(folds):
        print("  WARNING: fold labelling mismatch!")

    return PredefinedSplit(folds)
    # 'PredefinedSplit' creates one fold for each distinct value it sees in the list we hand it.
    # So the number of folds is decided entirely by how many unique numbers end up in folds.

def calculateMetrics(cm):
    metrics = {}
    if isinstance(cm, np.ndarray):
        tn, fp, fn, tp = cm.ravel().tolist()
    else:
        tn, fp, fn, tp = cm['tn'], cm['fp'], cm['fn'], cm['tp']
    metrics['acc'] = ((tp + tn) / (tp + fp + tn + fn))
    metrics['sens'] = (tp / (tp + fn))
    metrics['spec'] = (tn / (tn + fp))
    metrics['ppv'] = (tp / (tp + fp))
    metrics['npv'] = (tn / (tn + fn))
    return metrics


# creating our own function for the per-fold CV performance (2.1 in the outer-loop CV)

def aucScoring(model, features, targets):
# Gridsearch passes three things: model, features, and target to this function
    probs = model.predict_proba(features) # getting probability of "C"
    # .class = pick the column that we want to use: "C"/nc column
    c_index = list(model.classes_).index("C")
    preds = probs[:, c_index]  # keep only 'C' as prob of c+nc = 1
    targets = [1 if target == 'C' else 0 for target in targets] # converting labels to 0/1 and computing AUC
    return roc_auc_score(targets, preds)


# searching the best method
scorer = {'auc': aucScoring,
          'tn': make_scorer(lambda yt, yp: confusion_matrix(yt, yp, labels=["NC", "C"])[0, 0]),
          'fp': make_scorer(lambda yt, yp: confusion_matrix(yt, yp, labels=["NC", "C"])[0, 1]),
          'fn': make_scorer(lambda yt, yp: confusion_matrix(yt, yp, labels=["NC", "C"])[1, 0]),
          'tp': make_scorer(lambda yt, yp: confusion_matrix(yt, yp, labels=["NC", "C"])[1, 1])}

inner_cv = inner_k_fold_cv(df, training_indices)
grid = GridSearchCV(rf, param_grid, cv=inner_cv,
                    scoring=scorer, n_jobs=-1, refit = 'auc')
# rf = the estimator, param_grid = the hyperparameters to search, cv = the cross-validation strategy used to score each candidate,
# scoring="roc_auc" to decide which hyperparameter combination is the best, n_jobs=1 do one job at a time,
# refit=True to refit the best model on the whole training set after grid search.

# 6. Tune on the full training set, then test on the 30% testing set
# inner CV picks the best params (from training set)
grid.fit(getFeatures(df, training_indices), getTarget(df, training_indices))


cv_metrics = []

print(grid.best_score_)

cm = {'tn': np.ndarray(np.mean(grid.cv_results_['split0_test_tn']), np.mean(grid.cv_results_['split1_test_tn']), np.mean(grid.cv_results_['split2_test_tn']), np.mean(grid.cv_results_['split3_test_tn']), np.mean(grid.cv_results_['split4_test_tn'])), 
      'fp': np.ndarray(np.mean(grid.cv_results_['split0_test_fp']), np.mean(grid.cv_results_['split1_test_fp']), np.mean(grid.cv_results_['split2_test_fp']), np.mean(grid.cv_results_['split3_test_fp']), np.mean(grid.cv_results_['split4_test_fp'])), 
      'fn': np.ndarray(np.mean(grid.cv_results_['split0_test_fn']), np.mean(grid.cv_results_['split1_test_fn']), np.mean(grid.cv_results_['split2_test_fn']), np.mean(grid.cv_results_['split3_test_fn']), np.mean(grid.cv_results_['split4_test_fn'])), 
      'tp': np.ndarray(np.mean(grid.cv_results_['split0_test_tp']), np.mean(grid.cv_results_['split1_test_tp']), np.mean(grid.cv_results_['split2_test_tp']), np.mean(grid.cv_results_['split3_test_tp']), np.mean(grid.cv_results_['split4_test_tp']))}
metrics = calculateMetrics(cm)
for metric in metrics:
    metrics[metric] = list(metrics[metric])
metrics['roc_auc'] = list(grid.cv_results_['mean_test_auc'])
cv_metrics.append(metrics)

# spread of the inner CV scores (across all fold evaluations)
print("="*30)
print()
print("Inner CV scores (mean ± std over", len(cv_metrics), "evaluations):")
for key in ['acc', 'sens', 'spec', 'ppv', 'npv', 'roc_auc']:
    # pull this metric out of every dict
    values = [m[key] for m in cv_metrics]
    print(f"  {key:8s} {np.mean(values):.3f} ± {np.std(values):.3f}")

with open('cv_metrics.json', 'w') as f:
    f.write(json.dumps(cv_metrics))

print()
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
cm = confusion_matrix(
    getTarget(df, testing_indices), predicted, labels=["NC", "C"])
cm_df = pd.DataFrame(cm,
                     index=["True NC", "True C"],
                     columns=["Predicted NC", "Predicted C"])
print("\nConfusion matrix:")
print(cm_df)

# accuracy
# adapted from https://stackoverflow.com/a/42471653 (CC BY-SA 4.0)
y_test = getTarget(df, testing_indices)
score = accuracy_score(y_test, predicted)
print(f"Accuracy: {score:.3f}")
print()

# 9. Dxcover clinical matrics
# With labels=["NC","C"], the matrix is [[TN, FP], [FN, TP]].
tn, fp, fn, tp = cm.ravel()

# SANITY CHECK: compare sklearn's accuracy_score with the manual formula
manual_acc = (tp + tn) / (tp + tn + fp + fn)
print(f"[check] sklearn acc={score:.6f}  manual acc={manual_acc:.6f}")
print ()

print("Dxcover clinical matrics:")
print(f"Sensitivity (recall, C):  {tp/(tp+fn):.3f}")
print(f"Specificity (recall, NC): {tn/(tn+fp):.3f}")
print(f"PPV:                      {tp/(tp+fp):.3f}")
print(f"NPV:                      {tn/(tn+fn):.3f}")
print()

# ROC curve
# Adapted from https://stackoverflow.com/a/38467407 (CC BY-SA 4.0)

# the test inputs/labels
X_test = getFeatures(df, testing_indices)
y_test = getTarget(df, testing_indices)

# calculate the fpr (false positive rate) and tpr (true positive rate) for all thresholds
# .predict_proba get the model confidence score of the column (C/NC) as probability per patient
probs = best_rf.predict_proba(X_test)
# .class = pick the column that we want to use: "C"/nc column
c_index = list(best_rf.classes_).index("C")
preds = probs[:, c_index]  # keep only 'C' as prob of c+nc = 1
fpr, tpr, threshold = roc_curve(
    y_test, preds, pos_label="C")  # build the ROC curve
y_bin = [1 if y == "C" else 0 for y in y_test]   # C -> 1, NC -> 0
roc_auc = roc_auc_score(y_bin, preds)  # calculate AUC value

# plot ROC curve
specificity = 1 - fpr  # convert FPR -> Specificity
plt.title('Receiver Operating Characteristic of the Inner-loop Nested CV')
plt.plot(1-fpr, tpr, 'b', label='AUC = %0.6f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([1, 0], [0, 1], 'r--')  # diagonal, now in spec space
plt.xlim([1, 0])  # 1.0 on left, 0.0 on right
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig(f'Receiver_Operating_Characteristic_of_the_Inner-loop_Nested_CV.png')
print("="*30)
print()

# ===========================================
# # Outer loop - repeat the whole experiment n_runs times
# (the 5-fold inner CV lives inside grid.fit each time)
# ===========================================

# Set n = 51 for iterations
n_runs = 51

# create empty lists to collect each run's results
acc_list = []
sens_list = []
spec_list = []
ppv_list = []
npv_list = []
auc_list = []
te_cm_list = []

# create empty lists for per-fold validation
per_fold_cm_list = []
per_fold_auc_list = []

tr_metrics = {'acc': [], 'sens': [],
              'spec': [], 'ppv': [], 'npv': [], 'auc': [], 'cm': []}

# create empty lists to store each run's ROC curve points (for the plots)
fpr_list = []
tpr_list = []

print(f"Running {n_runs} iterations")

for i in range(n_runs):
    # make this block of code run different from the others and reproducible
    random.seed(i)

    # 1. new train/test split (different each run because the seed changed)
    training_indices, testing_indices = split(df, random_state=i)

    # SANITY CHECK: no patient in both train and test - only printed when mismatch
    train_pids = set(df.loc[training_indices, 'pid'])
    test_pids = set(df.loc[testing_indices, 'pid'])
    if len(train_pids & test_pids) > 0:
        print(f"  WARNING run {i+1}: patient leakage!")

    # 2. inner CV + grid search on this run's training set
    rf = RandomForestClassifier(
        criterion='gini', class_weight="balanced", random_state=i, n_jobs=-1)
    inner_cv = inner_k_fold_cv(df, training_indices, random_state=i)
    grid = GridSearchCV(rf, param_grid, cv=inner_cv,
                        scoring="roc_auc", n_jobs=-1, refit=True)
    grid.fit(getFeatures(df, training_indices),
             getTarget(df, training_indices))

    # 2.2. Get the training metrics
    best_rf = grid.best_estimator_
    predicted = best_rf.predict(getFeatures(df, training_indices))
    y_test = getTarget(df, training_indices)
    cm = confusion_matrix(y_test, predicted, labels=["NC", "C"])
    tn, fp, fn, tp = cm.ravel()
    tr_metrics['cm'].append(
        {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)})
    tr_metrics['acc'].append(accuracy_score(y_test, predicted))
    tr_metrics['sens'].append(tp / (tp + fn))
    tr_metrics['spec'].append(tn / (tn + fp))
    tr_metrics['ppv'].append(tp / (tp + fp))
    tr_metrics['npv'].append(tn / (tn + fn))
    probs = best_rf.predict_proba(getFeatures(df, training_indices))
    c_index = list(best_rf.classes_).index("C")
    preds = probs[:, c_index]
    fpr, tpr, threshold = roc_curve(y_test, preds, pos_label="C")
    y_bin = [1 if y == "C" else 0 for y in y_test]
    tr_metrics['auc'].append(roc_auc_score(y_bin, preds))

    # 3. predict on this run's test set
    best_rf = grid.best_estimator_
    predicted = best_rf.predict(getFeatures(df, testing_indices))
    y_test = getTarget(df, testing_indices)

    # 4. confusion matrix -> TN, FP, FN, TP
    cm = confusion_matrix(y_test, predicted, labels=["NC", "C"])
    tn, fp, fn, tp = cm.ravel()
    te_cm_list.append(
        {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)})

    # 5. the performance metrics
    accuracy = accuracy_score(y_test, predicted)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)

    # 6. ROC curve + AUC
    probs = best_rf.predict_proba(getFeatures(df, testing_indices))
    c_index = list(best_rf.classes_).index("C")
    preds = probs[:, c_index]
    fpr, tpr, threshold = roc_curve(y_test, preds, pos_label="C")
    y_bin = [1 if y == "C" else 0 for y in y_test]
    roc_auc = roc_auc_score(y_bin, preds)

    # 7. save everything from this run
    acc_list.append(accuracy)
    sens_list.append(sensitivity)
    spec_list.append(specificity)
    ppv_list.append(ppv)
    npv_list.append(npv)
    auc_list.append(roc_auc)
    fpr_list.append(fpr)
    tpr_list.append(tpr)

    # 8. print this run's row in the "sheet"
    print(f"Run {i+1:2d} | Acc {accuracy:.3f} | Sens {sensitivity:.3f} | "
          f"Spec {specificity:.3f} | PPV {ppv:.3f} | NPV {npv:.3f} | AUC {roc_auc:.3f}")

# Averages over all runs
print()
print(f"Averages over {n_runs} runs (mean ± std):")
print(f"Accuracy:    {np.mean(acc_list):.3f} ± {np.std(acc_list):.3f}")
print(f"Sensitivity: {np.mean(sens_list):.3f} ± {np.std(sens_list):.3f}")
print(f"Specificity: {np.mean(spec_list):.3f} ± {np.std(spec_list):.3f}")
print(f"PPV:         {np.mean(ppv_list):.3f} ± {np.std(ppv_list):.3f}")
print(f"NPV:         {np.mean(npv_list):.3f} ± {np.std(npv_list):.3f}")
print(f"AUC:         {np.mean(auc_list):.3f} ± {np.std(auc_list):.3f}")
print()
print("="*30)

te_metrics = {'acc': acc_list, 'sens': sens_list, 'spec': spec_list,
              'ppv': ppv_list, 'npv': npv_list, 'auc': auc_list, 'cm': te_cm_list}

# making separate .json files for training and testing
# training file: raw per-run data + averages + stds
with open('tr_metrics.json', 'w') as f:
    f.write(json.dumps(tr_metrics))
    f.write('\n')
    # line 2: the averages (mean of each metric)
    f.write(json.dumps({'average': {
        'acc':  np.mean(tr_metrics['acc']),
        'sens': np.mean(tr_metrics['sens']),
        'spec': np.mean(tr_metrics['spec']),
        'ppv':  np.mean(tr_metrics['ppv']),
        'npv':  np.mean(tr_metrics['npv']),
        'auc':  np.mean(tr_metrics['auc']),
        'cm':   {"TP": np.mean([c["TP"] for c in tr_metrics['cm']]),
                 "TN": np.mean([c["TN"] for c in tr_metrics['cm']]),
                 "FP": np.mean([c["FP"] for c in tr_metrics['cm']]),
                 "FN": np.mean([c["FN"] for c in tr_metrics['cm']])}}}))
    f.write('\n')
    # line 3: the standard deviations
    f.write(json.dumps({'std': {
        'acc':  np.std(tr_metrics['acc']),
        'sens': np.std(tr_metrics['sens']),
        'spec': np.std(tr_metrics['spec']),
        'ppv':  np.std(tr_metrics['ppv']),
        'npv':  np.std(tr_metrics['npv']),
        'auc':  np.std(tr_metrics['auc']),
        'cm':   {"TP": np.std([c["TP"] for c in tr_metrics['cm']]),
                 "TN": np.std([c["TN"] for c in tr_metrics['cm']]),
                 "FP": np.std([c["FP"] for c in tr_metrics['cm']]),
                 "FN": np.std([c["FN"] for c in tr_metrics['cm']])}}}))

# test file: original line + averages line
with open('te_metrics.json', 'w') as f:
    f.write(json.dumps(te_metrics))
    f.write('\n')
    # line 2: the averages (mean of each metric)
    f.write(json.dumps({'average': {
        'acc':  np.mean(acc_list),
        'sens': np.mean(sens_list),
        'spec': np.mean(spec_list),
        'ppv':  np.mean(ppv_list),
        'npv':  np.mean(npv_list),
        'auc':  np.mean(auc_list),
        'cm':   {"TP": np.mean([c["TP"] for c in te_metrics['cm']]),
                 "TN": np.mean([c["TN"] for c in te_metrics['cm']]),
                 "FP": np.mean([c["FP"] for c in te_metrics['cm']]),
                 "FN": np.mean([c["FN"] for c in te_metrics['cm']])}}}))
    f.write('\n')
    # line 3: the standard deviations
    f.write(json.dumps({'std': {
        'acc':  np.std(acc_list),
        'sens': np.std(sens_list),
        'spec': np.std(spec_list),
        'ppv':  np.std(ppv_list),
        'npv':  np.std(npv_list),
        'auc':  np.std(auc_list),
        'cm':   {"TP": np.std([c["TP"] for c in te_metrics['cm']]),
                 "TN": np.std([c["TN"] for c in te_metrics['cm']]),
                 "FP": np.std([c["FP"] for c in te_metrics['cm']]),
                 "FN": np.std([c["FN"] for c in te_metrics['cm']])}}}))


# Combined plot: all individual runs (faint) + mean curve (bold) on one figure
mean_fpr = np.linspace(0, 1, 100)   # 100 shared x-axis points
tpr_interp_list = []
for i in range(n_runs):
    interp_tpr = np.interp(mean_fpr, fpr_list[i], tpr_list[i])
    interp_tpr[0] = 0.0
    tpr_interp_list.append(interp_tpr)
mean_tpr = np.mean(tpr_interp_list, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)

plt.figure()
for i in range(n_runs):
    # label only the first faint curve, so the legend shows it just once
    label = 'Individual runs' if i == 0 else None
    plt.plot(1-fpr_list[i], tpr_list[i], color='blue', alpha=0.2, label=label)
plt.plot(1-mean_fpr, mean_tpr, color='red', linewidth=2,
         label='Mean AUC = %0.4f' % mean_auc)
plt.plot([1, 0], [0, 1], 'k--', alpha=0.6)   # diagonal reference
plt.xlim([1, 0])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC over {n_runs} runs (individual + mean)')
plt.legend(loc='lower right')
plt.savefig(f'ROC_individual_and_mean_over_{n_runs}_runs.png')
plt.close()