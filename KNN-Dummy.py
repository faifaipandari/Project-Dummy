# K-Nearest Neighbours: cancer (C) vs non-cancer (NC) from X_1-X_9
# 70:30 train/test split + 5-fold NESTED cross-validation
# Structure mirrors the Random Forest script. KNN-specific differences are flagged with [KNN].

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, auc, make_scorer
from sklearn.model_selection import (
    train_test_split, GridSearchCV, PredefinedSplit)
# [KNN] swap RandomForestClassifier -> KNeighborsClassifier, and add scaling tools
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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

# 1. Shuffling and splitting patients by patient ID (using all the data points per patient)


def split(data: pd.DataFrame, training_size: float = 0.7, shuffle: bool = True, random_state: int = 42):
    patients = {}  # Starts an empty dictionary
    # Go through every row, one at a time where index is a row number (0, 1, 2 ...)
    for index in range(len(data)):
        pid = data.loc[index, 'pid']   # which patient this row belongs to
        if pid not in patients:        # first time we see this patient
            # give them an empty list to hold their rows
            patients[pid] = []
        # add this row number to that patient's list
        patients[pid].append(index)

    patients = list(patients.values())  # keep just the lists of row numbers
    num_patients = len(patients)
    training_num = int(num_patients * training_size)
    testing_num = int(num_patients - training_num)

    # Splitting the patients
    if shuffle:
        random.seed(random_state)
        random.shuffle(patients)
    training_patients = patients[:training_num]
    testing_patients = patients[training_num:]

    training_indices = []  # collect all the training row numbers
    for patient in training_patients:
        training_indices += patient
    testing_indices = []
    for patient in testing_patients:
        testing_indices += patient

    return training_indices, testing_indices


# 2. Using the features information to predict C/NC
df = pd.read_csv("Dummy.csv")
features = [f"X_{i}" for i in range(1, 10)]


# 3. Split by patient: get the train/test row numbers, then pick out those rows
training_indices, testing_indices = split(df)


# 4. Base model + the hyperparameter grid to search
# [KNN] KNN classifies by distance, so features must be on a comparable scale.
#       wrap StandardScaler + KNeighborsClassifier in a Pipeline so the scaler is
#       re-fit INSIDE each CV fold (no data leakage). RF did not need this.
#       Note: KNN has no random_state / class_weight / criterion.
knn = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_jobs=-1)),
])
# [KNN] pipeline step is named "knn", so grid keys are prefixed with "knn__"
param_grid = {
    "knn__n_neighbors": [3, 5, 7, 9, 11],
    "knn__weights":     ["uniform", "distance"],
    "knn__metric":      ["euclidean", "manhattan"],
}

# 5. Nested cross-validation (run on the TRAINING set)
X_train = getFeatures(df, training_indices)  # dataFrame of X_1-X_9
Y_train = getTarget(df, training_indices)    # series of 'cnc'
# keep the patients with the same ID together
groups = df.loc[training_indices, 'pid']


def inner_k_fold_cv(df, indices, k=5, shuffle=True, random_state=42):
    # find each patient's label, using ONLY the rows in `indices`
    patient_label = {}
    for idx in indices:
        pid = df.loc[idx, 'pid']
        if pid not in patient_label:
            patient_label[pid] = df.loc[idx, 'cnc']   # one label per patient

    # separate patients by class, so each fold can get a share of both
    c_patients = [pid for pid, lab in patient_label.items() if lab == '1']
    nc_patients = [pid for pid, lab in patient_label.items() if lab == '0']

    # shuffle each class separately (reproducible via the seed)
    if shuffle:
        random.seed(random_state)
        random.shuffle(c_patients)
        random.shuffle(nc_patients)

    # deal patients out round-robin per class so every fold sees both C and NC
    patient_fold = {}
    for i, pid in enumerate(c_patients):
        patient_fold[pid] = i % k
    for i, pid in enumerate(nc_patients):
        patient_fold[pid] = i % k

    # one fold number per row, in the same order as `indices`
    folds = [patient_fold[df.loc[index, 'pid']] for index in indices]

    if len(indices) != len(folds):
        print("  WARNING: fold labelling mismatch!")

    return PredefinedSplit(folds)


def calculateMetrics(cm):
    metrics = {}
    if isinstance(cm, np.ndarray):
        tn, fp, fn, tp = cm.ravel().tolist()
    else:
        tn, fp, fn, tp = cm['tn'], cm['fp'], cm['fn'], cm['tp']

    def safe_div(numerator, denominator):
        # if the denominator is 0 the metric is undefined -> nan instead of a crash
        return numerator / denominator if denominator else np.nan

    metrics['acc'] = safe_div(tp + tn, tp + fp + tn + fn)
    metrics['sens'] = safe_div(tp, tp + fn)
    metrics['spec'] = safe_div(tn, tn + fp)
    metrics['ppv'] = safe_div(tp, tp + fp)
    metrics['npv'] = safe_div(tn, tn + fn)
    return metrics


# our own AUC scorer for the inner CV
def aucScoring(model, features, targets):
    probs = model.predict_proba(features)         # probability per patient
    c_index = list(model.classes_).index("1")     # pick the 'C' column
    preds = probs[:, c_index]
    targets = list(targets)
    return roc_auc_score(targets, preds)


# multi-metric scorer: AUC decides the winner, the rest give us the confusion matrix per fold
scorer = {'auc': aucScoring,
          'tn': make_scorer(lambda yt, yp: confusion_matrix(yt, yp, labels=["0", "1"])[0, 0]),
          'fp': make_scorer(lambda yt, yp: confusion_matrix(yt, yp, labels=["0", "1"])[0, 1]),
          'fn': make_scorer(lambda yt, yp: confusion_matrix(yt, yp, labels=["0", "1"])[1, 0]),
          'tp': make_scorer(lambda yt, yp: confusion_matrix(yt, yp, labels=["0", "1"])[1, 1])}

inner_cv = inner_k_fold_cv(df, training_indices)
grid = GridSearchCV(knn, param_grid, cv=inner_cv,
                    scoring=scorer, n_jobs=-1, refit='auc')

# 6. Tune on the full training set, then test on the 30% testing set
grid.fit(getFeatures(df, training_indices), getTarget(df, training_indices))

cv_metrics = []
# number of hyperparameter combos
n_combos = len(grid.cv_results_['mean_test_auc'])

for k in range(5):  # loop over each inner fold (0..4)
    combo_aucs = [grid.cv_results_[f'split{k}_test_auc'][c]
                  for c in range(n_combos)]

    best_combo = int(np.argmax(combo_aucs))   # combo that won ON THIS FOLD
    best_auc = combo_aucs[best_combo]

    cm = {m: grid.cv_results_[f'split{k}_test_{m}'][best_combo]
          for m in ['tn', 'fp', 'fn', 'tp']}
    metrics = calculateMetrics(cm)

    metrics['split'] = k
    metrics['params'] = grid.cv_results_['params'][best_combo]
    metrics['best_combo'] = best_combo
    metrics['roc_auc'] = best_auc
    cv_metrics.append(metrics)

print("="*30)
print()
print("Best combo:", grid.best_params_,
      "| mean AUC:", round(grid.best_score_, 3))
print()

keys = ['acc', 'sens', 'spec', 'ppv', 'npv', 'roc_auc']

print("Inner CV scores (mean ± std over", len(cv_metrics), "evaluations):")
for key in keys:
    values = [m[key] for m in cv_metrics]
    print(f"  {key:8s} {np.nanmean(values):.3f} ± {np.nanstd(values):.3f}")

# average + std across folds
cv_average = {key: np.nanmean([m[key] for m in cv_metrics]) for key in keys}
cv_std = {key: np.nanstd([m[key] for m in cv_metrics]) for key in keys}

# NOTE: uses cv_average / cv_std (NOT summarise, which is defined later and is for the outer loop)
with open('knn_cv_metrics.json', 'w') as f:
    f.write(json.dumps(cv_metrics))
    f.write('\n')
    f.write(json.dumps({'average': cv_average}))
    f.write('\n')
    f.write(json.dumps({'std': cv_std}))


# confusion matrix + AUC for the best combo, one row per split
best = grid.best_index_  # the mean-best combo's index
print()
print("Per-split results for the best combo:")
print(f"{'split':>5} | {'TN':>4} {'FP':>4} {'FN':>4} {'TP':>4} | {'AUC':>6}")
print("-"*36)
for k in range(5):
    tn = grid.cv_results_[f'split{k}_test_tn'][best]
    fp = grid.cv_results_[f'split{k}_test_fp'][best]
    fn = grid.cv_results_[f'split{k}_test_fn'][best]
    tp = grid.cv_results_[f'split{k}_test_tp'][best]
    auc_k = grid.cv_results_[f'split{k}_test_auc'][best]
    print(f"{k:>5} | {tn:>4.0f} {fp:>4.0f} {fn:>4.0f} {tp:>4.0f} | {auc_k:>6.3f}")
print()
print("="*30)
print()
print("Best hyperparameters:", grid.best_params_)
predicted = grid.best_estimator_.predict(getFeatures(df, testing_indices))

# 7. Report performance on the test set
print()

# [KNN] NO feature importances: KNN has no `feature_importances_` (RF's Gini section is removed).
best_knn = grid.best_estimator_

# 8. Confusion matrix on the test set
cm = confusion_matrix(
    getTarget(df, testing_indices), predicted, labels=["0", "1"])
cm_df = pd.DataFrame(cm,
                     index=["True NC", "True C"],
                     columns=["Predicted NC", "Predicted C"])
print("\nConfusion matrix:")
print(cm_df)

# accuracy
y_test = getTarget(df, testing_indices)
score = accuracy_score(y_test, predicted)
print(f"Accuracy: {score:.3f}")
print()

# 9. Dxcover clinical metrics
# With labels=["NC","C"], the matrix is [[TN, FP], [FN, TP]].
tn, fp, fn, tp = cm.ravel()

# SANITY CHECK: compare sklearn's accuracy_score with the manual formula
manual_acc = (tp + tn) / (tp + tn + fp + fn)
print(f"[check] sklearn acc={score:.6f}  manual acc={manual_acc:.6f}")
print()

print("Dxcover clinical metrics:")
print(f"Sensitivity (recall, C):  {tp/(tp+fn):.3f}")
print(f"Specificity (recall, NC): {tn/(tn+fp):.3f}")
print(f"PPV:                      {tp/(tp+fp):.3f}")
print(f"NPV:                      {tn/(tn+fn):.3f}")
print()

# ROC curve
X_test = getFeatures(df, testing_indices)
y_test = getTarget(df, testing_indices)

probs = best_knn.predict_proba(X_test)
c_index = list(best_knn.classes_).index("1")
preds = probs[:, c_index]  # keep only 'C' as prob
fpr, tpr, threshold = roc_curve(y_test, preds, pos_label="1")
y_bin = list(y_test)
roc_auc = roc_auc_score(y_bin, preds)

# plot ROC curve
specificity = 1 - fpr  # convert FPR -> Specificity
plt.title('Receiver Operating Characteristic of the single demonstration run (KNN)')
plt.plot(1-fpr, tpr, 'b', label='AUC = %0.6f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([1, 0], [0, 1], 'r--')  # diagonal, now in spec space
plt.xlim([1, 0])  # 1.0 on left, 0.0 on right
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig(
    'Receiver_Operating_Characteristic_of_the_Single_Demonstration_Run_KNN.png')
print("="*30)
print()

# ===========================================
# Outer loop - repeat the whole experiment n_runs times
# (the 5-fold inner CV lives inside grid.fit each time)
# ===========================================

n_runs = 51

# lists to collect each run's TEST results
acc_list = []
sens_list = []
spec_list = []
ppv_list = []
npv_list = []
auc_list = []
te_cm_list = []

tr_metrics = {'acc': [], 'sens': [],
              'spec': [], 'ppv': [], 'npv': [], 'auc': [], 'cm': []}

# each run's ROC curve points (for the plots)
fpr_list = []
tpr_list = []

# per-fold inner-CV metrics across all runs (n_runs x 5 folds)
per_fold_metrics_list = []

print(f"Running {n_runs} iterations")

for i in range(n_runs):
    random.seed(i)   # reproducible + different each run

    # 1. new train/test split (different each run because the seed changed)
    training_indices, testing_indices = split(df, random_state=i)

    # SANITY CHECK: no patient in both train and test
    train_pids = set(df.loc[training_indices, 'pid'])
    test_pids = set(df.loc[testing_indices, 'pid'])
    if len(train_pids & test_pids) > 0:
        print(f"  WARNING run {i+1}: patient leakage!")

    # 2. inner CV + grid search on this run's training set
    # [KNN] rebuild the pipeline each run (no random_state needed for KNN)
    knn = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_jobs=-1)),
    ])
    inner_cv = inner_k_fold_cv(df, training_indices, random_state=i)
    grid = GridSearchCV(knn, param_grid, cv=inner_cv,
                        scoring=scorer, n_jobs=-1, refit='auc')
    grid.fit(getFeatures(df, training_indices),
             getTarget(df, training_indices))

    # per-fold inner-CV metrics for the current run
    best = grid.best_index_  # the mean-best hyperparameter combo
    for k in range(5):       # 5 inner folds
        cm_k = {m: grid.cv_results_[f'split{k}_test_{m}'][best]
                for m in ['tn', 'fp', 'fn', 'tp']}
        fold_m = calculateMetrics(cm_k)
        fold_m['roc_auc'] = grid.cv_results_[f'split{k}_test_auc'][best]
        fold_m['run'] = i
        fold_m['split'] = k
        per_fold_metrics_list.append(fold_m)

    # 2.2. training metrics
    best_knn = grid.best_estimator_
    predicted = best_knn.predict(getFeatures(df, training_indices))
    y_test = getTarget(df, training_indices)
    cm = confusion_matrix(y_test, predicted, labels=["0", "1"])
    tn, fp, fn, tp = cm.ravel()
    tr_metrics['cm'].append(
        {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)})
    tr_metrics['acc'].append(accuracy_score(y_test, predicted))
    tr_metrics['sens'].append(tp / (tp + fn))
    tr_metrics['spec'].append(tn / (tn + fp))
    tr_metrics['ppv'].append(tp / (tp + fp))
    tr_metrics['npv'].append(tn / (tn + fn))
    probs = best_knn.predict_proba(getFeatures(df, training_indices))
    c_index = list(best_knn.classes_).index("1")
    preds = probs[:, c_index]
    fpr, tpr, threshold = roc_curve(y_test, preds, pos_label="1")
    y_bin = list(y_test)
    tr_metrics['auc'].append(roc_auc_score(y_bin, preds))

    # 3. predict on this run's test set
    best_knn = grid.best_estimator_
    predicted = best_knn.predict(getFeatures(df, testing_indices))
    y_test = getTarget(df, testing_indices)

    # 4. confusion matrix -> TN, FP, FN, TP
    cm = confusion_matrix(y_test, predicted, labels=["0", "1"])
    tn, fp, fn, tp = cm.ravel()
    te_cm_list.append(
        {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)})

    # 5. the performance metrics
    m = calculateMetrics(cm)
    accuracy, sensitivity, specificity = m['acc'], m['sens'], m['spec']
    ppv, npv = m['ppv'], m['npv']

    # 6. ROC curve + AUC
    probs = best_knn.predict_proba(getFeatures(df, testing_indices))
    c_index = list(best_knn.classes_).index("1")
    preds = probs[:, c_index]
    fpr, tpr, threshold = roc_curve(y_test, preds, pos_label="1")
    y_bin = list(y_test)
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

    # 8. print this run's row
    print(f"Run {i+1:2d} | Acc {accuracy:.3f} | Sens {sensitivity:.3f} | "
          f"Spec {specificity:.3f} | PPV {ppv:.3f} | NPV {npv:.3f} | AUC {roc_auc:.3f}")

# Averages over all runs (nan-aware, matching the RF file)
print()
print(f"Averages over {n_runs} runs (mean ± std):")
print(f"Accuracy:    {np.nanmean(acc_list):.3f} ± {np.nanstd(acc_list):.3f}")
print(
    f"Sensitivity: {np.nanmean(sens_list):.3f} ± {np.nanstd(sens_list):.3f}")
print(
    f"Specificity: {np.nanmean(spec_list):.3f} ± {np.nanstd(spec_list):.3f}")
print(f"PPV:         {np.nanmean(ppv_list):.3f} ± {np.nanstd(ppv_list):.3f}")
print(f"NPV:         {np.nanmean(npv_list):.3f} ± {np.nanstd(npv_list):.3f}")
print(f"AUC:         {np.nanmean(auc_list):.3f} ± {np.nanstd(auc_list):.3f}")
print()
print("="*30)

# inner-CV performance per fold, averaged over all runs
print()
print(f"Inner-CV per-fold performance (mean ± std over {n_runs} runs):")
fold_keys = ['acc', 'sens', 'spec', 'ppv', 'npv', 'roc_auc']
print(f"{'split':>5} | " + " | ".join(f"{key:>13}" for key in fold_keys))
print("-" * 95)
for k in range(5):
    rows = [r for r in per_fold_metrics_list if r['split'] == k]
    cells = []
    for key in fold_keys:
        vals = [r[key] for r in rows]
        cells.append(f"{np.nanmean(vals):.3f}±{np.nanstd(vals):.3f}")
    print(f"{k:>5} | " + " | ".join(f"{c:>13}" for c in cells))
print()
# save the raw per-fold rows
with open('knn_per_fold_metrics.json', 'w') as f:
    f.write(json.dumps(per_fold_metrics_list))

te_metrics = {'acc': acc_list, 'sens': sens_list, 'spec': spec_list,
              'ppv': ppv_list, 'npv': npv_list, 'auc': auc_list, 'cm': te_cm_list}


# raw per-run data + averages + stds for the .json files
def summarise(metrics, func):
    out = {}
    for k in ['acc', 'sens', 'spec', 'ppv', 'npv', 'auc']:
        out[k] = func(metrics[k])
    out['cm'] = {ck: func([c[ck] for c in metrics['cm']])
                 for ck in ["TP", "TN", "FP", "FN"]}
    return out


def write_metrics_json(path, metrics):
    with open(path, 'w') as f:
        f.write(json.dumps(metrics))
        f.write('\n')
        f.write(json.dumps({'average': summarise(metrics, np.nanmean)}))
        f.write('\n')
        f.write(json.dumps({'std': summarise(metrics, np.nanstd)}))


write_metrics_json('knn_tr_metrics.json', tr_metrics)
write_metrics_json('knn_te_metrics.json', te_metrics)

# Combined plot: all individual runs (faint) + mean curve (bold)
mean_fpr = np.linspace(0, 1, 100)
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
    label = 'Individual runs' if i == 0 else None
    plt.plot(1-fpr_list[i], tpr_list[i], color='blue', alpha=0.2, label=label)
plt.plot(1-mean_fpr, mean_tpr, color='red', linewidth=2,
         label='Mean AUC = %0.4f' % mean_auc)
plt.plot([1, 0], [0, 1], 'k--', alpha=0.6)
plt.xlim([1, 0])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC over {n_runs} runs (individual + mean) - KNN')
plt.legend(loc='lower right')
plt.savefig(f'ROC_individual_and_mean_over_{n_runs}_runs_KNN.png')
plt.close()
