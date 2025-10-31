import numpy as np
import random
import pandas as pd

# Learner
from sklearn.neighbors import NearestNeighbors as NN

#Function to find the K nearest neighhours
# def get_ngbr(df, knn):
#     #np.random.seed(0)
#     rand_sample_idx = random.randint(0, df.shape[0] - 1)
#     parent_candidate = df.iloc[rand_sample_idx]
#     distance,ngbr = knn.kneighbors(parent_candidate.values.reshape(1,-1),3,return_distance=True)
#     candidate_1 = df.iloc[ngbr[0][1]]
#     candidate_2 = df.iloc[ngbr[0][2]]
#     return distance,parent_candidate,candidate_1,candidate_2

import numpy as np
import random
import scipy.stats as stats
def precompute_mahalanobis(df):
    if hasattr(df, "select_dtypes"):
        data_num = df.select_dtypes(include=[np.number])
        data = data_num.values
        numeric_cols = data_num.columns
    else:
        data = np.array(df, dtype=float)
        numeric_cols = None

    mu = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)

    # regularize covariance to avoid singularity
    cov += np.eye(cov.shape[0]) * 1e-6

    inv_cov = np.linalg.pinv(cov)  # safe inverse
    return mu, inv_cov, numeric_cols


def mahalanobis_distance(x, mu, inv_cov, numeric_cols=None):
    if numeric_cols is not None:
        # keep only numeric part of x
        x = np.array(x[numeric_cols], dtype=float)
    else:
        x = np.array(x, dtype=float)

    diff = x - mu
    md = np.sqrt(diff.T @ inv_cov @ diff)
    return md


def get_ngbr(df, knn, mu=None, inv_cov=None, numeric_cols=None,
             lower_percentile=0.3, upper_percentile=0.7):
    """
    Pick parent candidates whose Mahalanobis distance lies
    between lower_percentile and upper_percentile quantiles.
    """

    # dimensionality
    df_dim = len(numeric_cols) if numeric_cols is not None else df.shape[1]

    # Chi-square quantiles
    lower_thresh = stats.chi2.ppf(lower_percentile, df_dim)
    upper_thresh = stats.chi2.ppf(upper_percentile, df_dim)

    while True:
        rand_sample_idx = random.randint(0, df.shape[0] - 1)
        parent_candidate = df.iloc[rand_sample_idx]

        # mahalanobis distance check
        md = mahalanobis_distance(parent_candidate, mu, inv_cov, numeric_cols)

        if lower_thresh <= md**2 <= upper_thresh:
            break

    # get nearest neighbors
    distance, ngbr = knn.kneighbors(
        parent_candidate.values.reshape(1, -1), 3, return_distance=True
    )
    candidate_1 = df.iloc[ngbr[0][1]]
    candidate_2 = df.iloc[ngbr[0][2]]

    return distance, parent_candidate, candidate_1, candidate_2


def generate_samples(no_of_samples, df, df_name, protected_attribute):
    total_data = df.values.tolist()
    knn = NN(n_neighbors=5, algorithm='auto').fit(df)

    column_name = df.columns.tolist()

    # precompute Mahalanobis once
    mu, inv_cov, numeric_cols = precompute_mahalanobis(df)

    for _ in range(no_of_samples):
        distance, parent_candidate, child_candidate_1, child_candidate_2 = get_ngbr(
            df, knn, mu, inv_cov, numeric_cols
        )

        mutant = []
        for key, value in parent_candidate.items():
            x1 = distance[0][1]
            x2 = distance[0][2]
            x3 = abs(x2 - x1)

            if isinstance(parent_candidate[key], (bool, str)):
                if x1 <= x3:
                    mutant.append(np.random.choice([parent_candidate[key], child_candidate_1[key]]))
                else:
                    mutant.append(np.random.choice([child_candidate_1[key], child_candidate_2[key]]))
            else:
                if x1 <= x3:
                    mutant.append(0.6*parent_candidate[key] + 0.25 * child_candidate_1[key] + 0.15*child_candidate_2[key])
                else:
                    mutant.append((x3*parent_candidate[key])/(x1+x3) + (x1*child_candidate_1[key])/(x1+x3))
        total_data.append(mutant)

    final_df = pd.DataFrame(total_data)
    final_df = final_df.set_axis(column_name, axis=1)

    return final_df

def situation(clf,X_train,y_train,keyword):
    #they have used  as a classifier
    X_flip = X_train.copy()
    X_flip[keyword] = np.where(X_flip[keyword]==1, 0, 1)
    a = np.array(clf.predict(X_train))
    b = np.array(clf.predict(X_flip))
    same = (a==b)
    #print(same) #[True  True  True ... False  True  True  True]
    same = [1 if each else 0 for each in same]  #[1 1 1... 0 1 1 1] if true makes it 1 ,else 0
    X_train['same'] = same #make a new column 'same' and put above list into it.
    X_train['y'] = y_train #make a new column 'y' and put y_train value into it.
    X_rest = X_train[X_train['same']==1] #This creates a new DataFrame (X_rest) that contains only the rows where the 'same' column is 1.
    y_rest = X_rest['y']
    X_rest = X_rest.drop(columns=['same','y'])

    print("Removed Points:",np.round((X_train.shape[0] - X_rest.shape[0]) / X_train.shape[0] * 100, 4),"% || ", X_train.shape[0]-X_rest.shape[0])
    point_removed=np.round((X_train.shape[0] - X_rest.shape[0]) / X_train.shape[0] * 100, 4)

    return X_rest,y_rest,point_removed

import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric

def Worker(df, protected_attribute):
    """
    Processes dataset with FairGenerate pipeline.
    Returns:
        processed_df: Dataset after synthetic data generation and situation testing
        metrics_df: DataFrame comparing Original vs Processed metrics
    """

    test_size_input = 0.20
    shuffle_value = True
    learner = 'LGR'

    # Lists to store metrics
    original_accuracy, original_recall, original_f1score, original_precision = [], [], [], []
    original_aod, original_eod, original_spd, original_di = [], [], [], []

    processed_accuracy, processed_recall, processed_f1score, processed_precision = [], [], [], []
    processed_aod, processed_eod, processed_spd, processed_di = [], [], [], []

    for random_seed in range(1, 2):  # single seed loop
        dataset_orig = df.copy()

        # Normalize dataset
        scaler = MinMaxScaler()
        dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig), columns=dataset_orig.columns)

        # Split dataset
        dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=test_size_input,
                                                                 shuffle=shuffle_value, random_state=random_seed)
        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], \
                           dataset_orig_train['Probability']
        X_test, y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'], \
                         dataset_orig_test['Probability']

        # ------------------ Original Metrics ------------------
        clf_lr = LogisticRegression(random_state=random_seed)
        clf_lr.fit(X_train, y_train)

        dataset_t = BinaryLabelDataset(favorable_label=1.0,
                                       unfavorable_label=0.0,
                                       df=dataset_orig_test,
                                       label_names=['Probability'],
                                       protected_attribute_names=[protected_attribute])

        y_pred = clf_lr.predict(X_test)
        dataset_pred = dataset_t.copy()
        dataset_pred.labels = y_pred

        attr = dataset_t.protected_attribute_names[0]
        idx = dataset_t.protected_attribute_names.index(attr)
        privileged_groups = [{attr: dataset_pred.privileged_protected_attributes[idx][0]}]
        unprivileged_groups = [{attr: dataset_pred.unprivileged_protected_attributes[idx][0]}]

        class_metrics = ClassificationMetric(dataset_t, dataset_pred, unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)

        original_accuracy.append(class_metrics.accuracy())
        original_recall.append(class_metrics.recall())
        original_precision.append(class_metrics.precision())
        original_f1score.append((2 * class_metrics.recall() * class_metrics.precision()) / (
                    class_metrics.precision() + class_metrics.recall()))

        original_aod.append(np.abs(class_metrics.average_odds_difference()))
        original_eod.append(np.abs(class_metrics.equal_opportunity_difference()))
        original_spd.append(np.abs(class_metrics.statistical_parity_difference()))
        original_di.append(np.abs(1 - class_metrics.disparate_impact()))

        # ------------------ FairGenerate: Synthetic Data ------------------
        clf1 = LogisticRegression(random_state=random_seed)
        clf1.fit(X_train, y_train)

        # Situation testing before synthetic data generation
        X_train, y_train, _ = situation(clf1, X_train, y_train, protected_attribute)

        clf2 = LogisticRegression(random_state=random_seed)
        clf2.fit(X_train, y_train)

        dataset_orig_train = X_train.copy()
        dataset_orig_train['Probability'] = y_train

        # Step 2: Data Balancing
        groups = [
            ((0, 0), dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 0)]),
            ((0, 1), dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 1)]),
            ((1, 0), dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 0)]),
            ((1, 1), dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 1)])
        ]

        counts = [len(g[1]) for g in groups]
        maximum = max(counts)

        # Generate synthetic samples for balancing
        new_dfs = []
        for (label_val, prot_val), group_df in groups:
            n_to_generate = maximum - len(group_df)
            if n_to_generate > 0 and len(group_df) > 0:
                group_df[protected_attribute] = group_df[protected_attribute].astype(str)
                group_df = generate_samples(n_to_generate, group_df, "dataset", protected_attribute)
                group_df[protected_attribute] = group_df[protected_attribute].astype(float)
            new_dfs.append(group_df)

        # Concatenate all groups
        df = pd.concat(new_dfs, ignore_index=True)

        # Situation testing after synthetic data generation
        X_train, y_train, _ = situation(clf2, df.drop(columns=['Probability']), df['Probability'], protected_attribute)
        processed_df = X_train.copy()
        processed_df['Probability'] = y_train

        # ------------------ Metrics on Processed Data ------------------
        clf_lr = LogisticRegression(random_state=random_seed)
        clf_lr.fit(X_train, y_train)

        dataset_t = BinaryLabelDataset(favorable_label=1.0,
                                       unfavorable_label=0.0,
                                       df=dataset_orig_test,
                                       label_names=['Probability'],
                                       protected_attribute_names=[protected_attribute])

        y_pred = clf_lr.predict(X_test)
        dataset_pred = dataset_t.copy()
        dataset_pred.labels = y_pred

        class_metrics = ClassificationMetric(dataset_t, dataset_pred, unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)

        processed_accuracy.append(class_metrics.accuracy())
        processed_recall.append(class_metrics.recall())
        processed_precision.append(class_metrics.precision())
        processed_f1score.append((2 * class_metrics.recall() * class_metrics.precision()) / (
                    class_metrics.precision() + class_metrics.recall()))

        processed_aod.append(np.abs(class_metrics.average_odds_difference()))
        processed_eod.append(np.abs(class_metrics.equal_opportunity_difference()))
        processed_spd.append(np.abs(class_metrics.statistical_parity_difference()))
        processed_di.append(np.abs(1 - class_metrics.disparate_impact()))

    # ------------------ Metrics DataFrame ------------------
    metrics = {
        "Metric": ["Recall", "Precision", "Accuracy", "F1 Score", "AOD", "EOD", "SPD", "DI"],
        "Original": [
            round(np.median(original_recall), 4),
            round(np.median(original_precision), 4),
            round(np.median(original_accuracy), 4),
            round(np.median(original_f1score), 4),
            round(np.median(original_aod), 4),
            round(np.median(original_eod), 4),
            round(np.median(original_spd), 4),
            round(np.median(original_di), 4)
        ],
        "Processed": [
            round(np.median(processed_recall), 4),
            round(np.median(processed_precision), 4),
            round(np.median(processed_accuracy), 4),
            round(np.median(processed_f1score), 4),
            round(np.median(processed_aod), 4),
            round(np.median(processed_eod), 4),
            round(np.median(processed_spd), 4),
            round(np.median(processed_di), 4)
        ],
    }

    metrics_df = pd.DataFrame(metrics)
    processed_df_full = pd.concat([processed_df, dataset_orig_test], ignore_index=True)
    return processed_df_full, metrics_df


def Worker_1(df, protected_attribute):
    """
    Processes dataset with FairGenerate pipeline.
    Returns:
        processed_df: Dataset after synthetic data generation and situation testing
        metrics_df: DataFrame comparing Original vs Processed metrics
    """

    test_size_input = 0.20
    shuffle_value = True
    learner = 'LGR'

    # Lists to store metrics
    original_accuracy, original_recall, original_f1score, original_precision = [], [], [], []
    original_aod, original_eod, original_spd, original_di = [], [], [], []

    processed_accuracy, processed_recall, processed_f1score, processed_precision = [], [], [], []
    processed_aod, processed_eod, processed_spd, processed_di = [], [], [], []

    for random_seed in range(1, 2):  # single seed loop
        dataset_orig = df.copy()

        # Normalize dataset
        scaler = MinMaxScaler()
        dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig), columns=dataset_orig.columns)

        # Split dataset
        dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=test_size_input,
                                                                 shuffle=shuffle_value, random_state=random_seed)
        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], \
                           dataset_orig_train['Probability']
        X_test, y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'], \
                         dataset_orig_test['Probability']

        # ------------------ Original Metrics ------------------
        clf_lr = LogisticRegression(random_state=random_seed)
        clf_lr.fit(X_train, y_train)

        dataset_t = BinaryLabelDataset(favorable_label=1.0,
                                       unfavorable_label=0.0,
                                       df=dataset_orig_test,
                                       label_names=['Probability'],
                                       protected_attribute_names=[protected_attribute])

        y_pred = clf_lr.predict(X_test)
        dataset_pred = dataset_t.copy()
        dataset_pred.labels = y_pred

        attr = dataset_t.protected_attribute_names[0]
        idx = dataset_t.protected_attribute_names.index(attr)
        privileged_groups = [{attr: dataset_pred.privileged_protected_attributes[idx][0]}]
        unprivileged_groups = [{attr: dataset_pred.unprivileged_protected_attributes[idx][0]}]

        class_metrics = ClassificationMetric(dataset_t, dataset_pred, unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)

        original_accuracy.append(class_metrics.accuracy())
        original_recall.append(class_metrics.recall())
        original_precision.append(class_metrics.precision())
        original_f1score.append((2 * class_metrics.recall() * class_metrics.precision()) / (
                    class_metrics.precision() + class_metrics.recall()))

        original_aod.append(np.abs(class_metrics.average_odds_difference()))
        original_eod.append(np.abs(class_metrics.equal_opportunity_difference()))
        original_spd.append(np.abs(class_metrics.statistical_parity_difference()))
        original_di.append(np.abs(1 - class_metrics.disparate_impact()))

        # ------------------ FairGenerate: Synthetic Data ------------------
        clf1 = LogisticRegression(random_state=random_seed)
        clf1.fit(X_train, y_train)

        # Situation testing before synthetic data generation
        X_train, y_train, _ = situation(clf1, X_train, y_train, protected_attribute)

        clf2 = LogisticRegression(random_state=random_seed)
        clf2.fit(X_train, y_train)

        dataset_orig_train = X_train.copy()
        dataset_orig_train['Probability'] = y_train

        # Step 2: Data Balancing
        groups = [
            ((0, 0), dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 0)]),
            ((0, 1), dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 1)]),
            ((1, 0), dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 0)]),
            ((1, 1), dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 1)])
        ]

        counts = [len(g[1]) for g in groups]
        maximum = max(counts)
        target_size = 2 * maximum

        # Generate synthetic samples for balancing
        new_dfs = []
        for (label_val, prot_val), group_df in groups:
            n_to_generate = target_size - len(group_df)
            if n_to_generate > 0 and len(group_df) > 0:
                group_df[protected_attribute] = group_df[protected_attribute].astype(str)
                group_df = generate_samples(n_to_generate, group_df, "dataset", protected_attribute)
                group_df[protected_attribute] = group_df[protected_attribute].astype(float)
            new_dfs.append(group_df)

        # Concatenate all groups
        df = pd.concat(new_dfs, ignore_index=True)

        # Situation testing after synthetic data generation
        X_train, y_train, _ = situation(clf2, df.drop(columns=['Probability']), df['Probability'], protected_attribute)
        processed_df = X_train.copy()
        processed_df['Probability'] = y_train

        # ------------------ Metrics on Processed Data ------------------
        clf_lr = LogisticRegression(random_state=random_seed)
        clf_lr.fit(X_train, y_train)

        dataset_t = BinaryLabelDataset(favorable_label=1.0,
                                       unfavorable_label=0.0,
                                       df=dataset_orig_test,
                                       label_names=['Probability'],
                                       protected_attribute_names=[protected_attribute])

        y_pred = clf_lr.predict(X_test)
        dataset_pred = dataset_t.copy()
        dataset_pred.labels = y_pred

        class_metrics = ClassificationMetric(dataset_t, dataset_pred, unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)

        processed_accuracy.append(class_metrics.accuracy())
        processed_recall.append(class_metrics.recall())
        processed_precision.append(class_metrics.precision())
        processed_f1score.append((2 * class_metrics.recall() * class_metrics.precision()) / (
                    class_metrics.precision() + class_metrics.recall()))

        processed_aod.append(np.abs(class_metrics.average_odds_difference()))
        processed_eod.append(np.abs(class_metrics.equal_opportunity_difference()))
        processed_spd.append(np.abs(class_metrics.statistical_parity_difference()))
        processed_di.append(np.abs(1 - class_metrics.disparate_impact()))

    # ------------------ Metrics DataFrame ------------------
    metrics = {
        "Metric": ["Recall", "Precision", "Accuracy", "F1 Score", "AOD", "EOD", "SPD", "DI"],
        "Original": [
            round(np.median(original_recall), 4),
            round(np.median(original_precision), 4),
            round(np.median(original_accuracy), 4),
            round(np.median(original_f1score), 4),
            round(np.median(original_aod), 4),
            round(np.median(original_eod), 4),
            round(np.median(original_spd), 4),
            round(np.median(original_di), 4)
        ],
        "Processed": [
            round(np.median(processed_recall), 4),
            round(np.median(processed_precision), 4),
            round(np.median(processed_accuracy), 4),
            round(np.median(processed_f1score), 4),
            round(np.median(processed_aod), 4),
            round(np.median(processed_eod), 4),
            round(np.median(processed_spd), 4),
            round(np.median(processed_di), 4)
        ],
    }

    metrics_df = pd.DataFrame(metrics)
    processed_df_full = pd.concat([processed_df, dataset_orig_test], ignore_index=True)
    return processed_df_full, metrics_df