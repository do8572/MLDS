# standard data science toolbox
import numpy as np
import pandas as pd

# imodels used datasets
from imodels.util.data_util import get_clean_dataset

# cross-validation of models
from sklearn.model_selection import StratifiedKFold, KFold

DATASETS = {
    "heart":              ("heart", "heart", 'imodels'),
    "breast-cancer":      ("breast-cancer", "breast_cancer", 'imodels'), # this is the wrong breast-cancer dataset (https://new.openml.org/search?type=data&sort=runs&id=13&status=active)
    "haberman":           ("haberman", "haberman", 'imodels'),
    "ionosphere":         ("ionosphere", "ionosphere", 'pmlb'),
    "diabetes":           ("diabetes", "diabetes", "pmlb"),
    "german-credit":      ("german-credit", "german", "pmlb"),
    "juvenile":           ("juvenile", "juvenile_clean", 'imodels'),
    "recidivism":         ("recidivism", "compas_two_year_clean", 'imodels'),
    'friedman1':          ('friedman1', 'friedman1', 'synthetic'),
    'friedman3':          ('friedman3', 'friedman3', 'synthetic'),
    "diabetes-regr":      ("diabetes-regr", "diabetes", 'sklearn'),
    # missing red-wine and geographical music added later
    'abalone':            ('abalone', '183', 'openml'),
    "satellite-image":    ("satellite-image", "294_satellite_image", 'pmlb'),
    "california-housing": ("california-housing", "california_housing", 'sklearn'),
}

def get_datasets(dataset_list):
    datasets = {}

    # load datasets
    for dataset_name in dataset_list:
        try:
            # add missing datasets (not present in code)
            if dataset_name == "red-wine":
                # red-wine dataset
                wine = pd.read_csv("data/missing_data/winequality-red.csv", sep = ";")
                wine = (wine-wine.mean())/wine.std()
                wine.rename(columns = {'quality':'label'}, inplace = True)

                datasets["red-wine"] = wine
            elif dataset_name == "music":
                # geographical-music dataset (omitted for now since it is a 2D output dataset (RF should and can predict latitude/longitude))
                # 116 & 117 are part of label
                music = pd.read_csv("data/missing_data/geo-music-big.txt", header = None)
                music.rename(columns = {116: 'label1', 117: 'label2'}, inplace = True)
                datasets["music"] = music
            else:
                info = DATASETS[dataset_name]
                X, y, feature_names = get_clean_dataset(info[1], data_source = info[2])
                df = pd.DataFrame(X, columns=feature_names)
                df["label"] = y
                datasets[info[0]] = df
        except:
            print(f"Dataset {dataset_name} not found.")

    return datasets

def leaf_count(dt):
    tree = dt.tree_
    num_leaves = 0

    for i in range(tree.node_count):
        if tree.children_left[i] > 0 or tree.children_right[i] > 0:
            num_leaves += 1

    return num_leaves

# choose alpha such that number of leaves closest to parameter number_of_leaves
def pick_alpha(X, y, number_of_leaves, dt_class):
    path = dt_class().cost_complexity_pruning_path(X, y)
    alphas = np.array(path["ccp_alphas"])
    alphas[alphas <= 0] = 0.005

    leaf_cnts = []

    for alpha in alphas:
        mccp = dt_class(ccp_alpha=alpha).fit(X, y)
        leaf_cnts.append(np.abs(number_of_leaves - leaf_count(mccp)))
    idx = np.argmin(leaf_cnts)

    return alphas[idx]

def pick_alpha_best(X, y, number_of_leaves, dt_class, scorer):
    path = dt_class(max_leaf_nodes=number_of_leaves).cost_complexity_pruning_path(X, y)
    alphas = path["ccp_alphas"]
    
    cv_scores = {}
    
    for alpha in alphas:
        if scorer.__name__ == "auc_roc_scorer":
            hs_skf = StratifiedKFold(n_splits=3, shuffle = True)
        else:
            hs_skf = KFold(n_splits=3, shuffle = True)
        cv_scores[alpha] = []
        for j, (cv_train_index, cv_val_index) in enumerate(hs_skf.split(X, y)):
            X_cv_train, y_cv_train = X[cv_train_index, :], y[cv_train_index]
            X_cv_val, y_cv_val = X[cv_val_index, :], y[cv_val_index]
            mccp = dt_class(ccp_alpha=alpha).fit(X_cv_train, y_cv_train)
            if scorer.__name__ == "auc_roc_scorer":
                print("classification")
                y_val_pred = mccp.predict_proba(X_cv_val)[:, 1]
            else:
                y_val_pred = mccp.predict(X_cv_val)
            cv_scores[alpha].append(scorer(y_cv_val, y_val_pred))
    
    cv_scores = {reg_param: np.mean(cv_scores[reg_param]) for reg_param in cv_scores.keys()}
    best_score = np.max([cv_scores[reg_param] for reg_param in cv_scores.keys()])
    best_param = [reg_param for reg_param in cv_scores.keys() if cv_scores[reg_param] == best_score][0]
    
    return best_param