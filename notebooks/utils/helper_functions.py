# import all relevant libraries

# import standard data science libraries
import numpy as np
import pandas as pd

# import ggplot for graphing
import plotnine
from plotnine import ggplot, aes, geom_line, geom_point, geom_errorbar, \
                     facet_wrap, position_dodge, theme, geom_violin, xlab, ylab, \
                     coord_flip, geom_bar, element_text, scale_x_discrete, scale_fill_manual, \
                     geom_hline, xlim, ylim, scale_color_discrete, labs

# suppress any warnings
import warnings



def plot_fig_4(res, dataset_order, target = "Target variable", save_to = None):
    """ Reproduce figure 4 (plot score as function of n_leaves or n_trees)
    
    args:
        target: str -> y-axis name
        save_to: str -> save to filename (don't save if None)
        res: pd.DataFrame -> experiment logs
            - task: str                -> classification or regression,
            - dataset: str             -> the evaluated dataset
            - boot_iter: int           -> the bootstrap number (100 bootstrap iterations used)
            - algorithm: str           -> the model (HS-CART or DT)
            - scoring: str             -> the name of the sklearn scoring function
            - max_leaves: int          -> specified max_leaves hyperparameter 
            - n_leaves: int            -> actual num. of leaves (discrepancies)
            - regularization: str      -> originally used only for lambda in hierarchical shrinkage but currently used to store all optimized hyperparameters (None if no hyperparameters optimized)
            - train_score: float       -> score on the training set
            - test_score: float        -> score on the test set
            - train_wall_time: float   -> duration of training phase (wall time)
            - test_wall_time: float    -> duration of test phase (wall time)
            - train_cpu_time: float    -> duration of training phase (cpu time)
            - test_cpu_time: float     -> duration of test phase (cpu time)
            - tunning_wall_time: float -> duration of tunning phase (wall time)
            - tunning_cpu_time: float  -> duration of tunning phase (cpu time)
        dataset_order: list[str] -> ordering of datasets
    returns:
        Nothing.
    """
    
    # copy data.frame so we don't accidentaly make any changes
    res = res.copy()

    # check whether we're plotting DT or RF
    if "n_leaves" in res.columns:

        # drop unneeded columns
        res_neccessary = res.drop(["regularization", "max_leaves", "boot_iter", "train_wall_time", "test_wall_time", "train_cpu_time", "test_cpu_time", "tunning_wall_time", "tunning_cpu_time"], axis = 1)
        
        # set x-axis variable for plot
        x_axis = "n_leaves"

        # calculate min. number of leaves per regularization method
        max_leaves = res_neccessary[["algorithm", "dataset", "n_leaves"]].groupby(["algorithm", "dataset"]).max().reset_index()
        min_leaves = max_leaves[["dataset", "n_leaves"]].groupby(["dataset"]).min().reset_index().values.tolist()
        
        to_keep = np.zeros(len(res_neccessary)).astype(bool)

        # cap graph at min. leaves
        for dataset_entry in min_leaves:

            # unpack values
            dataset, leaf_cap = dataset_entry

            # select columns which to keep
            to_keep = to_keep | ((res_neccessary["n_leaves"].to_numpy() <= leaf_cap) & (res_neccessary["dataset"].to_numpy() == dataset))

        # keep only columns that inline with leaf cap
        res_neccessary = res_neccessary.loc[to_keep]


        
    else:
        # drop unneeded columns
        res_neccessary = res.drop(["regularization", "boot_iter", "train_wall_time", "test_wall_time", "train_cpu_time", "test_cpu_time", "tunning_wall_time", "tunning_cpu_time"], axis = 1)

        # set x-axis variable for plot
        x_axis = "n_trees"
        
    
    # group by experiment
    res_group = res_neccessary.groupby(["task", "dataset", "algorithm", "scoring", x_axis])

    
    # get mean of each group
    res_mu = res_group.mean()
    res_mu.columns = ["train_mu", "test_mu"]
    
    # get standard deviation of each group
    res_std = res_group.std()
    res_std.columns = ["train_std", "test_std"]
    
    # combine mean and std into one data frame
    res = res_mu.join(res_std)
    
    # calculate confidence bounds (errbars)
    res["test_min"] = res["test_mu"] - res["test_std"]
    res["test_max"] = res["test_mu"] + res["test_std"]
    res["train_min"] = res["train_mu"] - res["train_std"]
    res["train_max"] = res["train_mu"] + res["train_std"]
    
    res = res.reset_index()
    
    # convert categorical variables to ordinal for visualization
    res["dataset"] = pd.Categorical(res["dataset"], categories = dataset_order, ordered = True)

    # drop Nans
    res = res.dropna()

    # ggplot results (Fig. 4 style)

    g = (
        # draw lineplot
        ggplot(res, aes(x=x_axis, y="test_mu", color="algorithm"))
        + geom_line()
        
        #draw error bars
        + geom_errorbar(aes(ymin="test_min", ymax="test_max"), width=.2,
                     position=position_dodge(0.05))

        # split plot by dataset
        + facet_wrap("dataset", scales = "free", ncol = 4)
        
        # make sure images don't overlap
        + theme(subplots_adjust={'wspace':0.28, 'hspace':0.50})
        
        # specify size of image
        + theme(figure_size=(16, 9))

        # properly name y-axis
        + ylab(target)

        # properly name x-axis
        + xlab("Number of leaves" if x_axis == "n_leaves" else "Number of trees")

        # specify element text size (for report readability)
        + theme(text=element_text(size=20))

        # position legend at bottom
        + theme(legend_position="bottom")

        # don't display legend title
        + theme(legend_title=element_text(color="white"), legend_margin = 20) 
        )
    
    # display plot
    print(g)

    if save_to is not None:
        g.save(filename = save_to, height = 5, width = 20, units = "in", dpi = 100)
    


def likelihood_of_being_better(A, B, rope = 0.005, min_samples = 5):
    """ Calculate the likelihood that a random sample from set A has higher score than from B.
    
    args:
        A: np.array -> 1D numpy array of results from method 1
        B: np.array -> 1D numpy array of results from method 2
        rope: float -> region of practical equivalence (results are considered the same if they differ by less than rope)
        min_samples -> min. size of both sets for our results to have any significance
    returns:
        p_better -> fraction of samples where A better than B
        p_same   -> fraction of samples where A equals B
        p_worse  -> fraction of samples where A worse than B
    """
    
    # sample size to small (return all zeros)
    if len(A) < 5 or len(B) < min_samples:
        return 0.0, 0.0, 0.0
    
    # take 1000 random samples from each model
    sample_a = np.random.choice(A, 1000)
    sample_b = np.random.choice(B, 1000)
    
    # calculate the fraction of times each model is better
    p_better = np.mean((sample_a < sample_b) & (np.abs(sample_a - sample_b) > rope))
    
    # calculate the fraction of times each model is the same
    p_same = np.mean(np.abs(sample_a - sample_b) < rope)
    
    # calculate the fraction of times each model is worse
    p_worse = np.mean((sample_a > sample_b) & (np.abs(sample_a - sample_b) > rope))
    
    return p_better, p_same, p_worse



def likelihood_of_improvement(A, B, rope = 0.005, min_samples = 5):
    """ Calculate the likelihood that model B (HS) improved model A.
    
    args:
        A: np.array -> 1D numpy array of results from method 1
        B: np.array -> 1D numpy array of results from method 2
        rope: float -> region of practical equivalence (results are considered the same if they differ by less than rope)
        min_samples -> min. size of both sets for our results to have any significance
    returns:
        p_better -> fraction of tuples where A better than B
        p_same   -> fraction of tuples where A equals B
        p_worse  -> fraction of tuples where A worse than B
    """
    
    # sample size to small (return all zeros)
    if len(A) < 5 or len(B) < min_samples:
        return 0.0, 0.0, 0.0
    
    # calculate the fraction of tuples where model B improves model A
    p_improvement = np.mean((A < B) & (np.abs(A - B) > rope))
    
    # calculate the fraction of tuples where model B same as model A
    p_same = np.mean(np.abs(A - B) < rope)
    
    # calculate the fraction of tuples where model B makes model A worse
    p_deterioration = np.mean((A > B) & (np.abs(A - B) > rope))
    
    return p_improvement, p_same, p_deterioration



def plot_distribution(data, save_to = None):
    with warnings.catch_warnings(record=True) as future_warn:  
        g = (
            ggplot(data, aes(x="algorithm", y="test_score"))
            + geom_violin(draw_quantiles = [0.25, 0.5, 0.75], fill = "skyblue", alpha = 0.85, style = "right", position = position_dodge())
            + ylab("Explanatory variate values")
            + xlab("Explanatory variates")
            + coord_flip()
            + facet_wrap("dataset")
        )

        print(g)
    
        if save_to is not None:
            g.save(filename = save_to, height = 5, width = 5, units = "in", dpi = 1000)



def plot_comparison(data, target = "", save_to = None):
    """ Plot comparison between two algorithms.
    
    args:
        target: str -> y-axis name
        save_to: str -> save to filename (don't save if None)
        data: pd.DataFrame -> comparison of samples
             - dataset: str -> dataset name
             - variable: str -> (better, same, worse)
             - value: float -> likelihood for each variable outcome
    """
    
    # specify plot figure size
    plotnine.options.figure_size = (5, 5)
    
    g = (
        # draw stacked barplot (better, same, worse)
        ggplot(data, aes(fill="variable", y="value", x="dataset")) + 
        geom_bar(position="stack", stat="identity") +
        
        # draw hline represeting 50-50 outcome
        geom_hline(yintercept = [0.5], color = "red", linetype="dashed") +
        
        # don't diplay axis
        xlab("") +
        ylab(target) + 
        
        # use preety colors for outcomes
        scale_fill_manual(values=["#E69F00", "#009E73", "#56B4E9"]) + 
        
        # rotate labels & increase size
        theme(axis_text_x=element_text(rotation=30, hjust=1, size= 11), legend_position="none") + 
        
        # remove legend
        theme(legend_position="none")
    )

    print(g)
    
    if save_to is not None:
        g.save(filename = save_to, height = 5, width = 5, units = "in", dpi = 100)



def check_top_scores(data, dataset_order, q, alg = "DT"):
    """ Compare only top Q best algorithm scores.
        
    args:
        data: pd.DataFrame -> experiment logs
            - task: str                -> classification or regression,
            - dataset: str             -> the evaluated dataset
            - boot_iter: int           -> the bootstrap number (100 bootstrap iterations used)
            - algorithm: str           -> the model (HS-CART or DT)
            - scoring: str             -> the name of the sklearn scoring function
            - max_leaves: int          -> specified max_leaves hyperparameter 
            - n_leaves: int            -> actual num. of leaves (discrepancies)
            - regularization: str      -> originally used only for lambda in hierarchical shrinkage but currently used to store all optimized hyperparameters (None if no hyperparameters optimized)
            - train_score: float       -> score on the training set
            - test_score: float        -> score on the test set
            - train_wall_time: float   -> duration of training phase (wall time)
            - test_wall_time: float    -> duration of test phase (wall time)
            - train_cpu_time: float    -> duration of training phase (cpu time)
            - test_cpu_time: float     -> duration of test phase (cpu time)
            - tunning_wall_time: float -> duration of tunning phase (wall time)
            - tunning_cpu_time: float  -> duration of tunning phase (cpu time)
        dataset_order: list[str] -> ordering of datasets
        q: float -> quantile of top values to keep
        alg:str -> whether DT or RF
    returns:
        top_imp_disp: pd.DataFrame -> comparison of samples
             - dataset: str -> dataset name
             - variable: str -> (better, same, worse)
             - value: float -> likelihood for each variable outcome
    """
    
    # initialize top score dict
    top_scores = {algorithm: {dataset: [] for dataset in np.unique(data["dataset"])} for algorithm in np.unique(data["algorithm"])}

    # traverse through all groups of (algorithm, dataset)
    for algorithm in np.unique(data["algorithm"]):
        for dataset in np.unique(data["dataset"]):
            
            # sort values in each group from largest to smallest
            scores = data.loc[(data["algorithm"] == algorithm) & (data["dataset"] == dataset)].sort_values(["test_score"], ascending = False)["test_score"].to_numpy()

            # pick only the top quantile scores from each group
            num_scores = len(scores)
            top_percentile = int(np.round(q * num_scores))
            top_score = scores[:top_percentile]

            # store top-scores
            top_scores[algorithm][dataset] = top_score

    # convert top-scores to dataset
    top_disp = pd.DataFrame(columns = ["algorithm", "dataset", "test_score"])

    for algorithm in np.unique(data["algorithm"]):
        for dataset in np.unique(data["dataset"]):
            for score in top_scores[algorithm][dataset]:
                top_disp = pd.concat([top_disp, pd.DataFrame({"algorithm": [algorithm], "dataset": [dataset], "test_score": score})])

    # calculate likelihood that HS-DT is better than DT
    top_improvement = {dataset: None  for dataset in np.unique(data["dataset"])}
    for dataset in np.unique(data["dataset"]):
        
        # check which experiment we are checking for (DT or RF)
        if alg == "DT":

            # get top scores for each algorithm
            top_subset_dt = top_disp.loc[(top_disp["algorithm"] == "DT") & (top_disp["dataset"] == dataset)]
            top_subset_hs = top_disp.loc[(top_disp["algorithm"] == "HS (CART)") & (top_disp["dataset"] == dataset)]

        else:
            # get top scores for each algorithm
            top_subset_dt = top_disp.loc[(top_disp["algorithm"] == " RF") & (top_disp["dataset"] == dataset)]
            top_subset_hs = top_disp.loc[(top_disp["algorithm"] == "HS-RF") & (top_disp["dataset"] == dataset)]
        
        # compare randomly sampled pairs of top scores which one is better
        top_improvement[dataset] = likelihood_of_being_better(np.array(top_subset_dt["test_score"]), np.array(top_subset_hs["test_score"]))
    
    # convert probability than one algorithm is better than the other into a dataframe
    top_imp_disp = {"dataset": [], "better": [], "worse": [], "same": []}

    for dataset in top_improvement.keys():
        top_imp_disp["dataset"].append(dataset)
        top_imp_disp["better"].append(top_improvement[dataset][0])
        top_imp_disp["same"].append(top_improvement[dataset][1])
        top_imp_disp["worse"].append(top_improvement[dataset][2])
    top_imp_disp = pd.DataFrame(top_imp_disp)
    
    # stack outcome probabilities into one column
    top_imp_disp = pd.melt(top_imp_disp, id_vars=['dataset'], value_vars=['better', "same", 'worse'])
    
    # convert categorical variables to ordinal for visualization
    top_imp_disp["dataset"] = pd.Categorical(top_imp_disp["dataset"], categories = dataset_order, ordered = True)
    top_imp_disp["variable"] = pd.Categorical(top_imp_disp["variable"], categories = ['worse', "same", 'better'], ordered = True)
    
    return top_imp_disp



def check_improvement(data, dataset_order, alg = "DT"):
    """ Compare HS and TBM score if there is any improvement.
        
    args:
        data: pd.DataFrame -> experiment logs
            - task: str                -> classification or regression,
            - dataset: str             -> the evaluated dataset
            - boot_iter: int           -> the bootstrap number (100 bootstrap iterations used)
            - algorithm: str           -> the model (HS-CART or DT)
            - scoring: str             -> the name of the sklearn scoring function
            - max_leaves: int          -> specified max_leaves hyperparameter 
            - n_leaves: int            -> actual num. of leaves (discrepancies)
            - regularization: str      -> originally used only for lambda in hierarchical shrinkage but currently used to store all optimized hyperparameters (None if no hyperparameters optimized)
            - train_score: float       -> score on the training set
            - test_score: float        -> score on the test set
            - train_wall_time: float   -> duration of training phase (wall time)
            - test_wall_time: float    -> duration of test phase (wall time)
            - train_cpu_time: float    -> duration of training phase (cpu time)
            - test_cpu_time: float     -> duration of test phase (cpu time)
            - tunning_wall_time: float -> duration of tunning phase (wall time)
            - tunning_cpu_time: float  -> duration of tunning phase (cpu time)
        dataset_order: list[str] -> ordering of datasets
        alg: str -> whether DT or RF
    returns:
        imp_disp: pd.DataFrame -> comparison of samples
             - dataset: str -> dataset name
             - variable: str -> (better, same, worse)
             - value: float -> likelihood for each variable outcome
    """
    
    # initialize dataset
    improvement = {dataset: None  for dataset in np.unique(data["dataset"])}
    
    # traverse through all groups of (dataset)
    for dataset in np.unique(data["dataset"]):

        # check which experiment we are checking for (DT or RF)
        if alg == "DT":
        
            # get top results for each algorithm 
            subset_dt = data.loc[(data["algorithm"] == "DT") & (data["dataset"] == dataset)]
            subset_hs = data.loc[(data["algorithm"] == "HS (CART)") & (data["dataset"] == dataset)]

        else:

            # get top results for each algorithm 
            subset_dt = data.loc[(data["algorithm"] == " RF") & (data["dataset"] == dataset)]
            subset_hs = data.loc[(data["algorithm"] == "HS-RF") & (data["dataset"] == dataset)]
        
        # compare tuples of results which one is better
        improvement[dataset] = likelihood_of_improvement(np.array(subset_dt["test_score"]), np.array(subset_hs["test_score"]))

    # convert results to dataframe
    imp_disp = {"dataset": [], "better": [], "worse": [], "same": []}

    for dataset in improvement.keys():
        imp_disp["dataset"].append(dataset)
        imp_disp["better"].append(improvement[dataset][0])
        imp_disp["same"].append(improvement[dataset][1])
        imp_disp["worse"].append(improvement[dataset][2])
    imp_disp = pd.DataFrame(imp_disp)
    
    # stack outcome probabilities into one column
    imp_disp = pd.melt(imp_disp, id_vars=['dataset'], value_vars=['better', "same", 'worse'])
    
    # convert categorical variables to ordinal for visualization
    imp_disp["dataset"] = pd.Categorical(imp_disp["dataset"], categories = dataset_order, ordered = True)
    imp_disp["variable"] = pd.Categorical(imp_disp["variable"], categories = ['worse', "same", 'better'], ordered = True)
    
    return imp_disp



def check_heuristic(data, dataset_order, defaults, tolerance):
    """ Compare HS and TBM score if there is any improvement.
        
    args:
        data: pd.DataFrame -> experiment logs
            - task: str                -> classification or regression,
            - dataset: str             -> the evaluated dataset
            - boot_iter: int           -> the bootstrap number (100 bootstrap iterations used)
            - algorithm: str           -> the model (HS-CART or DT)
            - scoring: str             -> the name of the sklearn scoring function
            - max_leaves: int          -> specified max_leaves hyperparameter 
            - n_leaves: int            -> actual num. of leaves (discrepancies)
            - regularization: str      -> originally used only for lambda in hierarchical shrinkage but currently used to store all optimized hyperparameters (None if no hyperparameters optimized)
            - train_score: float       -> score on the training set
            - test_score: float        -> score on the test set
            - train_wall_time: float   -> duration of training phase (wall time)
            - test_wall_time: float    -> duration of test phase (wall time)
            - train_cpu_time: float    -> duration of training phase (cpu time)
            - test_cpu_time: float     -> duration of test phase (cpu time)
            - tunning_wall_time: float -> duration of tunning phase (wall time)
            - tunning_cpu_time: float  -> duration of tunning phase (cpu time)
        dataset_order: list[str] -> ordering of datasets
        defaults: dict{str: float} -> dataset : default number of leaves
        tolerance: int -> telarated deviation from actual number of leaves
    returns:
        heuristic_imp_disp: pd.DataFrame -> comparison of samples
             - dataset: str -> dataset name
             - variable: str -> (better, same, worse)
             - value: float -> likelihood for each variable outcome
    """
    
    # initialize heuristic score dictionary
    heuristic_scores = data.copy()
    heuristic_scores = {algorithm: {dataset: [] for dataset in np.unique(data["dataset"])} for algorithm in np.unique(data["algorithm"])}

    # for each group (algorithm, dataset)
    for dataset in np.unique(data["dataset"]):
        
        # specify number of leaves based on heaursitic
        n_leaves = defaults[dataset]
        
        for algorithm in np.unique(data["algorithm"]):
            
            # pick points near default number of leaves
            heuristic_scores[algorithm][dataset] = np.array(data.loc[(data["algorithm"] == algorithm) & (data["dataset"] == dataset) & (np.abs(data["n_leaves"] - n_leaves) < tolerance)]["test_score"])

    # convert scores to dataframe
    heuristic_disp = pd.DataFrame(columns = ["algorithm", "dataset", "test_score"])

    for algorithm in np.unique(data["algorithm"]):
        for dataset in np.unique(data["dataset"]):
            for score in heuristic_scores[algorithm][dataset]:
                heuristic_disp = pd.concat([heuristic_disp, pd.DataFrame({"algorithm": [algorithm], "dataset": [dataset], "test_score": score})])

    # calculate likelihood of improvement
    heuristic_improvement = {dataset: None  for dataset in np.unique(data["dataset"])}
    
    for dataset in np.unique(data["dataset"]):
        
        # split points based on dataset
        heuristic_subset_dt = heuristic_disp.loc[(heuristic_disp["algorithm"] == "DT") & (heuristic_disp["dataset"] == dataset)]
        heuristic_subset_hs = heuristic_disp.loc[(heuristic_disp["algorithm"] == "HS (CART)") & (heuristic_disp["dataset"] == dataset)]
        
        # compare randomly sampled pairs of tuples which one is better
        heuristic_improvement[dataset] = likelihood_of_improvement(np.array(heuristic_subset_dt["test_score"]), np.array(heuristic_subset_hs["test_score"]))

    # get dataframe of whether HS improves TBM when using default parameters
    heuristic_imp_disp = {"dataset": [], "better": [], "worse": [], "same": []}

    for dataset in heuristic_improvement.keys():
        heuristic_imp_disp["dataset"].append(dataset)
        heuristic_imp_disp["better"].append(heuristic_improvement[dataset][0])
        heuristic_imp_disp["same"].append(heuristic_improvement[dataset][1])
        heuristic_imp_disp["worse"].append(heuristic_improvement[dataset][2])
    heuristic_imp_disp = pd.DataFrame(heuristic_imp_disp)

    # stack outcome probabilities into one column
    heuristic_imp_disp = pd.melt(heuristic_imp_disp, id_vars=['dataset'], value_vars=['better', "same", 'worse'])
    
    # convert categorical variables to ordinal for visualization
    heuristic_imp_disp["dataset"] = pd.Categorical(heuristic_imp_disp["dataset"], categories = dataset_order, ordered = True)
    heuristic_imp_disp["variable"] = pd.Categorical(heuristic_imp_disp["variable"], categories = ['worse', "same", 'better'], ordered = True)
    
    return heuristic_imp_disp



def plot_comparison_x(data, target = "", save_to = None):
    """ Plot comparison between two algorithms for each x (num. leaves or trees).
    
    args:
        target: str -> y-axis name
        save_to: str -> save to filename (don't save if None)
        data: pd.DataFrame -> comparison of samples
             - dataset: str -> dataset name
             - variable: str -> (better, same, worse)
             - value: float -> likelihood for each variable outcome
    """
    
    # specify plot figure size
    plotnine.options.figure_size = (15, 15)
    
    g = (
        # draw stacked barplot (better, same, worse)
        ggplot(data, aes(fill="variable", y="value", x="n_leaves")) + 
        geom_bar(position="stack", stat="identity") +
        
        # draw hline represeting 50-50 outcome
        geom_hline(yintercept = [0.5], color = "red", linetype="dashed") +
        
        # don't diplay axis
        xlab("") +
        ylab(target) + 
        
        # use preety colors for outcomes
        scale_fill_manual(values=["#E69F00", "#009E73", "#56B4E9"]) + 
        
        # rotate labels & increase size
        theme(axis_text_x=element_text(rotation=30, hjust=1, size= 11), legend_position="none") + 
        
        # remove legend
        theme(legend_position="none") + 
        
        # group by dataset
        facet_wrap("dataset")
    )

    print(g)
    
    if save_to is not None:
        g.save(filename = save_to, height = 5, width = 20, units = "in", dpi = 100)

def check_improvement_by_x(data, dataset_order, rope = 0.005):
    """ Compare HS and TBM score if there is any improvement for each leaf or tree.
        
    args:
        data: pd.DataFrame -> experiment logs
            - task: str                -> classification or regression,
            - dataset: str             -> the evaluated dataset
            - boot_iter: int           -> the bootstrap number (100 bootstrap iterations used)
            - algorithm: str           -> the model (HS-CART or DT)
            - scoring: str             -> the name of the sklearn scoring function
            - max_leaves: int          -> specified max_leaves hyperparameter 
            - n_leaves: int            -> actual num. of leaves (discrepancies)
            - regularization: str      -> originally used only for lambda in hierarchical shrinkage but currently used to store all optimized hyperparameters (None if no hyperparameters optimized)
            - train_score: float       -> score on the training set
            - test_score: float        -> score on the test set
            - train_wall_time: float   -> duration of training phase (wall time)
            - test_wall_time: float    -> duration of test phase (wall time)
            - train_cpu_time: float    -> duration of training phase (cpu time)
            - test_cpu_time: float     -> duration of test phase (cpu time)
            - tunning_wall_time: float -> duration of tunning phase (wall time)
            - tunning_cpu_time: float  -> duration of tunning phase (cpu time)
        dataset_order: list[str] -> ordering of datasets
        rope: float -> region of practical equivalence
    returns:
        heuristic_imp_disp: pd.DataFrame -> comparison of samples
             - dataset: str -> dataset name
             - n_leaves or n_trees: int -> number of leaves or trees
             - variable: str -> (better, same, worse)
             - value: float -> likelihood for each variable outcome
    """
    
    # select important columns of data
    leaf_improvement = data.loc[(data.algorithm == "HS (CART)")][["dataset", "test_score", "n_leaves"]]
    
    # add score from DT
    leaf_improvement["dt_score"] = np.array(data.loc[(data.algorithm == "DT")]["test_score"])
    
    # calculate improvement
    leaf_improvement["diff"] = np.abs(leaf_improvement["test_score"] - leaf_improvement["dt_score"])
    
    # calculate each of the three scenarios
    leaf_improvement["better"] = (leaf_improvement["test_score"] > leaf_improvement["dt_score"]) & (leaf_improvement["diff"] > rope)
    leaf_improvement["same"] = (leaf_improvement["diff"] < rope)
    leaf_improvement["worse"] = (leaf_improvement["test_score"] < leaf_improvement["dt_score"]) & (leaf_improvement["diff"] > rope)
    
    # calculate probability for each scenario
    leaf_imp_disp = pd.melt(leaf_improvement.groupby(["dataset", "n_leaves"]).mean().reset_index(), id_vars=["dataset", "n_leaves"], value_vars=["better", "same", "worse"])
    
    # convert categorical variables to ordinal for visualization
    leaf_imp_disp["dataset"] = pd.Categorical(leaf_imp_disp["dataset"], categories = dataset_order, ordered = True)
    leaf_imp_disp["variable"] = pd.Categorical(leaf_imp_disp["variable"], categories = ['worse', "same", 'better'], ordered = True)
    
    return leaf_imp_disp

def comparison_rnd_by_x(data, dataset_order, rope = 0.005):
    """ Compare random HS-TBM and TBM score to check which is better for each leaf or tree.
        
    args:
        data: pd.DataFrame -> experiment logs
            - task: str                -> classification or regression,
            - dataset: str             -> the evaluated dataset
            - boot_iter: int           -> the bootstrap number (100 bootstrap iterations used)
            - algorithm: str           -> the model (HS-CART or DT)
            - scoring: str             -> the name of the sklearn scoring function
            - max_leaves: int          -> specified max_leaves hyperparameter 
            - n_leaves: int            -> actual num. of leaves (discrepancies)
            - regularization: str      -> originally used only for lambda in hierarchical shrinkage but currently used to store all optimized hyperparameters (None if no hyperparameters optimized)
            - train_score: float       -> score on the training set
            - test_score: float        -> score on the test set
            - train_wall_time: float   -> duration of training phase (wall time)
            - test_wall_time: float    -> duration of test phase (wall time)
            - train_cpu_time: float    -> duration of training phase (cpu time)
            - test_cpu_time: float     -> duration of test phase (cpu time)
            - tunning_wall_time: float -> duration of tunning phase (wall time)
            - tunning_cpu_time: float  -> duration of tunning phase (cpu time)
        dataset_order: list[str] -> ordering of datasets
        rope: float -> region of practical equivalence
    returns:
        heuristic_imp_disp: pd.DataFrame -> comparison of samples
             - dataset: str -> dataset name
             - n_leaves or n_trees: int -> number of leaves or trees
             - variable: str -> (better, same, worse)
             - value: float -> likelihood for each variable outcome
    """
    
    # initialize dictionary
    leaf_scores = {algorithm: {dataset: {n_leaves: None for n_leaves in np.unique(data["n_leaves"])} for dataset in np.unique(data["dataset"])} for algorithm in np.unique(data["algorithm"])}
    
    # traverse through all groups of (algorithm, dataset, n_leaves)
    for algorithm in np.unique(data["algorithm"]):
        for dataset in np.unique(data["dataset"]):
            for n_leaves in np.unique(data["n_leaves"]):
                
                # get the scores for each group
                leaf_scores[algorithm][dataset][n_leaves] = np.array(data.loc[(data["algorithm"] == algorithm) & (data["dataset"] == dataset) & (np.abs(data["n_leaves"] - n_leaves) < 0.5)]["test_score"])

    # convert to dataframe
    leaf_disp = pd.DataFrame(columns = ["algorithm", "dataset", "test_score"])
    
    # traverse through all groups of (algorithm, dataset, n_leaves)
    for algorithm in np.unique(data["algorithm"]):
        for dataset in np.unique(data["dataset"]):
            for n_leaves in np.unique(data["n_leaves"]):
                
                # append each score to dataframe
                for score in leaf_scores[algorithm][dataset][n_leaves]:
                    leaf_disp = pd.concat([leaf_disp, pd.DataFrame({"algorithm": [algorithm], "dataset": [dataset], "n_leaves": n_leaves, "test_score": score})])
    
    # calculate likelihood that HS is better than some other method
    leaf_improvement = {dataset: {n_leaves: None for n_leaves in np.unique(data["n_leaves"])}  for dataset in np.unique(data["dataset"])}
    
    # traverse through all groups of (algorithm, dataset)
    for dataset in np.unique(data["dataset"]):
        for n_leaves in np.unique(data["n_leaves"]):
            
            # Compare algorithms
            leaf_subset_dt = leaf_disp.loc[(leaf_disp["algorithm"] == "DT") & (leaf_disp["dataset"] == dataset)]
            leaf_subset_hs = leaf_disp.loc[(leaf_disp["algorithm"] == "HS (CART)") & (leaf_disp["dataset"] == dataset)]
            leaf_improvement[dataset][n_leaves] = likelihood_of_being_better(np.array(leaf_subset_dt["test_score"]), np.array(leaf_subset_hs["test_score"]))

    # convert comparisons to dataframe
    leaf_imp_disp = {"dataset": [], "n_leaves": [], "better": [], "worse": [], "same": []}

    for dataset in leaf_improvement.keys():
        for n_leaves in leaf_improvement[dataset].keys():
            leaf_imp_disp["dataset"].append(dataset)
            leaf_imp_disp["n_leaves"].append(n_leaves)
            leaf_imp_disp["better"].append(leaf_improvement[dataset][n_leaves][0])
            leaf_imp_disp["same"].append(leaf_improvement[dataset][n_leaves][1])
            leaf_imp_disp["worse"].append(leaf_improvement[dataset][n_leaves][2])
    leaf_imp_disp = pd.DataFrame(leaf_imp_disp)
    
    # calculate probability for each scenario
    leaf_imp_disp = pd.melt(leaf_imp_disp.groupby(["dataset", "n_leaves"]).mean().reset_index(), id_vars=["dataset", "n_leaves"], value_vars=["better", "same", "worse"])
    
    # convert categorical variables to ordinal for visualization
    leaf_imp_disp["dataset"] = pd.Categorical(leaf_imp_disp["dataset"], categories = dataset_order, ordered = True)
    leaf_imp_disp["variable"] = pd.Categorical(leaf_imp_disp["variable"], categories = ['worse', "same", 'better'], ordered = True)
    
    return leaf_imp_disp

def dt_overtime(dt_filename, ccp_filename, to_file):
    """ Calculate averate overtime of CCP regularization.
    
    args:
        dt_filename:str -> name of experiment log csv file DT
        ccp_filename:str -> name of experiment log csv file for CCP
        to_file:str -> name of file to save to
    output:
        diff:pd.DataFrame -> mean difference between HS and other functions (REG - HS)
    """
    
    # read dataset files
    dt = pd.read_csv(dt_filename)
    ccp = pd.read_csv(ccp_filename)
    
    # DT doesn't have tunning (those values NaN), set tunning to zero
    dt = dt.fillna(0)
    
    # separate each regularization algorithm into separate dataset
    hs = dt.loc[dt["algorithm"] == "HS (CART)"]
    dt = dt.loc[dt["algorithm"] == "DT"]
    ccp = ccp.loc[ccp["algorithm"] == "CCP"]
    
    # calculate aggregate wall & cpu time for DT
    dt["wall_time"] = dt["train_wall_time"] + dt["test_wall_time"]
    dt["cpu_time"] = dt["train_cpu_time"] + dt["test_cpu_time"]
    
    # calculate aggregate wall & cpu time for HS
    hs["wall_time"] = hs["train_wall_time"] + hs["test_wall_time"] + hs["tunning_wall_time"]
    hs["cpu_time"] = hs["train_cpu_time"] + hs["test_cpu_time"] + hs["tunning_cpu_time"]
    
    # calculate aggregate wall & cpu time for CCP
    ccp["wall_time"] = ccp["train_wall_time"] + ccp["test_wall_time"] + ccp["tunning_wall_time"]
    ccp["cpu_time"] = ccp["train_cpu_time"] + ccp["test_cpu_time"] + ccp["tunning_cpu_time"]
    ccp["wall_ratio"] = (ccp["train_wall_time"] + ccp["test_wall_time"]) / dt["wall_time"]
    ccp["cpu_ratio"] = (ccp["train_cpu_time"] + ccp["test_cpu_time"]) / dt["cpu_time"]
    
    # keep only aggregated values
    dt = dt[["dataset", "wall_time", "cpu_time"]]
    hs = hs[["dataset", "wall_time", "cpu_time"]]
    ccp = ccp[["dataset", "wall_time", "cpu_time"]]
    
    # calculate mean execution time for each dataset
    hs_mu = np.round(hs.groupby(["dataset"]).mean(), 3)
    ccp_mu = np.round(ccp.groupby(["dataset"]).mean(), 3)
    
    # calculate std execution time for each dataset
    hs_std = np.round(hs.groupby(["dataset"]).std(), 3)
    ccp_std = np.round(ccp.groupby(["dataset"]).std(), 3)
    
    # calculate number of samples in each dataset (all sample sizes are equal - for each algorithm)
    # pick random/first sample size n since all are the same anyway
    n_ccp = ccp.groupby(["dataset"]).count().to_numpy()[0,0]
    
    # calculate diff. in means between CCP and HS
    ccp_dmu = np.round(ccp_mu - hs_mu, 3).rename(columns = {"cpu_time": "ccp-cpu", "wall_time": "ccp-wall"})
    
    # calculate std of diff. between CCP and HS
    ccp_dstd = np.sqrt(ccp_std**2 + hs_std**2).rename(columns = {"cpu_time": "ccp-cpu-std", "wall_time": "ccp-wall-std"})

    # calculate standard error of means between HS and other regularization methods
    ccp_dsterr = np.round(ccp_dstd / np.sqrt(n_ccp), 3).fillna("-")
    
    # combine the two datasets
    ccp_diff = pd.concat([hs_mu, ccp_dmu], axis = 1)
    
    # fill in missing values
    ccp_diff = ccp_diff.fillna("-")

    # output
    diff = ccp_diff.copy()
    
    # convert into latex format
    for i in ccp_diff.index.values.tolist():
        ccp_diff.loc[i, "ccp-cpu" ] = f"${ccp_diff.loc[i, 'ccp-cpu' ]} \pm {ccp_dsterr.loc[i, 'ccp-cpu-std' ]}$" 
        ccp_diff.loc[i, "ccp-wall"] = f"${ccp_diff.loc[i, 'ccp-wall' ]} \pm {ccp_dsterr.loc[i, 'ccp-wall-std' ]}$" 
        
    # save table to save file
    ccp_diff.to_latex(to_file)
    
    return diff

def rf_overtime(rf_filename, to_file):
    """ Calculate averate overtime of regularization methods relative to HS.
    
    args:
        rf_filename:str -> name of experiment log csv file RF related experiments
        to_file:str -> name of file to save to
    output:
        diff:pd.DataFrame -> mean difference between HS and other functions (REG - HS)
    """
    
    # read dataset files
    rf = pd.read_csv(rf_filename)
    
    # get rid of unnamed column (should remember to always put index false when saving)
    rf = rf.drop(["Unnamed: 0"], axis = 1)
    
    # RF doesn't have tunning (those values NaN), set tunning to zero
    rf = rf.fillna(0)
    
    # calculate aggregate wall & cpu time for RF
    rf["wall_time"] = rf["train_wall_time"] + rf["test_wall_time"] + rf["tunning_wall_time"]
    rf["cpu_time"] = rf["train_cpu_time"] + rf["test_cpu_time"] + rf["tunning_cpu_time"]
    
    # separate each regularization algorithm into separate dataset
    mtry = rf.loc[rf["algorithm"] == "RF-MTRY"][["dataset", "wall_time", "cpu_time"]]
    dmax = rf.loc[rf["algorithm"] == "RF-DEPTH"][["dataset", "wall_time", "cpu_time"]]
    hsrf = rf.loc[rf["algorithm"] == "HS-RF"][["dataset", "wall_time", "cpu_time"]]
    rf = rf.loc[rf["algorithm"] == "RF"][["dataset", "wall_time", "cpu_time"]]
    
    # calculate mean execution time for each dataset
    mtry_mu = np.round(mtry.groupby(["dataset"]).mean(), 1)
    dmax_mu = np.round(dmax.groupby(["dataset"]).mean(), 1)
    hsrf_mu = np.round(hsrf.groupby(["dataset"]).mean(), 1)
    
    # calculate std execution time for each dataset
    mtry_std = np.round(mtry.groupby(["dataset"]).std(), 1)
    dmax_std = np.round(dmax.groupby(["dataset"]).std(), 1)
    hsrf_std = np.round(hsrf.groupby(["dataset"]).std(), 1)
    
    # calculate std of diff. between HS and other regularization methods
    mtry_dstd = np.sqrt(mtry_std**2 + hsrf_std**2).rename(columns = {"wall_time": "mtry-wall-std", "cpu_time": "mtry-cpu-std"})
    dmax_dstd = np.sqrt(dmax_std**2 + hsrf_std**2).rename(columns = {"wall_time": "dmax-wall-std", "cpu_time": "dmax-cpu-std"})

    # calculate number of samples in each dataset (all sample sizes are equal - for each algorithm)
    # pick random/first sample size n since all are the same anyway
    n_mtry = mtry.groupby(["dataset"]).count().to_numpy()[0,0]
    n_dmax = dmax.groupby(["dataset"]).count().to_numpy()[0,0]

    # calculate standard error of means between HS and other regularization methods
    mtry_dsterr = np.round(mtry_dstd / np.sqrt(n_mtry), 1)
    dmax_dsterr = np.round(dmax_dstd / np.sqrt(n_dmax), 1)
    
    # calculate diff. in means between HS and other regularization methods
    mtry_dmu = np.round(dmax_mu - hsrf_mu, 1).rename(columns = {"wall_time": "mtry-wall", "cpu_time": "mtry-cpu"})
    dmax_dmu = np.round(mtry_mu - hsrf_mu, 1).rename(columns = {"wall_time": "dmax-wall", "cpu_time": "dmax-cpu"})
    
    # combine the two datasets
    rf_diff = pd.concat([hsrf_mu, mtry_dmu, dmax_dmu], axis = 1)
    rf_diff = rf_diff[["wall_time", "mtry-wall", "dmax-wall", "cpu_time", "mtry-cpu", "dmax-cpu"]]

    # output
    diff = rf_diff.copy()
    
    # convert into latex format
    for i in rf_diff.index.values.tolist():
        rf_diff.loc[i, "mtry-cpu" ] = f"${rf_diff.loc[i, 'mtry-cpu' ]} \pm {mtry_dsterr.loc[i, 'mtry-cpu-std'   ]}$" 
        rf_diff.loc[i, "mtry-wall"] = f"${rf_diff.loc[i, 'mtry-wall']} \pm {mtry_dsterr.loc[i, 'mtry-wall-std' ]}$" 
        rf_diff.loc[i, "dmax-cpu "] = f"${rf_diff.loc[i, 'dmax-cpu' ]} \pm {dmax_dsterr.loc[i, 'dmax-cpu-std'   ]}$" 
        rf_diff.loc[i, "dmax-wall"] = f"${rf_diff.loc[i, 'dmax-wall']} \pm {dmax_dsterr.loc[i, 'dmax-wall-std' ]}$" 
    
    # save table to save file
    rf_diff.to_latex(to_file)
    
    return diff

def rel_time(data, alg = "RF"):
    """ Calculate fraction time HS and TBM take up in HS-TBM.
    
    args:
        data: pd.DataFrame -> experiment logs
            - task: str                -> classification or regression,
            - dataset: str             -> the evaluated dataset
            - boot_iter: int           -> the bootstrap number (100 bootstrap iterations used)
            - algorithm: str           -> the model (HS-CART or DT)
            - scoring: str             -> the name of the sklearn scoring function
            - max_leaves: int          -> specified max_leaves hyperparameter 
            - n_leaves: int            -> actual num. of leaves (discrepancies)
            - regularization: str      -> originally used only for lambda in hierarchical shrinkage but currently used to store all optimized hyperparameters (None if no hyperparameters optimized)
            - train_score: float       -> score on the training set
            - test_score: float        -> score on the test set
            - train_wall_time: float   -> duration of training phase (wall time)
            - test_wall_time: float    -> duration of test phase (wall time)
            - train_cpu_time: float    -> duration of training phase (cpu time)
            - test_cpu_time: float     -> duration of test phase (cpu time)
            - tunning_wall_time: float -> duration of tunning phase (wall time)
            - tunning_cpu_time: float  -> duration of tunning phase (cpu time)
    outputs:
        res: pd.DataFrame -> relative time spent by each algorithm
            - dataset:str     -> the evaluated dataset
            - algorithm:str   -> the model (HS or TBM)
            - wall-time:float -> relative mean wall time
            - cpu-time:float  -> relative mean cpu time
            - wall-std:float  -> std of relative wall time
            - cpu-std:float   -> std of relative cpu time
    """

    # copy data
    data = data.copy()
    
    # sum together train and test time
    data["wall-time"] = data["test_wall_time"] + data["train_wall_time"]
    data["cpu-time"]  = data["test_cpu_time"]  + data["train_cpu_time"]
    
    if alg == "RF":
        
        # separate HS-TBM and TBM train time
        hs_tbm  = data.loc[data["algorithm"] == "HS-RF"]
        tbm = data.loc[data["algorithm"] == "RF"]

        # calculate HS time as t(HS) = t(HS-TBM) - t(TBM)
        hs_wall = np.array(hs_tbm["wall-time"]) - np.array(tbm["wall-time"])
        hs_cpu = np.array(hs_tbm["cpu-time"]) - np.array(tbm["cpu-time"])
        
        # calculate what fraction of total execution time is used up by HS
        hs_wrel = hs_wall / np.array(hs_tbm["wall-time"])
        hs_crel = hs_cpu / np.array(hs_tbm["cpu-time"])
        
        # calculate what fraction of total execution time is used up by RF
        tbm_wrel = 1 - hs_wrel
        tbm_crel = 1 - hs_crel

    else:

        # separate HS-TBM and TBM train time
        hs_tbm  = data.loc[data["algorithm"] == "HS (CART)"]
        tbm = data.loc[data["algorithm"] == "DT"]

        # calculate HS time as t(HS) = t(HS-TBM)
        hs_wrel = np.array(hs_tbm["wall-time"]) / (np.array(hs_tbm["wall-time"]) + np.array(tbm["wall-time"]))
        hs_crel = np.array(hs_tbm["cpu-time"]) / (np.array(hs_tbm["cpu-time"]) + np.array(tbm["cpu-time"]))

        # calculate what fraction of total execution time is used up by RF
        tbm_wrel = np.array(tbm["wall-time"]) / (np.array(hs_tbm["wall-time"]) + np.array(tbm["wall-time"]))
        tbm_crel = np.array(tbm["cpu-time"]) / (np.array(hs_tbm["cpu-time"]) + np.array(tbm["cpu-time"]))
    
    # assign relative times back to original dataset
    tbm["wall-time"] = tbm_wrel
    hs_tbm["wall-time"]  = hs_wrel
    
    tbm["cpu-time"] = tbm_crel
    hs_tbm["cpu-time"]  = hs_crel
    
    # join the two datasets
    res = pd.concat([tbm, hs_tbm], axis = 0)
    
    # calculate groupwise mean and std for each algorithm 
    res_mu = res.groupby(["dataset", "algorithm"]).mean()[["wall-time", "cpu-time"]]
    res_std = res.groupby(["dataset", "algorithm"]).std()[["wall-time", "cpu-time"]].rename(columns = {"wall-time": "wall-std", "cpu-time": "cpu-std"})
    
    # join mean and std into one dataset
    res = np.round(pd.concat([res_mu, res_std], axis = 1), 2).reset_index()
    
    return res

def plot_rel_time(data, target, dataset_order, alg, save_to = None):
    """ Plot relative time spent for HS-TBM by HS and TBM.
    
    args:
        - data: pd.DataFrame -> relative time spent by each algorithm
            - dataset:str     -> the evaluated dataset
            - algorithm:str   -> the model (HS or TBM)
            - wall-time:float -> relative mean wall time
            - cpu-time:float  -> relative mean cpu time
            - wall-std:float  -> std of relative wall time
            - cpu-std:float   -> std of relative cpu time
        - target:str -> "cpu-time" or "wall-time"
        - dataset_order:list[str] -> ordered datasets
        - save_to:str -> location to save to 
        - alg:str -> RF or DT
    output:
        Nothing
    """

    if alg == "RF":
        # make sure RF in front of HS-RF for plot
        data.loc[data["algorithm"] == "RF", "algorithm"] = " RF"

    # calculate maximum std
    std_max = data[["wall-std", "cpu-std"]].max().max()
    
    # order datasets by size
    data["dataset"] = pd.Categorical(data["dataset"], categories = dataset_order, ordered = True)

    g = (
            # draw stacked barplot (better, same, worse)
            ggplot(data, aes(fill="algorithm", y=target, x="dataset")) + 
            geom_bar(position="stack", stat="identity") +

            # draw hline represeting 50-50 outcome
            geom_hline(yintercept = [0.5], color = "red", linetype="dashed") +

            # draw hline represeting standard deviation
            geom_hline(yintercept = [0.50 + std_max], color = "blue") +

            # draw hline represeting standard deviation
            geom_hline(yintercept = [0.50 - std_max], color = "blue") +

            # don't diplay axis
            xlab("") +
            ylab("Fraction of execution time") + 

            # use preety colors for outcomes
            scale_fill_manual(values=["#E69F00", "#009E73", "#56B4E9"]) + 

            # rotate labels & increase size
            theme(axis_text_x=element_text(rotation=30, hjust=1, size= 11), legend_position="none") + 

            # remove legend
            theme(legend_position="none") 
        )
    
    # display graph
    print(g)
    
    # save graph
    g.save(filename = save_to, height = 5, width = 5, units = "in", dpi = 100)



def make_modelling_imp_dataset(data, dataset_order, rope = 0.005, save_to = None):
    """ Create modeling dataset representing whether HS improves DT or not.
    
    args:
        save_to: str -> save to filename (don't save if None)
        data: pd.DataFrame -> experiment logs
            - task: str                -> classification or regression,
            - dataset: str             -> the evaluated dataset
            - boot_iter: int           -> the bootstrap number (100 bootstrap iterations used)
            - algorithm: str           -> the model (HS-CART or DT)
            - scoring: str             -> the name of the sklearn scoring function
            - max_leaves: int          -> specified max_leaves hyperparameter 
            - n_leaves: int            -> actual num. of leaves (discrepancies)
            - regularization: str      -> originally used only for lambda in hierarchical shrinkage but currently used to store all optimized hyperparameters (None if no hyperparameters optimized)
            - train_score: float       -> score on the training set
            - test_score: float        -> score on the test set
            - train_wall_time: float   -> duration of training phase (wall time)
            - test_wall_time: float    -> duration of test phase (wall time)
            - train_cpu_time: float    -> duration of training phase (cpu time)
            - test_cpu_time: float     -> duration of test phase (cpu time)
            - tunning_wall_time: float -> duration of tunning phase (wall time)
            - tunning_cpu_time: float  -> duration of tunning phase (cpu time)
        dataset_order: list[str] -> ordering of datasets
    outputs:
        Nothing.
    """
    
    # choose only HS rows, with only the used columns
    leaf_improvement = data.loc[(data.algorithm == "HS (CART)")][["dataset", "test_score", "n_leaves"]]
    
    # append decision tree score for comparison (order for HS and DT the same)
    leaf_improvement["dt_score"] = np.array(data.loc[(data.algorithm == "DT")]["test_score"])
    
    # calculate difference between HS and DT score
    leaf_improvement["diff"] = np.abs(leaf_improvement["test_score"] - leaf_improvement["dt_score"])
    
    # create categorical arrays where HS is better, same or worse than DT
    leaf_improvement["better"] = (leaf_improvement["test_score"] > leaf_improvement["dt_score"]) & (leaf_improvement["diff"] > rope)
    leaf_improvement["same"] = (leaf_improvement["diff"] < rope)
    leaf_improvement["worse"] = (leaf_improvement["test_score"] < leaf_improvement["dt_score"]) & (leaf_improvement["diff"] > rope)
    
    # create categorical vector HS is better - 1, same - 2 or worse -3 than DT
    leaf_improvement["imp"] = 0
    leaf_improvement.loc[leaf_improvement.better, "imp"] = 1
    leaf_improvement.loc[leaf_improvement.same, "imp"] = 2
    leaf_improvement.loc[leaf_improvement.worse, "imp"] = 3
    
    # numerically encode datasets
    leaf_improvement["dataset"] = leaf_improvement["dataset"].replace(dataset_order, list(range(8)))
    
    # save results to save to
    leaf_improvement[["dataset", "imp"]].to_csv(save_to , index = False)

def make_modelling_rnd_dataset(data, dataset_order, rope = 0.005, save_to = None):
    """ Create modeling dataset representing whether HS improves DT or not.
    
    args:
        save_to: str -> save to filename (don't save if None)
        data: pd.DataFrame -> experiment logs
            - task: str                -> classification or regression,
            - dataset: str             -> the evaluated dataset
            - boot_iter: int           -> the bootstrap number (100 bootstrap iterations used)
            - algorithm: str           -> the model (HS-CART or DT)
            - scoring: str             -> the name of the sklearn scoring function
            - max_leaves: int          -> specified max_leaves hyperparameter 
            - n_leaves: int            -> actual num. of leaves (discrepancies)
            - regularization: str      -> originally used only for lambda in hierarchical shrinkage but currently used to store all optimized hyperparameters (None if no hyperparameters optimized)
            - train_score: float       -> score on the training set
            - test_score: float        -> score on the test set
            - train_wall_time: float   -> duration of training phase (wall time)
            - test_wall_time: float    -> duration of test phase (wall time)
            - train_cpu_time: float    -> duration of training phase (cpu time)
            - test_cpu_time: float     -> duration of test phase (cpu time)
            - tunning_wall_time: float -> duration of tunning phase (wall time)
            - tunning_cpu_time: float  -> duration of tunning phase (cpu time)
        dataset_order: list[str] -> ordering of datasets
    outputs:
        Nothing.
    """
    
    # choose only HS rows, with only the used columns
    leaf_improvement = data.loc[(data.algorithm == "HS (CART)")][["dataset", "test_score", "n_leaves"]]
    
    # append decision tree score for comparison (order for HS and DT the same)
    leaf_improvement["dt"] = np.array(data.loc[(data.algorithm == "DT")]["test_score"])
    
    # simply rename column to HS
    leaf_improvement["hs"] = leaf_improvement["test_score"]
    leaf_improvement = leaf_improvement.drop(["test_score"], axis = 1)
    
    # for each group (dataset, n_leaves)
    for dataset in np.unique(data["dataset"]):
        for n_leaves in np.unique(data["n_leaves"]):
            
            # select all results from group for both HS and DT
            group_improvement = leaf_improvement.loc[(leaf_improvement.dataset == dataset) & (leaf_improvement.n_leaves == n_leaves)]
            
            # randomly sample with replacement for each algorithm
            leaf_improvement.loc[(leaf_improvement.dataset == dataset) & (leaf_improvement.n_leaves == n_leaves), "hs"] = np.random.choice(np.array(group_improvement["hs"]), len(group_improvement["hs"]))
            leaf_improvement.loc[(leaf_improvement.dataset == dataset) & (leaf_improvement.n_leaves == n_leaves), "dt"] = np.random.choice(np.array(group_improvement["dt"]), len(group_improvement["dt"]))
    
    # calculate difference between HS and DT score
    leaf_improvement["diff"] = np.abs(leaf_improvement["hs"] - leaf_improvement["dt"])
    
    # create categorical arrays where HS is better, same or worse than DT
    leaf_improvement["better"] = (leaf_improvement["hs"] > leaf_improvement["dt"]) & (leaf_improvement["diff"] > rope)
    leaf_improvement["same"] = (leaf_improvement["diff"] < rope)
    leaf_improvement["worse"] = (leaf_improvement["hs"] < leaf_improvement["dt"]) & (leaf_improvement["diff"] > rope)
    
    # create categorical vector HS is better - 1, same - 2 or worse -3 than DT
    leaf_improvement["imp"] = 0
    leaf_improvement.loc[leaf_improvement.better, "imp"] = 1
    leaf_improvement.loc[leaf_improvement.same, "imp"] = 2
    leaf_improvement.loc[leaf_improvement.worse, "imp"] = 3
    
    # numerically encode datasets
    leaf_improvement["dataset"] = leaf_improvement["dataset"].replace(dataset_order, list(range(8)))
    
    # save results to save to
    leaf_improvement[["dataset", "imp"]].to_csv(save_to, index = False)



def check_best_scores(data, dataset_order, leaf_tol = 2, size_tol = 0.80, rope = 0.05, savevec = [None, None, None, None]):
    """ Build for categorical sample datasets each containing binary values indicating whether the corresponding regularization algorithm was the best.

    args:
        savevec: list[str] -> save to filename (don't save if None) for each of the four regularization methods
        data: pd.DataFrame -> experiment logs
            - task: str                -> classification or regression,
            - dataset: str             -> the evaluated dataset
            - boot_iter: int           -> the bootstrap number (100 bootstrap iterations used)
            - algorithm: str           -> the model (HS-CART or DT)
            - scoring: str             -> the name of the sklearn scoring function
            - max_leaves: int          -> specified max_leaves hyperparameter 
            - n_leaves: int            -> actual num. of leaves (discrepancies)
            - regularization: str      -> originally used only for lambda in hierarchical shrinkage but currently used to store all optimized hyperparameters (None if no hyperparameters optimized)
            - train_score: float       -> score on the training set
            - test_score: float        -> score on the test set
            - train_wall_time: float   -> duration of training phase (wall time)
            - test_wall_time: float    -> duration of test phase (wall time)
            - train_cpu_time: float    -> duration of training phase (cpu time)
            - test_cpu_time: float     -> duration of test phase (cpu time)
            - tunning_wall_time: float -> duration of tunning phase (wall time)
            - tunning_cpu_time: float  -> duration of tunning phase (cpu time)
        dataset_order: list[str] -> ordering of datasets
        leaf_tol:int -> tolerance when comparing methods with different leaf counts
        size_tol:float -> ratio of HS-CCP samples the other methods should
        rope:float -> region of practical importance
    outputs:
        Nothing.
    """

    # data for CCP & LBS comparison graph

    # select CCP test scores only and keep only used rows 
    leaf_improvement = data.loc[(data.algorithm == "HS (CART-CCP)")][["dataset", "test_score", "n_leaves"]]

    # rename test_score -> ccp
    leaf_improvement = leaf_improvement.rename(columns = {"test_score": "hs-ccp"})

    # create columns for other regularization method scores
    leaf_improvement["hs"]  = 0
    leaf_improvement["lbs"]  = 0
    leaf_improvement["ccp"] = np.array(data.loc[(data.algorithm == "CCP"), "test_score"])

    # for every group (dataset, n_leaves)
    for dataset in np.unique(leaf_improvement["dataset"]):
        for n_leaves in np.unique(leaf_improvement["n_leaves"]):

            # calculate exact HS values
            hs_approx = np.array(data.loc[(data.algorithm == "HS (CART)") & (data.dataset == dataset) & (np.abs(data.n_leaves - n_leaves) < leaf_tol), "test_score"])

            # calculate approx. LBS values (values approximated because we only have max. leaves)
            lbs_approx = np.array(data.loc[(data.algorithm == "LBS") & (data.dataset == dataset) & (np.abs(data.n_leaves - n_leaves) < leaf_tol), "test_score"])

            # get size of samples for ccp and hs
            group_improvement = leaf_improvement.loc[(leaf_improvement.dataset == dataset) & (leaf_improvement.n_leaves == n_leaves)]

            # as long as sample size sufficiently large
            if len(hs_approx) > size_tol * len(group_improvement) and len(lbs_approx) > size_tol * len(group_improvement):

                # for each regularization method randomly sample from its scores
                leaf_improvement.loc[(leaf_improvement.dataset == dataset) & (leaf_improvement.n_leaves == n_leaves), "ccp"] = np.random.choice(np.array(group_improvement["ccp"]), len(group_improvement["hs-ccp"]))
                leaf_improvement.loc[(leaf_improvement.dataset == dataset) & (leaf_improvement.n_leaves == n_leaves), "hs"] = np.random.choice(hs_approx, len(group_improvement["hs"]))
                leaf_improvement.loc[(leaf_improvement.dataset == dataset) & (leaf_improvement.n_leaves == n_leaves), "lbs"] = np.random.choice(lbs_approx, len(group_improvement["lbs"]))
                leaf_improvement.loc[(leaf_improvement.dataset == dataset) & (leaf_improvement.n_leaves == n_leaves), "hs-ccp"] = np.random.choice(np.array(group_improvement["hs-ccp"]), len(group_improvement["hs-ccp"]))
            else:

                # if size to small trow away data
                leaf_improvement = leaf_improvement.loc[~((leaf_improvement.dataset == dataset) & (leaf_improvement.n_leaves == n_leaves))]

    # calculate maximum score for each randomly sampled tuple
    max_score = leaf_improvement[["ccp", "lbs", "hs", "hs-ccp"]].max(axis = 1)

    # create binary array signalizing that a certain method of regularization was best
    ccp_best = np.array(leaf_improvement["ccp"] >= max_score - rope)
    hs_best = np.array(leaf_improvement["hs"] >= max_score - rope)
    lbs_best = np.array(leaf_improvement["lbs"] >= max_score - rope)
    hs_ccp_best = np.array(leaf_improvement["hs-ccp"] >= max_score - rope)
    
    # pack binary vectors into a list
    regvec = [hs_best, lbs_best, ccp_best, hs_ccp_best]

    # initialize improvement to 1 - meaning method was not the best
    leaf_improvement["imp"] = 1

    # for each method save the binarry vector signifying how many times the method was the best
    for regmet, sevamet in zip(regvec, savevec):
        # copy dataframe to make sure we're not changing anything
        reg_best = leaf_improvement.copy()
        
        # set tuples where regularization the best to 2
        reg_best.loc[regmet, "imp"] = 2
        
        # encode datasets from 1 - 8 (smallest to largest)
        reg_best["dataset"] = reg_best["dataset"].replace(dataset_order, list(range(8)))
        
        # save dataframe
        if sevamet is not None: 
            reg_best[["dataset", "imp"]].to_csv(sevamet, index = False)
        
    return regvec



def check_best_rt_scores(data, dataset_order, rope = 0.05, savevec = [None, None, None]):
    """ Build for categorical sample datasets each containing binary values indicating whether the corresponding regularization algorithm was the best.

    args:
        savevec: list[str] -> save to filename (don't save if None) for each of the four regularization methods
        data: pd.DataFrame -> experiment logs
            - task: str                -> classification or regression,
            - dataset: str             -> the evaluated dataset
            - boot_iter: int           -> the bootstrap number (100 bootstrap iterations used)
            - algorithm: str           -> the model (HS-CART or DT)
            - scoring: str             -> the name of the sklearn scoring function
            - max_leaves: int          -> specified max_leaves hyperparameter 
            - n_leaves: int            -> actual num. of leaves (discrepancies)
            - regularization: str      -> originally used only for lambda in hierarchical shrinkage but currently used to store all optimized hyperparameters (None if no hyperparameters optimized)
            - train_score: float       -> score on the training set
            - test_score: float        -> score on the test set
            - train_wall_time: float   -> duration of training phase (wall time)
            - test_wall_time: float    -> duration of test phase (wall time)
            - train_cpu_time: float    -> duration of training phase (cpu time)
            - test_cpu_time: float     -> duration of test phase (cpu time)
            - tunning_wall_time: float -> duration of tunning phase (wall time)
            - tunning_cpu_time: float  -> duration of tunning phase (cpu time)
        dataset_order: list[str] -> ordering of datasets
        rope:float -> region of practical importance
    outputs:
        Nothing.
    """
    
    rf_improvement = data[["dataset", "algorithm", "test_score", "n_trees"]]

    # likelihood hs better than other regularization method

    # select only rows for RF
    rf_disp = rf_improvement.loc[(rf_improvement.algorithm == "RF")]
    rf_disp = rf_disp.rename(columns = {"test_score": "rf"})

    # append other regularization scores to dataframe
    rf_disp["mtry"] = np.array(rf_improvement.loc[(rf_improvement.algorithm == "RF-MTRY")]["test_score"])
    rf_disp["depth"] = np.array(rf_improvement.loc[(rf_improvement.algorithm == "RF-DEPTH")]["test_score"])
    rf_disp["hs"] = np.array(rf_improvement.loc[(rf_improvement.algorithm == "HS-RF")]["test_score"])
    
    # define regvec
    regvec = ["mtry", "depth", "rf"]

    for reg, save in zip(regvec, savevec):

        # get difference for rope
        rf_disp["diff"] = np.abs(rf_disp["hs"] - rf_disp[reg])

        # check whether regularization method is better or worse than regularization
        rf_disp["better"] = (rf_disp["hs"] > rf_disp[reg]) & (rf_disp["diff"] > rope)
        rf_disp["same"] = (rf_disp["diff"] < rope)
        rf_disp["worse"] = (rf_disp["hs"] < rf_disp[reg]) & (rf_disp["diff"] > rope)

        # numerically encode values (1 - better, 2 - same, 3 - worse)
        rf_disp["imp"] = 0
        rf_disp.loc[rf_disp.better, "imp"] = 1
        rf_disp.loc[rf_disp.same, "imp"] = 2
        rf_disp.loc[rf_disp.worse, "imp"] = 3

        # numerically encode dataset
        rf_disp["dataset"] = rf_disp["dataset"].replace(dataset_order, list(range(8)))

        # save to file
        if save is not None:
            rf_disp[["dataset", "imp"]].to_csv(save, index = False)



def check_rnd_best(data, dataset_order, rope = 0.05, save_to = None):
    """ Build for categorical sample datasets each containing binary values indicating whether the corresponding regularization algorithm was the best (random samples).

    args:
        save_to: str -> save to filename (don't save if None) for each of the four regularization methods
        data: pd.DataFrame -> experiment logs
            - task: str                -> classification or regression,
            - dataset: str             -> the evaluated dataset
            - boot_iter: int           -> the bootstrap number (100 bootstrap iterations used)
            - algorithm: str           -> the model (HS-CART or DT)
            - scoring: str             -> the name of the sklearn scoring function
            - max_leaves: int          -> specified max_leaves hyperparameter 
            - n_leaves: int            -> actual num. of leaves (discrepancies)
            - regularization: str      -> originally used only for lambda in hierarchical shrinkage but currently used to store all optimized hyperparameters (None if no hyperparameters optimized)
            - train_score: float       -> score on the training set
            - test_score: float        -> score on the test set
            - train_wall_time: float   -> duration of training phase (wall time)
            - test_wall_time: float    -> duration of test phase (wall time)
            - train_cpu_time: float    -> duration of training phase (cpu time)
            - test_cpu_time: float     -> duration of test phase (cpu time)
            - tunning_wall_time: float -> duration of tunning phase (wall time)
            - tunning_cpu_time: float  -> duration of tunning phase (cpu time)
        dataset_order: list[str] -> ordering of datasets
        rope:float -> region of practical importance
    outputs:
        Nothing.
    """
    
    # select only HS-RF rows and relevant columns
    tree_improvement = data.loc[(data.algorithm == "HS-RF")][["dataset", "test_score", "n_trees"]]
    tree_improvement = tree_improvement.rename(columns={"test_score": "hs"})
    
    # append TBM column
    tree_improvement["dt"] = np.array(data.loc[(data.algorithm == "RF")]["test_score"])
    
    # for each group (dataset, n_trees)
    for dataset in np.unique(data["dataset"]):
        for n_trees in np.unique(data["n_trees"]):
            
            # select samples within group
            group_improvement = tree_improvement.loc[(tree_improvement.dataset == dataset) & (tree_improvement.n_trees == n_trees)]
            
            # randomly sample each group
            tree_improvement.loc[(tree_improvement.dataset == dataset) & (tree_improvement.n_trees == n_trees), "hs"] = np.random.choice(np.array(group_improvement["hs"]), len(group_improvement["hs"]))
            tree_improvement.loc[(tree_improvement.dataset == dataset) & (tree_improvement.n_trees == n_trees), "dt"] = np.random.choice(np.array(group_improvement["dt"]), len(group_improvement["dt"]))
    
    # calculate difference between HS and TBM
    tree_improvement["diff"] = np.abs(tree_improvement["hs"] - tree_improvement["dt"])
    
    # create binary vectors indicating whether HS was better or worse than DT
    tree_improvement["better"] = (tree_improvement["hs"] > tree_improvement["dt"]) & (tree_improvement["diff"] > rope)
    tree_improvement["same"] = (tree_improvement["diff"] < rope)
    tree_improvement["worse"] = (tree_improvement["hs"] < tree_improvement["dt"]) & (tree_improvement["diff"] > rope)
    
    # create categorical variable indicating wether HS was better or worse than DT
    tree_improvement["imp"] = 0
    tree_improvement.loc[tree_improvement.better, "imp"] = 1
    tree_improvement.loc[tree_improvement.same, "imp"] = 2
    tree_improvement.loc[tree_improvement.worse, "imp"] = 3
    
    # numerically encode dataset
    tree_improvement["dataset"] = tree_improvement["dataset"].replace(dataset_order, list(range(8)))
    
    # save data
    tree_improvement[["dataset", "imp"]].to_csv(save_to, index = False)

def check_rnd_one_of_the_best(data, dataset_order, rope = 0.05, savevec = [None, None, None]):
    """ Build for categorical sample datasets each containing binary values indicating whether the corresponding regularization algorithm was the best (random samples).

    args:
        save_to: str -> save to filename (don't save if None) for each of the four regularization methods
        data: pd.DataFrame -> experiment logs
            - task: str                -> classification or regression,
            - dataset: str             -> the evaluated dataset
            - boot_iter: int           -> the bootstrap number (100 bootstrap iterations used)
            - algorithm: str           -> the model (HS-CART or DT)
            - scoring: str             -> the name of the sklearn scoring function
            - max_leaves: int          -> specified max_leaves hyperparameter 
            - n_leaves: int            -> actual num. of leaves (discrepancies)
            - regularization: str      -> originally used only for lambda in hierarchical shrinkage but currently used to store all optimized hyperparameters (None if no hyperparameters optimized)
            - train_score: float       -> score on the training set
            - test_score: float        -> score on the test set
            - train_wall_time: float   -> duration of training phase (wall time)
            - test_wall_time: float    -> duration of test phase (wall time)
            - train_cpu_time: float    -> duration of training phase (cpu time)
            - test_cpu_time: float     -> duration of test phase (cpu time)
            - tunning_wall_time: float -> duration of tunning phase (wall time)
            - tunning_cpu_time: float  -> duration of tunning phase (cpu time)
        dataset_order: list[str] -> ordering of datasets
        rope:float -> region of practical importance
    outputs:
        Nothing.
    """
    
    # select only HS-RF rows and relevant columns
    tree_improvement = data.loc[(data.algorithm == "HS-RF")][["dataset", "test_score", "n_trees"]]
    tree_improvement = tree_improvement.rename(columns={"test_score": "hs"})
    
    # append TBM column
    tree_improvement["mtry"] = np.array(data.loc[(data.algorithm == "RF-MTRY")]["test_score"])
    tree_improvement["dmax"] = np.array(data.loc[(data.algorithm == "RF-DEPTH")]["test_score"])
    
    # for each group (dataset, n_trees)
    for dataset in np.unique(data["dataset"]):
        for n_trees in np.unique(data["n_trees"]):
            
            # select samples within group
            group_improvement = tree_improvement.loc[(tree_improvement.dataset == dataset) & (tree_improvement.n_trees == n_trees)]
            
            # randomly sample each group
            tree_improvement.loc[(tree_improvement.dataset == dataset) & (tree_improvement.n_trees == n_trees), "hs"] = np.random.choice(np.array(group_improvement["hs"]), len(group_improvement["hs"]))
            tree_improvement.loc[(tree_improvement.dataset == dataset) & (tree_improvement.n_trees == n_trees), "mtry"] = np.random.choice(np.array(group_improvement["mtry"]), len(group_improvement["mtry"]))
            tree_improvement.loc[(tree_improvement.dataset == dataset) & (tree_improvement.n_trees == n_trees), "dmax"] = np.random.choice(np.array(group_improvement["dmax"]), len(group_improvement["dmax"]))
    
    # calculate the maximum for each row
    max_score = tree_improvement[["hs", "dmax", "mtry"]].max(axis = 1)
    
    # create binary vectors indicating whether method was better than all other methods
    hs_best = np.array(tree_improvement["hs"] >= max_score - rope)
    dmax_best = np.array(tree_improvement["dmax"] >= max_score - rope)
    mtry_best = np.array(tree_improvement["mtry"] >= max_score - rope)

    tree_improvement["imp"] = 1
    
    # encode binary vectors into numerical values & save them
    hs_df_best = tree_improvement.copy()
    hs_df_best.loc[hs_best, "imp"] = 2
    hs_df_best["dataset"] = hs_df_best["dataset"].replace(dataset_order, list(range(8)))
    hs_df_best[["dataset", "imp"]].to_csv(savevec[0], index = False)

    dmax_df_best = tree_improvement.copy()
    dmax_df_best.loc[dmax_best, "imp"] = 2
    dmax_df_best["dataset"] = dmax_df_best["dataset"].replace(dataset_order, list(range(8)))
    dmax_df_best[["dataset", "imp"]].to_csv(savevec[1], index = False)

    mtry_df_best = tree_improvement.copy()
    mtry_df_best.loc[hs_best, "imp"] = 2
    mtry_df_best["dataset"] = mtry_df_best["dataset"].replace(dataset_order, list(range(8)))
    mtry_df_best[["dataset", "imp"]].to_csv(savevec[2], index = False)