# import all relevant libraries

# import standard data science libraries
import numpy as np
import pandas as pd

# import ggplot for graphing
import plotnine
from plotnine import ggplot, aes, geom_line, geom_point, geom_errorbar, \
                     facet_wrap, position_dodge, theme, geom_violin, xlab, ylab, \
                     coord_flip, geom_bar, element_text, scale_x_discrete, scale_fill_manual, \
                     geom_hline, xlim, ylim, scale_color_discrete

# suppress any warnings
import warnings

def plot_fig_4(res):
    """ Reproduce figure 4 (plot score as function of n_leaves)
    
    args:
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
    returns:
        Nothing.
    """
    
    # copy data.frame so we don't accidentaly make any changes
    res = res.copy()
    
    # drop unneeded columns
    res_neccessary = res.drop(["regularization", "max_leaves", "boot_iter", "train_wall_time", "test_wall_time", "train_cpu_time", "test_cpu_time", "tunning_wall_time", "tunning_cpu_time"], axis = 1)
    
    # group by experiment
    res_group = res_neccessary.groupby(["task", "dataset", "algorithm", "scoring", "n_leaves"])
    
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
    
    # ggplot results (Fig. 4 style)
    g = (
        # draw lineplot
        ggplot(res, aes(x="n_leaves", y="test_mu", color="algorithm"))
        + geom_line()
        
        #draw error bars
        + geom_errorbar(aes(ymin="test_min", ymax="test_max"), width=.2,
                     position=position_dodge(0.05))
        
        # split plot by dataset
        + facet_wrap("dataset", scales = "free", ncol = 4)
        
        # make sure images don't overlap
        + theme(subplots_adjust={'wspace':0.5, 'hspace':0.3})
        
        # specify size of image
        + theme(figure_size=(16, 8))
        )
    
    # display plot
    display(g)
    
# calculate the likelihoods that one method will return a better result than the other method
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