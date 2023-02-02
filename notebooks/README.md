# Notebooks

## Claims

All of the claims that are addressed in the paper with corresponding notebooks are listed here.

### Claim 1: Hierarchical Shrinkage increases the predictive power of tree-based models

- [analysis/claim1-bayesian](analysis/claim1-bayesian.ipynb)
- [experiments/claim1](experiments/claim1.ipynb)
- [modelling/claim1](modelling/claim1.R)

### Claim 2: Hierarchical Shrinkage is better than other regularization algorithms for tree-based models

- [analysis/claim2-bayesian](analysis/claim2-bayesian.ipynb)
- [experiments/claim2](experiments/claim2.ipynb)
- [experiments/claim2 - hsCART outperforms CART with LBS](experiments/claim2%20-%20hsCART%20outperforms%20CART%20with%20LBS.ipynb)
- [modelling/claim2](modelling/claim2.R)

### Claim 3: Hierarchical Shrinkage is faster than other regularization algorithms for tree-based models

- [analysis/claim3-bayesian](analysis/claim3-bayesian.ipynb)

### Claim 4: Hierarchical Shrinkage leads to more intuitive and robust explanations of random forests

- [experiments/claim4 - HS removes sampling artifacts and simplifies boundaries](experiments/claim4%20-%20HS%20removes%20sampling%20artifacts%20and%20simplifies%20boundaries.ipynb)
- [experiments/claim4 - HS reduces explanation variance](experiments/claim4%20-%20HS%20reduces%20explanation%20variance.ipynb)
- [experiments/claim4 - HS makes it easier to interpret model interactions](experiments/claim4%20-%20HS%20makes%20it%20easier%20to%20interpret%20model%20interactions.ipynb)

## Folder structure

- [analysis](analysis/): folder that includes the code for the Bayesian analysis of the claims,
- [data](data/): folder that includes the data, downloaded by the function `imodels.util.data_util.get_clean_dataset()`. The folder [missing_data](data/missing_data/) was created by us to allow for the datasets the authors did not provide code for,
- [experiments](experiments/): this folder includes the notebooks for specific claims. Some claims (claim 4) have multiple notebooks, one notebook per part of claim,
- [graphs](graphs/): folder with figures from the experiments and analysis notebooks. They are organised into subfolders,
- [modelling](modelling/): folder with code for bayesian modelling in R,
- [results](results/): folder with tables with intermediate results,
- [utils](utils/): folder with some helpful functions.

The notebooks [lbs-transform](lbs-transform.ipynb) and [time](time.ipynb) are for the analysis of the results and creating figures from those results.