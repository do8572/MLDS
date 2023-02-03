# [Re] Hierarchical Shrinkage: Improving the Accuracy and Interpretability of Tree-Based Methods

The aim of this repository is to reproduce the claims, presented in paper [Hierarchical Shrinkage: Improving the Accuracy and Interpretability of Tree-Based Methods](agarwal22b.pdf).

> Agarwal, A., Tan, Y.S., Ronen, O., Singh, C. &amp; Yu, B.. (2022). Hierarchical Shrinkage: Improving the accuracy and interpretability of tree-based models.. <i>Proceedings of the 39th International Conference on Machine Learning</i>, in <i>Proceedings of Machine Learning Research</i> 162:111-135 Available from https://proceedings.mlr.press/v162/agarwal22b.html.


## Environment

For the Python environment, we decided to go with conda. Below is the list of commands to create an environment and install all libraries.

```bash
conda create -n rehs python=3.10
conda activate rehs
conda install -c conda-forge numpy pandas scikit-learn scikit-optimize scipy nb_conda shap plotnine matplotlib tqdm
pip install imodels pmlb bartpy
```
Since we used two different computers, the environment files from both of them are in the [env](env) folder.

For the R environment we used libraries `cmdstanr` and `bayesplot`:
```R
install.packages("cmdstanr", repos = c("https://mc-stan.org/r-packages/", getOption("repos")))
install.packages("bayesplot")
```

## Folder structure

The top-level folders are as follows:
- [data](data/): includes manually downloaded datasets,
- [env](env/): the .yml files with the Python environment information,
- [notebooks](notebooks/): the notebooks for all of the experiments,
- [tests](tests/): test files.

Each folder has its own README.md file for better clarification.
