# [Re] Hierarchical Shrinkage: Improving the Accuracy and Interpretability of Tree-Based Methods

## Environment

For the Python environment, we decided to go with conda. Below is the list of commands to create an environment and install all libraries.

```bash
conda create -n mlds python=3.10
conda activate mlds
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
