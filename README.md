# Reproducibility chalenge (Claims)

###  1.1 HS improves predictive performance (Fig. 4)

### 1.2 Improvements more significant for smaller datasets

### 1.3 Improvements tend to increase with number of leaves

### 1.4 CCP and HS can be used synergistically

### 1.5 use of HS could allow for otherwise undetected groups

### 1.6 decrease bias^2 + var (Fig. 3)

### 1.7 hsCART outperforms CART (LBS)

### 2.1 HS improves prediction performance of RF (comparison of hs vs mtry vs depth parameters)

* Tune each hyperparametr via CV and average over 10 random splits

### 2.2 hsRF achieves maximum performance with fewer trees than RF (5 times fewer trees)

### 2.4 hsRF much faster to fit than BART (10 - 15 times faster)

### 3.1 removes sampling artifacts & simplifies boundaries (Fig. 5)

### 3.2 Reduces explanation (SHAP) variance (Fig. 6/7)

### 3.3 makes it easier to interpret model interactions

### 3.4 fitted function is closer to being additive 

# Environment

With conda:
```bash
conda create -n mlds python=3.10
conda activate mlds
conda install -c conda-forge numpy scikit-learn scipy pandas nb_conda shap
pip install imodels pmlb
```

With virtualenv:
```bash
```

# Run tests
```bash
python -m unittest discover tests -v
```