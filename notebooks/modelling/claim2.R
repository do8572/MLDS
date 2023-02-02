# libraries --------------------------------------------------------------------
library(cmdstanr)
library(ggplot2)
library(bayesplot)
library(posterior)
library(tidyverse)
library(HDInterval)
library(cowplot)

# set modelling as root
setwd("~/Documents/FRI/MLDS/repro/MLDS/notebooks/modelling")

# specify datasets evaluated
read.from.vec <- list("./data/reg-hs.csv",    "./data/reg-lbs.csv",     "./data/reg-ccp.csv",    "./data/reg-hsccp.csv",     "./data/reg-mtry.csv",     "./data/reg-dmax.csv",     "./data/reg-hsrf.csv")

# specify file to save to
save.to.vec <- list("./data/reg-hs-res.csv", "./data/reg-lbs-res.csv", "./data/reg-ccp-res.csv", "./data/reg-hsccp-res.csv", "./data/reg-mtry-res.csv", "./data/reg-dmax-res.csv", "./data/reg-hsrf-res.csv") 

# number of actual categories
n_categories <- list(4,4,4,4, 3,3,3)

# compile categorical(bernoulli) model
model_n <- cmdstan_model("./models/categorical.stan")

# repeat for all datasets
for (i in 1:length(read.from.vec)){
  
  # read dataset (dataset, category); category usually (better, same, worse) or just (better, worse) 
  data <- read.csv(read.from.vec[[i]])
  
  # number of datasets
  n_datasets <- max(data$dataset) + 1
  
  # prepare data for model
  stan_data <- list(n=nrow(data), m = n_datasets, k = 2, c = n_categories[[i]], y = data$imp, gid = data$dataset+1)
  
  # fit model
  fit_n <- model_n$sample(
    data = stan_data,
    parallel_chains = 4
  )
  
  # diagnostics
  mcmc_trace(fit_n$draws())
  fit_n$summary()
  
  # convert samples to data frame
  df <- as_draws_df(fit_n$draws())
  
  # save results
  write.csv(df, file = save.to.vec[[i]], row.names = F)
  
}

