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
read.from.vec <- list("./data/any.csv", "./data/random.csv", "./data/any-rf-classification.csv", "./data/random-rf.csv")

# specify file to save to
save.to.vec <- list("./data/any-res.csv", "./data/random-res.csv", "./data/any-rf-res.csv", "./data/rf-random-res.csv") 

# compile categorical(bernoulli) model
model_n <- cmdstan_model("./models/categorical.stan")

# repeat for all datasets
for (i in 1:length(read.from.vec)){
  
  # read dataset (dataset, category); category usually (better, same, worse) or just (better, worse) 
  data <- read.csv(read.from.vec[[i]])
  
  # number of datasets
  n_datasets <- max(data$dataset) + 1
  
  # number of categories
  n_categories <- length(unique(data$imp))
  
  # prepare data for model
  stan_data <- list(n=nrow(data), m = n_datasets, k = n_categories, y = data$imp, gid = data$dataset+1)
  
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

