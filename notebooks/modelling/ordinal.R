# libraries --------------------------------------------------------------------
library(cmdstanr)
library(ggplot2)
library(bayesplot)
library(posterior)
library(tidyverse)
library(HDInterval)
library(cowplot)

setwd("~/Documents/FRI/MLDS/repro/MLDS/notebooks/modelling")

# data prep and model compilation ----------------------------------------------
# load data
data <- read.csv("./data/ccp-classification.csv")

# normal model -----------------------------------------------------------------
model_n <- cmdstan_model("./models/categorical.stan")

# data prep
stan_data <- list(n=nrow(data), m=8, k = 3, y = data$imp, gid = data$dataset+1)

# fit
fit_n <- model_n$sample(
  data = stan_data,
  parallel_chains = 4
)

# diagnostics
mcmc_trace(fit_n$draws())
fit_n$summary()

# convert samples to data frame
df <- as_draws_df(fit_n$draws())

# draw results
write.csv(df, file = "./data/reg-rf.csv", row.names = F)

