data {
  int<lower=0> n;               // number of observations
  int<lower=0> m;               // number of datasets
  int<lower=0> k;               // number of outcomes
  int<lower=1,upper=k> y[n];    // dependent variables
  int<lower=1, upper=m> gid[n]; // dataset id for each outcome
}

parameters {
  simplex[k] theta_upper;      // probabilities for each category
  simplex[k] theta[m];         // group conditional probabilities for each category
}

model {
  theta_upper ~ normal(1.0 / k, 0.5); // we assume each method is equally likely to be the best
  
  for (i in 1:m){
    theta[i] ~ normal(theta_upper, 0.5); // upper thetas are sampled from a normal distribution
  }
  
  for (i in 1:n)
    y[i] ~ categorical(theta[gid[i]]); // find best parameters for categorical distribution
}
