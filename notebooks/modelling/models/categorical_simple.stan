data {
  int<lower=0> n;               // number of observations
  int<lower=0> m;               // number of datasets
  int<lower=0> k;               // number of outcomes (always equals 2)
  int<lower=0> c;               // number of real outcomes
  int<lower=1,upper=k> y[n];    // dependent variables
  int<lower=1, upper=m> gid[n]; // dataset id for each outcome
}

parameters {
  simplex[k] theta;             // probabilities for each category
}

model {
  theta ~ normal(1.0 / c, 0.5); // we assume each method is equally likely to be the best
  
  for (i in 1:n)
    y[i] ~ categorical(theta); // find best parameters for categorical distribution
}
