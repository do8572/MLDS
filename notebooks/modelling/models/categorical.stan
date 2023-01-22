data {
  int<lower=0> n;               // number of observations
  int<lower=0> m;
  int<lower=0> k;               // number of outcomes
  int<lower=1,upper=k> y[n]; // dependent variables
  int<lower=1, upper=m> gid[n];
}

parameters {
  simplex[k] theta_upper;
  simplex[k] theta[m];
}

model {
  theta_upper ~ normal(0.5, 0.25);
  
  for (i in 1:m){
    theta[i] ~ normal(theta_upper, 0.25);
  }
  
  for (i in 1:n)
    y[i] ~ categorical(theta[gid[i]]);
}
