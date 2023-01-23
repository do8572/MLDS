data {
  int<lower=0> n;               // number of observations
  int<lower=0> m;
  int<lower=0> k;               // number of outcomes
  int<lower=1,upper=k> y[n]; // dependent variables
  int<lower=1, upper=m> gid[n];
}

parameters {
  simplex[k] theta;
}

model {
  theta ~ normal(0, 0.3);
  
  for (i in 1:n)
    y[i] ~ categorical(theta);
}
