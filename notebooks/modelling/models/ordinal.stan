// The input data is a vector 'y' of length 'N'.
data {
  int<lower=0> K;
  int<lower=0> M;
  int<lower=0> N;
  int<lower=1, upper=K> y[N];
  int<lower=1, upper=M> gid[N];
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  real<lower=0> sigma;
  real<lower=0> mu_c;
  real<lower=0, upper=1> theta[M, K-1];
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  // mu_c ~ normal(0.5, 1);
  // sigma ~ normal(1, 1);
  // theta ~ normal(mu_c, sigma);
  
  for (i in 1:N){
    y[i] ~ categorical(theta[gid[i]]);
  }
}

