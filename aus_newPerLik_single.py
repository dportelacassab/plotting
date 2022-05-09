bm = """  #Bayesian  Model - Actual MultiSubject Error Model   Two States
functions{
  real getRate(real mode, real st){
      real tr_ = (mode + sqrt(mode*mode + 4*st*st) )/(2*st*st);
      return tr_;
  }
  real getShape(real mode, real st){
      real ts_ = 1 + mode*getRate(mode,st);
      return ts_;
  }
}
data{
     int n;
     real A1mu_f;              real<lower=0> A1s_f;
     real B1mu_f;              real<lower=0> B1s_f;
     real A1mu_s;              real<lower=0> A1s_s;
     real B1mu_s;              real<lower=0> B1s_s;
     real<lower=0> sep_mode;   real<lower=0> sep_sd;
     real<lower=0> set_mode;   real<lower=0> set_sd;

     vector[n] cn;
     vector[n] p;
     vector[n] Y;
}
transformed data{
  real<lower=0> sep_sh;    real<lower=0> sep_ra;
  real<lower=0> set_sh;    real<lower=0> set_ra;

  sep_sh = getShape(sep_mode, sep_sd);
  sep_ra = getRate(sep_mode, sep_sd);
  set_sh = getShape(set_mode, set_sd);
  set_ra = getRate(set_mode, set_sd);
}
parameters{

  real A1f;
  real B1f;
  real A1s;
  real B1s;
  vector[n+1] Xf;
  vector[n+1] Xs;

  real<lower=0> s_ep;
  real<lower=0> s_et;
}
transformed parameters {
  real<lower=0,upper=1> Af;
  real<lower=0,upper=1> Bf;
  real<lower=0,upper=1> As;
  real<lower=0,upper=1> Bs;

  Af = inv_logit(A1f);
  Bf = inv_logit(B1f);
  As = inv_logit(A1s);
  Bs = inv_logit(B1s);

}
model{
    vector[n+1] X;

    A1f ~ normal(A1mu_f,A1s_f);
    B1f ~ normal(B1mu_f,B1s_f);
    A1s ~ normal(A1mu_s,A1s_s);
    B1s ~ normal(B1mu_s,B1s_s);
    s_ep ~ gamma(sep_sh, sep_ra);
    s_et ~ gamma(set_sh, set_ra);

    Xf[1] ~ normal( 0 , s_et );
    Xs[1] ~ normal( 0 , s_et );
    X[1] = Xf[1] + Xs[1];
    for (i in 1:n){
      Y[i] ~ normal( X[i]+p[i] , s_ep);                             //Measurement Equation
      Xf[i+1] ~ normal( Af*Xf[i] - Bf*Y[i]*cn[i], s_et);            //Transition Equation
      Xs[i+1] ~ normal( As*Xs[i] - Bs*Y[i]*cn[i], s_et);            //Transition Equation
      X[i+1] = Xf[i+1] + Xs[i+1];                                   //Transition Equation      
    }

}

"""

import numpy as np
import math
from scipy.stats import norm
from scipy.stats import mode, gaussian_kde
from scipy.optimize import minimize, shgo


def generate_1D_data(size, mu=0.5, sigma=1, round=3):
    data = np.random.normal(mu, sigma, size)
    return np.around(data, round)


def kde(array, cut_down=True, bw_method='scott'):
    if cut_down:
        bins, counts = np.unique(array, return_counts=True)
        f_mean = counts.mean()
        f_above_mean = bins[counts > f_mean]
        bounds = [f_above_mean.min(), f_above_mean.max()]
        array = array[np.bitwise_and(bounds[0] < array, array < bounds[1])]
    return gaussian_kde(array, bw_method=bw_method)


def mode_estimation(array, cut_down=True, bw_method='scott'):
    kernel = kde(array, cut_down=cut_down, bw_method=bw_method)
    bounds = np.array([[array.min(), array.max()]])
    results = shgo(lambda x: -kernel(x)[0], bounds=bounds, n=100 * len(array))
    return results.x[0]
