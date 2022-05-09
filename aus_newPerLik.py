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
     int h;
     int n;
     real mu_a1mu_f;             real<lower=0> s_a1mu_f;
     real mode_a1sd_f;           real<lower=0> s_a1sd_f;

     real mu_b1mu_f;             real<lower=0> s_b1mu_f;
     real mode_b1sd_f;           real<lower=0> s_b1sd_f;

     real mu_a1mu_s;             real<lower=0> s_a1mu_s;
     real mode_a1sd_s;           real<lower=0> s_a1sd_s;

     real mu_b1mu_s;             real<lower=0> s_b1mu_s;
     real mode_b1sd_s;           real<lower=0> s_b1sd_s;

     real<lower=0> sep_modemode; real<lower=0> sep_modesd;
     real<lower=0> sep_Sdmode;   real<lower=0> sep_Sdsd;

     real<lower=0> set_modemode; real<lower=0> set_modesd;
     real<lower=0> set_Sdmode;   real<lower=0> set_Sdsd;

     vector[n] cn;
     vector[n] p;
     matrix[n, h] Y;
}
transformed data{
  real<lower=0> sepMode_sh;    real<lower=0> sepSd_sh;   
  real<lower=0> sepMode_ra;    real<lower=0> sepSd_ra;      

  real<lower=0> setMode_sh;    real<lower=0> setSd_sh;
  real<lower=0> setMode_ra;    real<lower=0> setSd_ra;

  real<lower=0> sh_a1sd_f;     real<lower=0> sh_a1sd_s;
  real<lower=0> ra_a1sd_f;     real<lower=0> ra_a1sd_s;

  real<lower=0> sh_b1sd_f;     real<lower=0> sh_b1sd_s;
  real<lower=0> ra_b1sd_f;     real<lower=0> ra_b1sd_s;


  sepMode_sh = getShape(sep_modemode, sep_modesd);
  sepMode_ra = getRate(sep_modemode, sep_modesd);
  sepSd_sh = getShape(sep_Sdmode, sep_Sdsd);
  sepSd_ra = getRate(sep_Sdmode, sep_Sdsd);

  setMode_sh = getShape(set_modemode, set_modesd);
  setMode_ra = getRate(set_modemode, set_modesd);
  setSd_sh = getShape(set_Sdmode, set_Sdsd);
  setSd_ra = getRate(set_Sdmode, set_Sdsd);

  sh_a1sd_f = getShape(mode_a1sd_f, s_a1sd_f);
  ra_a1sd_f = getRate(mode_a1sd_f, s_a1sd_f);
  sh_b1sd_f = getShape(mode_b1sd_f, s_b1sd_f);
  ra_b1sd_f = getRate(mode_b1sd_f, s_b1sd_f);

  sh_a1sd_s = getShape(mode_a1sd_s, s_a1sd_s);
  ra_a1sd_s = getRate(mode_a1sd_s, s_a1sd_s);
  sh_b1sd_s = getShape(mode_b1sd_s, s_b1sd_s);
  ra_b1sd_s = getRate(mode_b1sd_s, s_b1sd_s);
}
parameters{
  real A1mu_f;              real<lower=0> A1s_f;
  real B1mu_f;              real<lower=0> B1s_f;
  real A1mu_s;              real<lower=0> A1s_s;
  real B1mu_s;              real<lower=0> B1s_s;
  real<lower=0> sep_mode;   real<lower=0> sep_sd;
  real<lower=0> set_mode;   real<lower=0> set_sd;

  vector[h] A1f;
  vector[h] B1f;
  vector[h] A1s;
  vector[h] B1s;
  matrix[n+1, h] Xf;
  matrix[n+1, h] Xs;

  vector<lower=0>[h] s_ep;
  vector<lower=0>[h] s_et;
}
transformed parameters {
  vector<lower=0,upper=1>[h] Af;
  vector<lower=0,upper=1>[h] Bf;
  vector<lower=0,upper=1>[h] As;
  vector<lower=0,upper=1>[h] Bs;

  real<lower=0> sep_ra;
  real<lower=0> sep_sh;
  real<lower=0> set_ra;
  real<lower=0> set_sh;

  Af = inv_logit(A1f);
  Bf = inv_logit(B1f);
  As = inv_logit(A1s);
  Bs = inv_logit(B1s);

  sep_sh = getShape(sep_mode, sep_sd);
  sep_ra = getRate(sep_mode, sep_sd);
  set_sh = getShape(set_mode, set_sd);
  set_ra = getRate(set_mode, set_sd);

  if(sep_sh<0 || sep_ra <0 || set_sh<0 ||set_ra<0){
    print("sep_mode =", sep_mode);
    print("sep_sd =", sep_sd);
    print("set_mode =", set_mode);
    print("set_sd =", set_sd);       
    print("sep_sh =", sep_sh);
    print("sep_ra =", sep_ra);
    print("set_sh =", set_ra);
    print("set_ra =", set_ra);
  }

}
model{
  matrix[n+1, h] X;

  A1mu_f ~ normal(mu_a1mu_f,s_a1mu_f);
  A1s_f ~ gamma(sh_a1sd_f,ra_a1sd_f);
  B1mu_f ~ normal(mu_b1mu_f,s_b1mu_f);
  B1s_f ~ gamma(sh_b1sd_f,ra_b1sd_f);
  A1mu_s ~ normal(mu_a1mu_s,s_a1mu_s);
  A1s_s ~ gamma(sh_a1sd_s,ra_a1sd_s);
  B1mu_s ~ normal(mu_b1mu_s,s_b1mu_s);
  B1s_s ~ gamma(sh_b1sd_s,ra_b1sd_s);

  sep_mode ~ gamma(sepMode_sh, sepMode_ra);
  sep_sd ~ gamma(sepSd_sh, sepSd_ra);
  set_mode ~ gamma(setMode_sh, setMode_ra);
  set_sd ~ gamma(setSd_sh, setSd_ra);


  for (j in 1:h){
    A1f[j] ~ normal(A1mu_f,A1s_f);
    B1f[j] ~ normal(B1mu_f,B1s_f);
    A1s[j] ~ normal(A1mu_s,A1s_s);
    B1s[j] ~ normal(B1mu_s,B1s_s);
    s_ep[j] ~ gamma(sep_sh, sep_ra);
    s_et[j] ~ gamma(set_sh, set_ra);

    Xf[1,j] ~ normal(0 , s_et[j]);
    Xs[1,j] ~ normal(0 , s_et[j]);
    X[1,j] = Xf[1,j] + Xs[1,j];
    for (i in 1:n){
      Y[i,j] ~ normal( X[i,j]+p[i] , s_ep[j]);                            //Measurement Equation
      Xf[i+1,j] ~ normal( Af[j]*Xf[i,j] - Bf[j]*Y[i,j]*cn[i], s_et[j]);   //Transition Equation
      Xs[i+1,j] ~ normal( As[j]*Xs[i,j] - Bs[j]*Y[i,j]*cn[i], s_et[j]);   //Transition Equation
      X[i+1,j] = Xf[i+1,j] + Xs[i+1,j];                                   //Transition Equation      
    }

  }
}
generated quantities {

    real ll_[h];  //this is the loglikelihood on each of the sample inside the algorith, so we need this quantity and then compute over the mean of the whole samples.
    for (j in 1:h){
        for (i in 1:n){
            ll_[j] += normal_lpdf(Y[i,j] | Xf[i,j] + Xs[i,j] + p[i], s_ep[j]);
        }
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

# DIC computation. then check for the WAIC computation

# logpy = sum(dpois(y, x*colMeans(thetas1),log = T))
# pDIC = 2*(logpy - mean(loglike1$loglike) )
# DIC1 = -2*logpy + 2*pDIC;DIC1

# ypos = NULL;mus1=NULL;des1=NULL
# B = 100000
# for (i in 1:B) {
#   ypos = rpois(6,as.numeric(x*thetas1[i,]))
#   mus1[i] = mean(ypos) #t.mc[i] notese que uno calcula un estadistico de la muestra posterior obtenida y luego esa cosa la usa para obtner el ppp usando mean(cosa>muestral)
#   des1[i] = sd(ypos)
#   if (i%%floor(0.1*B) == 0) cat(i/B*100, "% completado ...", "\n", sep = "")
# }

# write.csv(mus1,"mus1.csv")
# write.csv(des1,"des1.csv")
# read.csv("mus1.csv")$x -> mus1
# read.csv("des1.csv")$x -> des1

# mean(mus1>mean(y))
# # [1] 0.50296
# #La probabilidad e que la dpp genere una muestra posterior con
# #mayor a la media observada
# mean(des1>sd(y))
# # [1] 0.48387
# #La probabilidad e que la dpp genere una muestra con desviacion mayor a
# #la desviacion observada

# # mean(coev>coev(y)) por ejemplo esto tambien lo podriamos calcular..
# # mean(estaditicoCalcu>estadiObs)
