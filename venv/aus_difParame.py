bm = """
functions{
  real getRate(real mode, real st){
      real tr_ = ( mode + sqrt(mode*mode + 4*st*st) )/(2*st*st);
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
     real mu_a1mu_f;real<lower=0> s_a1mu_f;
     real mode_a1sd_f;real<lower=0> s_a1sd_f;
     real mu_b1mu_f;real<lower=0> s_b1mu_f;
     real mode_b1sd_f;real<lower=0> s_b1sd_f;

     real sep_modemode;real<lower=0> sep_modesd;
     real sep_Sdmode;real<lower=0> sep_Sdsd;
     real set_modemode;real<lower=0> set_modesd;
     real set_Sdmode;real<lower=0> set_Sdsd;
    
     real dA1_modemode; real<lower=0> dA1_modesd;
     real dA1_Sdmode;   real<lower=0> dA1_Sdsd;
     real dB1_modemode; real<lower=0> dB1_modesd;
     real dB1_Sdmode;   real<lower=0> dB1_Sdsd;
     
     vector[n] cn;
     vector[n] p;
     matrix[n, h] Y;
}
transformed data{
  real<lower=0> sepMode_sh;
  real<lower=0> sepMode_ra;
  real<lower=0> sepSd_sh;
  real<lower=0> sepSd_ra;

  real<lower=0> setMode_sh;
  real<lower=0> setMode_ra;
  real<lower=0> setSd_sh;
  real<lower=0> setSd_ra;

  real<lower=0> sh_a1sd_f;
  real<lower=0> ra_a1sd_f;
  real<lower=0> sh_b1sd_f;
  real<lower=0> ra_b1sd_f;

  sepMode_sh = getShape(sep_modemode, sep_modesd);
  sepMode_ra = getRate(sep_modemode, sep_modesd);
  sepSd_sh = getShape(sep_Sdmode, sep_Sdsd);
  sepSd_ra = getRate(sep_Sdmode, sep_Sdsd);

  setMode_sh = getShape(set_modemode, set_modesd);
  setMode_ra = getRate(set_modemode, set_modesd);
  setSd_sh = getShape(set_Sdmode, set_Sdsd);
  setSd_ra = getRate(set_Sdmode, set_Sdsd);
  
  dA1Mode_sh = getShape(dA1_modemode, dA1_modesd);
  dA1Mode_ra = getRate(dA1_modemode, dA1_modesd);
  dA1Sd_sh = getShape(dA1_Sdmode, dA1_Sdsd);
  dA1Sd_ra = getRate(dA1_Sdmode, dA1_Sdsd);
  
  dB1Mode_sh = getShape(dB1_modemode, dB1_modesd);
  dB1Mode_ra = getRate(dB1_modemode, dB1_modesd);
  dB1Sd_sh = getShape(dB1_Sdmode, dB1_Sdsd);
  dB1Sd_ra = getRate(dB1_Sdmode, dB1_Sdsd);
  
  sh_a1sd_f = getShape(mode_a1sd_f, s_a1sd_f);
  ra_a1sd_f = getRate(mode_a1sd_f, s_a1sd_f);
  sh_b1sd_f = getShape(mode_b1sd_f, s_b1sd_f);
  ra_b1sd_f = getRate(mode_b1sd_f, s_b1sd_f);
}
parameters{
  real A1mu_f;real<lower=0> A1s_f;
  real B1mu_f;real<lower=0> B1s_f;

  real<lower=0> sep_mode;real<lower=0> sep_sd;
  real<lower=0> set_mode;real<lower=0> set_sd;

  vector<lower=0>[h] dA1;
  vector<lower=0>[h] dB1;

  vector[h] A1f;
  vector[h] B1f;
  matrix[n+1, h] Xf;
  matrix[n+1, h] Xs;
  vector<lower=0>[h] s_ep;
  vector<lower=0>[h] s_et;
}
transformed parameters {
  vector[h] A1s;
  vector[h] B1s;
  vector<lower=0,upper=1>[h] Af;
  vector<lower=0,upper=1>[h] Bf;
  vector<lower=0,upper=1>[h] As;
  vector<lower=0,upper=1>[h] Bs;
  real<lower=0> sep_ra;
  real<lower=0> sep_sh;
  real<lower=0> set_ra;
  real<lower=0> set_sh;

  A1s = A1f + dA1;
  B1s = B1f - dB1;
  Af = inv_logit(A1f);
  Bf = inv_logit(B1f);
  As = inv_logit(A1s);
  Bs = inv_logit(B1s);
  sep_sh = getShape(sep_mode, sep_sd);
  sep_ra = getRate(sep_mode, sep_sd);
  set_sh = getShape(set_mode, set_sd);
  set_ra = getRate(set_mode, set_sd);
  
  dA1_sh = getShape(dA1_mode, dA1_sd);
  dA1_ra = getRate(dA1_mode, dA1_sd);
  dB1_sh = getShape(dB1_mode, dB1_sd);
  dB1_ra = getRate(dB1_mode, dB1_sd);
}
model{
  matrix[n+1, h] X;

  A1mu_f ~ normal(mu_a1mu_f,s_a1mu_f);
  A1s_f ~ gamma(sh_a1sd_f,ra_a1sd_f);
  B1mu_f ~ normal(mu_b1mu_f,s_b1mu_f);
  B1s_f ~ gamma(sh_b1sd_f,ra_b1sd_f);
  sep_mode ~ gamma(sepMode_sh, sepMode_ra);
  sep_sd ~ gamma(sepSd_sh, sepSd_ra);
  set_mode ~ gamma(setMode_sh, setMode_ra);
  set_sd ~ gamma(setSd_sh, setSd_ra);
  
  dA1_mode ~ gamma(dA1Mode_sh, dA1Mode_ra);
  dA1_sd ~ gamma(dA1Sd_sh, dA1Sd_ra);
  dB1_mode ~ gamma(dB1Mode_sh, dB1Mode_ra);
  dB1_sd ~ gamma(dB1Sd_sh, dB1Sd_ra);

  for (j in 1:h){
    A1f[j] ~ normal(A1mu_f,A1s_f);
    B1f[j] ~ normal(B1mu_f,B1s_f);
    s_ep[j] ~ gamma(sep_sh, sep_ra);
    s_et[j] ~ gamma(set_sh, set_ra);

    dA1[j] ~ gamma(dA1_sh,dA1_ra);
    dB1[j] ~ gamma(dB1_sh,dB1_ra);

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
"""

# da1 = {
#     'n': n,
#     'h': h1,
#     'cn': cn,
#     'p': p_,
#     'Y': y_gen_task1,
#
#     'mu_a1mu_f': 1.4, 's_a1mu_f': 0.1,
#     'mode_a1sd_f': 0.3, 's_a1sd_f': 0.01,
#
#     'mu_b1mu_f': -1.5, 's_b1mu_f': 0.2,
#     'mode_b1sd_f': 0.25, 's_b1sd_f': 0.05,
#
#     'mu_a1mu_s': 2.5, 's_a1mu_s': 0.001,
#     'mode_a1sd_s': 0.3, 's_a1sd_s': 0.05,
#
#     'mu_b1mu_s': -3.1, 's_b1mu_s': 0.12,
#     'mode_b1sd_s': 0.5, 's_b1sd_s': 0.09,
#
#     'sep_modemode': 5, 'sep_modesd': 0.35,
#     'sep_Sdmode': 0.25, 'sep_Sdsd': 0.05,
#
#     'set_modemode': 1, 'set_modesd': 0.5,
#     'set_Sdmode': 0.2, 'set_Sdsd': 0.04,
#
#     'dA1_modemode': , 'dA1_modesd': ,
#     'dA1_Sdmode': , 'dA1_Sdsd': ,
#
#     'dB1_modemode': , 'dB1_modesd': ,
#     'dB1_Sdmode': , 'dB1_Sdsd':
# }
