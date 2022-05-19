import numpy as np
import matplotlib.pyplot as plt

###############################################################################################
#        Prior Specifications     #############################################################
###############################################################################################

def getRate(mode, st):
    tr_ = (mode + np.sqrt(mode * mode + 4 * st * st)) / (2 * st * st);
    return tr_;
def getShape(mode, st):
    ts_ = 1 + mode * getRate(mode, st);
    return ts_;

####                   Hyper Priors                     #######################################
###############################################################################################


###############################################################################################
####            dA1            ################################################################
###############################################################################################
dA1_modemode = 5;      dA1_modesd = 0.35
dA1_Sdmode = 0.25;     dA1_Sdsd = 0.05

dA1_ = np.zeros([1000])
for i in range(0, 1000):
    dA1Mode_sh = getShape(dA1_modemode, dA1_modesd)
    dA1Mode_ra = getRate(dA1_modemode, dA1_modesd)
    dA1_mode = np.random.gamma(shape=dA1Mode_sh, scale=1 / dA1Mode_ra, size=1)

    dA1Sd_sh = getShape(dA1_Sdmode, dA1_Sdsd)
    dA1Sd_ra = getRate(dA1_Sdmode, dA1_Sdsd)
    dA1_sd = np.random.gamma(shape=dA1Sd_sh, scale=1 / dA1Sd_ra, size=1)

    dA1_sh = getShape(dA1_mode, dA1_sd)
    dA1_ra = getRate(dA1_mode, dA1_sd)
    dA1_[i] = np.random.gamma(shape=dA1_sh, scale=1 / dA1_ra, size=1)

plt.hist(dA1_)

###############################################################################################
####            dB1            ################################################################
###############################################################################################
dB1_modemode = 5;      dB1_modesd = 0.35
dB1_Sdmode = 0.25;     dB1_Sdsd = 0.05

dB1_ = np.zeros([1000])
for i in range(0, 1000):
    dB1Mode_sh = getShape(dB1_modemode, dB1_modesd)
    dB1Mode_ra = getRate(dB1_modemode, dB1_modesd)
    dB1_mode = np.random.gamma(shape=dB1Mode_sh, scale=1 / dB1Mode_ra, size=1)

    dB1Sd_sh = getShape(dB1_Sdmode, dB1_Sdsd)
    dB1Sd_ra = getRate(dB1_Sdmode, dB1_Sdsd)
    dB1_sd = np.random.gamma(shape=dB1Sd_sh, scale=1 / dB1Sd_ra, size=1)

    dB1_sh = getShape(dB1_mode, dB1_sd)
    dB1_ra = getRate(dB1_mode, dB1_sd)
    dB1_[i] = np.random.gamma(shape=dB1_sh, scale=1 / dB1_ra, size=1)

plt.hist(dB1_)

####                      Priors                       ########################################
###############################################################################################

###############################################################################################
####            A1f            ################################################################
###############################################################################################
A1f_ = np.zeros([1000])
for i in range(0, 1000):
    mu_a1mu_f = 1.4;
    s_a1mu_f = 0.1
    A1f_mu = np.random.normal(loc=mu_a1mu_f, scale=s_a1mu_f, size=1)
    mode_a1sd_f = 0.3;
    s_a1sd_f = 0.01
    sh_a1fsd = getShape(mode_a1sd_f, s_a1sd_f);
    ra_a1fsd = getRate(mode_a1sd_f, s_a1sd_f);
    A1f_sd = np.random.gamma(shape=sh_a1fsd, scale=1 / ra_a1fsd, size=1)
    A1f_[i] = np.random.normal(loc=A1f_mu, scale=A1f_sd, size=1)

plt.hist(expit(A1f_))
plt.hist(A1f_)

mu_a1mu_f = 1.5;
s_a1mu_f = 0.6
A1f_mu = np.random.normal(loc=mu_a1mu_f, scale=s_a1mu_f, size=100)
plt.hist(A1f_mu)
np.mean(expit(A1f_mu))

np.max(A1f_mu)
expit(np.max(A1f_mu))

np.min(A1f_mu)
expit(np.min(A1f_mu))

expit(0.5)
expit(1.5)
expit(2.5)


###############################################################################################
####            B1f            ################################################################
###############################################################################################

B1f_ = np.zeros([1000])
for i in range(0, 1000):
    mu_b1mu_f = -1.5;
    s_b1mu_f = 0.2
    B1f_mu = np.random.normal(loc=mu_b1mu_f, scale=s_b1mu_f, size=1)
    mode_b1sd_f = 0.25;
    s_b1sd_f = 0.05
    sh_b1fmu = getShape(mode_b1sd_f, s_b1sd_f);
    ra_b1fmu = getRate(mode_b1sd_f, s_b1sd_f);
    B1f_sd = np.random.gamma(shape=sh_b1fmu, scale=1 / ra_b1fmu, size=1)
    B1f_[i] = np.random.normal(loc=B1f_mu, scale=B1f_sd, size=1)

import seaborn as sns

sns.distplot(B1f_, kde=True, bins=20, hist=True)
np.mean(B1f_)
np.sqrt(np.var(B1f_))

plt.hist(expit(B1f_))
plt.hist(B1f_)


###############################################################################################
####            A1s            ################################################################
###############################################################################################

A1s_ = np.zeros([1000])
for i in range(0, 1000):
    mu_a1mu_s = 2.5;
    s_a1mu_s = 0.001
    A1s_mu = np.random.normal(loc=mu_a1mu_s, scale=s_a1mu_s, size=1)
    mode_a1sd_s = 0.3;
    s_a1sd_s = 0.05
    sh_a1smu = getShape(mode_a1sd_s, s_a1sd_s);
    ra_a1smu = getRate(mode_a1sd_s, s_a1sd_s);
    A1s_sd = np.random.gamma(shape=sh_a1smu, scale=1 / ra_a1smu, size=1)
    A1s_[i] = np.random.normal(loc=A1s_mu, scale=A1s_sd, size=1)

plt.hist(expit(A1s_))
np.max(expit(A1s_))
plt.hist(A1s_)

###############################################################################################
####            B1s            ################################################################
###############################################################################################

B1s_ = np.zeros([1000])
for i in range(0, 1000):
    mu_b1mu_s = -3.1;
    s_b1mu_s = 0.12
    B1s_mu = np.random.normal(loc=mu_b1mu_s, scale=s_b1mu_s, size=1)
    mode_b1sd_s = 0.5;
    s_b1sd_s = 0.09
    sh_b1smu = getShape(mode_b1sd_s, s_b1sd_s);
    ra_b1smu = getRate(mode_b1sd_s, s_b1sd_s);
    B1s_sd = np.random.gamma(shape=sh_b1smu, scale=1 / ra_b1smu, size=1)
    B1s_[i] = np.random.normal(loc=B1s_mu, scale=B1s_sd, size=1)

plt.hist(expit(B1s_))
plt.hist(B1s_)


###############################################################################################
####            sep            ################################################################
###############################################################################################

sep_ = np.zeros([1000])
for i in range(0, 1000):
    sep_modemode = 5;
    sep_modesd = 0.35
    sepMode_sh = getShape(sep_modemode, sep_modesd)
    sepMode_ra = getRate(sep_modemode, sep_modesd)
    sep_mode = np.random.gamma(shape=sepMode_sh, scale=1 / sepMode_ra, size=1)
    sep_Sdmode = 0.25;
    sep_Sdsd = 0.05
    sepSd_sh = getShape(sep_Sdmode, sep_Sdsd)
    sepSd_ra = getRate(sep_Sdmode, sep_Sdsd)
    sep_sd = np.random.gamma(shape=sepSd_sh, scale=1 / sepSd_ra, size=1)

    sep_sh = getShape(sep_mode, sep_sd)
    sep_ra = getRate(sep_mode, sep_sd)
    sep_[i] = np.random.gamma(shape=sep_sh, scale=1 / sep_ra, size=1)

plt.hist(sep_)

###############################################################################################
####            set            ################################################################
###############################################################################################

set_ = np.zeros([1000])
for i in range(0, 1000):
    set_modemode = 1;
    set_modesd = 0.5
    setMode_sh = getShape(set_modemode, set_modesd)
    setMode_ra = getRate(set_modemode, set_modesd)
    set_mode = np.random.gamma(shape=setMode_sh, scale=1 / setMode_ra, size=1)
    set_Sdmode = 0.2;
    set_Sdsd = 0.04
    setSd_sh = getShape(set_Sdmode, set_Sdsd)
    setSd_ra = getRate(set_Sdmode, set_Sdsd)
    set_sd = np.random.gamma(shape=setSd_sh, scale=1 / setSd_ra, size=1)

    set_sh = getShape(set_mode, set_sd)
    set_ra = getRate(set_mode, set_sd)
    set_[i] = np.random.gamma(shape=set_sh, scale=1 / set_ra, size=1)

plt.hist(set_)
