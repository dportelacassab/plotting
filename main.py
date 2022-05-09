import pystan;
import stan
import arviz as az
import pandas as pd;
import numpy as np
import matplotlib.pyplot as plt;
from scipy.special import expit, logit
import time;
import os
import seaborn

h = 60  # number of subjects
p_ = np.concatenate([np.repeat(np.array([0]), 135),
                     np.repeat(np.array([-30]), 108),
                     np.repeat(np.array([30]), 12),
                     np.repeat(np.array([0]), 120)])

cn = np.concatenate([np.repeat(np.array([1]), 135 + 108 + 12),
                     np.repeat(np.array([0]), 60),
                     np.repeat(np.array([1]), 60)])
n = p_.size

path = '/Applications/Diego Alejandro/2021-2 (Internship)/2ssm_stan/DataGeneration/'
os.chdir(path);
os.getcwd()

generatedParameters = pd.read_csv('generatedParameters.csv', header=0);
print(generatedParameters.head())
generatedParameters = generatedParameters.values;
generatedParameters = generatedParameters[:, 1:h + 1]

As_gen = generatedParameters[:, 0]
Bs_gen = generatedParameters[:, 1]
Af_gen = generatedParameters[:, 2]
Bf_gen = generatedParameters[:, 3]
s2eps_gen = generatedParameters[:, 4]
s2eta_gen = generatedParameters[:, 5]

xf_gen = pd.read_csv('xf_generated.csv', header=0).values;xf_gen = xf_gen[:, 1:(h + 1)]
xs_gen = pd.read_csv('xs_generated.csv', header=0).values;xs_gen = xs_gen[:, 1:(h + 1)]
y_gen = pd.read_csv('y_generated.csv', header=0).values;y_gen = y_gen[:, 1:(h + 1)]

for ind_ in range(0, h - 58):
    fig = plt.figure(figsize=(15, 10), dpi=380);
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    axes.plot(-y_gen[:, ind_] + p_, 'b-', -xs_gen[:, ind_] - xf_gen[:, ind_], 'r-', -xs_gen[:, ind_], 'y-',
              -xf_gen[:, ind_], 'g-', p_, 'k-')
    axes.legend(['y', 'x', 'xs', 'xf', 'p'], loc="best", prop={'size': 10});
    axes.set_title(
        f'Subject: {ind_} As_ = {As_gen[ind_]:.4f}, Bs_ = {Bs_gen[ind_]:.4f}, Af_ = {Af_gen[ind_]:.4f}, Bf_ = {Bf_gen[ind_]:.4f}, sep_ = {np.sqrt(s2eps_gen[ind_]):1.4f}, set_ = {np.sqrt(s2eta_gen[ind_]):1.4f}')

h1 = 6
y_gen_task1 = y_gen[:, 0:h1]
y_gen_task1.shape
y_gen_task1[0:10, :]

# h1 is gonna be 12 from now
da1 = {
    'n': n,
    'h': h1,
    'cn': cn,
    'p': p_,
    'Y': y_gen_task1,

    'mu_a1mu_f': 1.4, 's_a1mu_f': 0.1,
    'mode_a1sd_f': 0.3, 's_a1sd_f': 0.01,

    'mu_b1mu_f': -1.5, 's_b1mu_f': 0.2,
    'mode_b1sd_f': 0.25, 's_b1sd_f': 0.05,

    'mu_a1mu_s': 2.5, 's_a1mu_s': 0.001,
    'mode_a1sd_s': 0.3, 's_a1sd_s': 0.05,

    'mu_b1mu_s': -3.1, 's_b1mu_s': 0.12,
    'mode_b1sd_s': 0.5, 's_b1sd_s': 0.09,

    'sep_modemode': 5, 'sep_modesd': 0.35,
    'sep_Sdmode': 0.25, 'sep_Sdsd': 0.05,

    'set_modemode': 1, 'set_modesd': 0.5,
    'set_Sdmode': 0.2, 'set_Sdsd': 0.04
}

path = '/Applications/Diego Alejandro/2021-2 (Internship)/2ssm_stan/aem_2s_newPerturbation/SampleWithLoglikehood/'
os.chdir(path);
os.getcwd()
import aus_newPerLik

# aus1.bm
s1 = time.time()
smt = pystan.StanModel(model_code=aus_newPerLik.bm, verbose=True)
e1 = time.time();
print(f"Runtime: {(e1 - s1) / 30} minutes")

path = '/Applications/Diego Alejandro/2021-2 (Internship)/2ssm_stan/aem_2s_newPerturbation/SampleWithLoglikehood/moreInd/'
os.chdir(path);
os.getcwd()


def initfun1():
    return dict(A1mu_f=0.25, A1s_f=0.30,
                B1mu_f=-1.00, B1s_f=0.15,
                A1mu_s=3.50, A1s_s=0.25,
                B1mu_s=-2.26, B1s_s=0.21,
                sep_mode=5.00, sep_sd=0.25,
                set_mode=1.00, set_sd=0.20,
                A1f=0.40 * np.ones([h1]), B1f=-1.38 * np.ones([h1]),
                A1s=2.75 * np.ones([h1]), B1s=-2.19 * np.ones([h1]),
                Af=0.60 * np.ones([h1]), Bf=0.20 * np.ones([h1]),
                As=0.94 * np.ones([h1]), Bs=0.10 * np.ones([h1]),
                s_ep=1.00 * np.ones([h1]), s_et=1.10 * np.ones([h1]))


s2 = time.time()
samplesp = smt.sampling(data=da1, iter=10000, chains=4, warmup=4000, seed=1, init=initfun1)
# samplesp = smt.sampling(data=da1, iter=10000, chains=4,warmup = 4000,seed = 1)
e2 = time.time();
print(f"Runtime: {(e2 - s2) / 60} minutes")

# Parameters ranges
# As - from 0.95-0.995
# Bs - from 0.05-0.15
# Af - from 0.4-0.8
# Bf - from 0.15-0.5
# sep- from 0.5-1.5
# set- from 0.05-1.5

# 12 individuals

print(samplesp.stansummary(["Af"]));
print(Af_gen[0:6])
print(samplesp.stansummary(["Bf"]));
print(Bf_gen[0:6])
print(samplesp.stansummary(["As"]));
print(As_gen[0:6])
print(samplesp.stansummary(["Bs"]));
print(Bs_gen[0:6])
print(samplesp.stansummary(["s_ep"]));
print(np.sqrt(s2eps_gen[0:6]))
print(samplesp.stansummary(["s_et"]));
print(np.sqrt(s2eta_gen[0:6]))

print(samplesp.stansummary(["A1mu_f"]))
print(samplesp.stansummary(["A1s_f"]))
print(samplesp.stansummary(["B1s_f"]))
print(samplesp.stansummary(["B1s_s"]))

print(samplesp.stansummary(["ll_"]))

# print(samplesp.stansummary(["A1f"]));print(A1f_gen)

# path = '/Applications/Diego Alejandro/2021-2 (Internship)/2ssm_stan/aem_2s_newPerturbation/10.000 samples'
path = '/Applications/Diego Alejandro/2021-2 (Internship)/2ssm_stan/aem_2s_newPerturbation/SampleWithLoglikehood'
os.chdir(path);
os.getcwd()

h = h1
# Extracting all
# constrained variables
Af_stan = samplesp.extract(['Af']);Af_stan = Af_stan['Af']
Bf_stan = samplesp.extract(['Bf']);Bf_stan = Bf_stan['Bf']
As_stan = samplesp.extract(['As']);As_stan = As_stan['As']
Bs_stan = samplesp.extract(['Bs']);Bs_stan = Bs_stan['Bs']
s_ep_stan = samplesp.extract(['s_ep']);s_ep_stan = s_ep_stan['s_ep']
s_et_stan = samplesp.extract(['s_et']);s_et_stan = s_et_stan['s_et']
Xf_stan = samplesp.extract(['Xf']);Xf_stan = Xf_stan['Xf'];
Xf_reshaped = Xf_stan.reshape(Xf_stan.shape[0], -1);Xf_reshaped.shape
Xs_stan = samplesp.extract(['Xs']);
Xs_stan = Xs_stan['Xs'];
Xs_reshaped = Xs_stan.reshape(Xs_stan.shape[0], -1);Xs_reshaped.shape
ll_stan = samplesp.extract(['ll_']);ll_stan = ll_stan['ll_']
# unconstrained variables
A1f_stan = samplesp.extract(['A1f']);A1f_stan = A1f_stan['A1f']
B1f_stan = samplesp.extract(['B1f']);B1f_stan = B1f_stan['B1f']
A1s_stan = samplesp.extract(['A1s']);A1s_stan = A1s_stan['A1s']
B1s_stan = samplesp.extract(['B1s']);B1s_stan = B1s_stan['B1s']
# prior extraction
A1mu_f_stan = samplesp.extract(['A1mu_f']);
A1s_f_stan = samplesp.extract(['A1s_f']);A1s_f_stan
B1mu_f_stan = samplesp.extract(['B1mu_f']);
B1s_f_stan = samplesp.extract(['B1s_f']);
A1mu_s_stan = samplesp.extract(['A1mu_s']);
A1s_s_stan = samplesp.extract(['A1s_s']);
B1mu_s_stan = samplesp.extract(['B1mu_s']);
B1s_s_stan = samplesp.extract(['B1s_s']);

sep_mode_stan = samplesp.extract(['sep_mode']);
sep_sd_stan = samplesp.extract(['sep_sd']);
set_mode_stan = samplesp.extract(['set_mode']);
set_sd_stan = samplesp.extract(['set_sd']);

# Saving all
# constrained variables
pd.DataFrame(Af_stan).to_csv("Af_trace.csv")
pd.DataFrame(Bf_stan).to_csv("Bf_trace.csv")
pd.DataFrame(As_stan).to_csv("As_trace.csv")
pd.DataFrame(Bs_stan).to_csv("Bs_trace.csv")
pd.DataFrame(s_ep_stan).to_csv("sep_trace.csv")
pd.DataFrame(s_et_stan).to_csv("set_trace.csv")
np.savetxt("Xf_trace.txt", Xf_reshaped)
np.savetxt("Xs_trace.txt", Xs_reshaped)
pd.DataFrame(ll_stan).to_csv("ll_trace.csv")
# unconstrained variables
pd.DataFrame(A1f_stan).to_csv("A1f_trace.csv")
pd.DataFrame(B1f_stan).to_csv("B1f_trace.csv")
pd.DataFrame(A1s_stan).to_csv("A1s_trace.csv")
pd.DataFrame(B1s_stan).to_csv("B1s_trace.csv")
# priors
pd.DataFrame(A1mu_f_stan).to_csv("A1mu_f_trace.csv")
pd.DataFrame(A1s_f_stan).to_csv("A1s_f_trace.csv")
pd.DataFrame(B1mu_f_stan).to_csv("B1mu_f_trace.csv")
pd.DataFrame(B1s_f_stan).to_csv("B1s_f_trace.csv")
pd.DataFrame(A1mu_s_stan).to_csv("A1mu_s_trace.csv")
pd.DataFrame(A1s_s_stan).to_csv("A1s_s_trace.csv")
pd.DataFrame(B1mu_s_stan).to_csv("B1mu_s_trace.csv")
pd.DataFrame(B1s_s_stan).to_csv("B1s_s_trace.csv")
pd.DataFrame(sep_mode_stan).to_csv("sep_mode_trace.csv")
pd.DataFrame(sep_sd_stan).to_csv("sep_sd_trace.csv")
pd.DataFrame(set_mode_stan).to_csv("set_mode_trace.csv")
pd.DataFrame(set_sd_stan).to_csv("set_sd_trace.csv")

# Reading
Af_stan = pd.read_csv('Af_trace.csv', header=0).values;Af_stan = Af_stan[:, 1:(h + 1)]
Bf_stan = pd.read_csv('Bf_trace.csv', header=0).values;Bf_stan = Bf_stan[:, 1:(h + 1)]
As_stan = pd.read_csv('As_trace.csv', header=0).values;As_stan = As_stan[:, 1:(h + 1)]
Bs_stan = pd.read_csv('Bs_trace.csv', header=0).values;Bs_stan = Bs_stan[:, 1:(h + 1)]
sep_stan = pd.read_csv('sep_trace.csv', header=0).values;sep_stan = sep_stan[:, 1:(h + 1)]
set_stan = pd.read_csv('set_trace.csv', header=0).values;set_stan = set_stan[:, 1:(h + 1)]
loaded = np.loadtxt("Xf_trace.txt");Xf_stan = loaded.reshape(loaded.shape[0], loaded.shape[1] // h, h)
loaded = np.loadtxt("Xs_trace.txt");Xs_stan = loaded.reshape(loaded.shape[0], loaded.shape[1] // h, h)
X_stan = Xf_stan + Xs_stan

B1f_sd_stan = pd.read_csv('B1s_f_trace.csv', header=0);
B1f_sd_stan.hist('B1s_f')
# A1f_sd_stan.shape
# A1f_sd_stan

# before going to the WAIC, check the traces, look that neff samples are still very low
# !!!!!!


# improve the plot
# לשפר את הפלוט
# את הגרף - graf - grafico

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# axes.xaxis.set_ticks(np.arange(np.min(gens[z])-0.05, np.max(gens[z])+0.05, 0.04))
# axes.yaxis.set_ticks(np.arange(np.min(hdis[z]), np.max(hdis[z]), ticy[z]))
# put the x ticks evenly spaced in the unconstraiend space
# but with the constrained labels
# For teh plots of A_s and A_f, label as 1-A instead of A (
# So if A is 0.99 you label the point as 0.01 and if A is 0.95 you label it
# as 0.05 (percent forgotten)
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

for i, txt in enumerate(np.round(gens[z], 3)):
    axes.annotate('  ' + str(i) + '\n  ' + str(txt), (gens[z][i], modes[z][i]), color='b')
axes.set_xlabel('generated');
axes.set_ylabel('mode');
axes.set_title(coln[z])


import aus_newPerLik_single

pat_ = ['ind1/', 'ind2/', 'ind3/', 'ind4/', 'ind5/', 'ind6/']
s1 = time.time()
smt_ind = pystan.StanModel(model_code=aus_newPerLik_single.bm, verbose=True)
e1 = time.time();
print(f"Runtime: {(e1 - s1) / 30} minutes")


def initfun2():
    return dict(A1f=0.40, B1f=-1.38,
                A1s=2.75, B1s=-2.19,
                Af=0.60, Bf=0.20,
                As=0.94, Bs=0.10,
                s_ep=1.00, s_et=1.10)


for z in range(0, h1):
    da1_ind = {
        'n': n,
        'cn': cn,
        'p': p_,
        'Y': y_gen[:, z],
        'A1mu_f': 0.25, 'A1s_f': 0.30,
        'B1mu_f': -1.00, 'B1s_f': 0.15,
        'A1mu_s': 3.50, 'A1s_s': 0.25,
        'B1mu_s': -2.26, 'B1s_s': 0.21,
        'sep_mode': 5.00, 'sep_sd': 0.25,
        'set_mode': 1.00, 'set_sd': 0.20
    }

    s2 = time.time()
    samplesp_ind = smt_ind.sampling(data=da1_ind, iter=10000, chains=4, warmup=4000, seed=1, init=initfun2)
    e2 = time.time();
    print(f"Runtime: {(e2 - s2) / 60} minutes")

    path = '/Applications/Diego Alejandro/2021-2 (Internship)/2ssm_stan/aem_2s_newPerturbation/SampleWithLoglikehood/IndividualStans/' + \
           pat_[z]
    os.chdir(path);
    os.getcwd()

    # Extracting all
    # constrained variables
    Af_stan = samplesp_ind.extract(['Af']);
    Af_stan = Af_stan['Af']
    Bf_stan = samplesp_ind.extract(['Bf']);
    Bf_stan = Bf_stan['Bf']
    As_stan = samplesp_ind.extract(['As']);
    As_stan = As_stan['As']
    Bs_stan = samplesp_ind.extract(['Bs']);
    Bs_stan = Bs_stan['Bs']
    s_ep_stan = samplesp_ind.extract(['s_ep']);
    s_ep_stan = s_ep_stan['s_ep']
    s_et_stan = samplesp_ind.extract(['s_et']);
    s_et_stan = s_et_stan['s_et']
    Xf_stan = samplesp_ind.extract(['Xf']);
    Xf_stan = Xf_stan['Xf'];
    Xf_reshaped = Xf_stan.reshape(Xf_stan.shape[0], -1);
    Xf_reshaped.shape
    Xs_stan = samplesp_ind.extract(['Xs']);
    Xs_stan = Xs_stan['Xs'];
    Xs_reshaped = Xs_stan.reshape(Xs_stan.shape[0], -1);
    Xs_reshaped.shape
    # ll_stan = samplesp_ind.extract(['ll_']);ll_stan = ll_stan['ll_']
    # unconstrained variables   #prior extraction
    A1f_stan = samplesp_ind.extract(['A1f']);
    A1f_stan = A1f_stan['A1f']
    B1f_stan = samplesp_ind.extract(['B1f']);
    B1f_stan = B1f_stan['B1f']
    A1s_stan = samplesp_ind.extract(['A1s']);
    A1s_stan = A1s_stan['A1s']
    B1s_stan = samplesp_ind.extract(['B1s']);
    B1s_stan = B1s_stan['B1s']

    # Saving all
    # constrained variables
    pd.DataFrame(Af_stan).to_csv("Af_trace.csv")
    pd.DataFrame(Bf_stan).to_csv("Bf_trace.csv")
    pd.DataFrame(As_stan).to_csv("As_trace.csv")
    pd.DataFrame(Bs_stan).to_csv("Bs_trace.csv")
    pd.DataFrame(s_ep_stan).to_csv("sep_trace.csv")
    pd.DataFrame(s_et_stan).to_csv("set_trace.csv")
    np.savetxt("Xf_trace.txt", Xf_reshaped)
    np.savetxt("Xs_trace.txt", Xs_reshaped)
    pd.DataFrame(ll_stan).to_csv("ll_trace.csv")
    # unconstrained variables
    pd.DataFrame(A1f_stan).to_csv("A1f_trace.csv")
    pd.DataFrame(B1f_stan).to_csv("B1f_trace.csv")
    pd.DataFrame(A1s_stan).to_csv("A1s_trace.csv")
    pd.DataFrame(B1s_stan).to_csv("B1s_trace.csv")

if __name__ == '__main__':
