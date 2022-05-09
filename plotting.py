import arviz as az
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit, logit
import os

h = 6

####################################################################################
# Reading the Generated Samples  ###################################################
path = '/Applications/Diego Alejandro/2021-2 (Internship)/2ssm_stan/DataGeneration/';os.chdir(path);os.getcwd()

generatedParameters = pd.read_csv('generatedParameters.csv', header=0);print(generatedParameters.head())
generatedParameters = generatedParameters.values;
generatedParameters = generatedParameters[:, 1:h + 1]
generatedParameters[0:5, :]

As_gen = generatedParameters[:, 0]
Bs_gen = generatedParameters[:, 1]
Af_gen = generatedParameters[:, 2]
Bf_gen = generatedParameters[:, 3]
s2eps_gen = generatedParameters[:, 4]
s2eta_gen = generatedParameters[:, 5]

xf_gen = pd.read_csv('xf_generated.csv', header=0).values;xf_gen = xf_gen[:, 1:(h + 1)]
xs_gen = pd.read_csv('xs_generated.csv', header=0).values;xs_gen = xs_gen[:, 1:(h + 1)]
x_gen = pd.read_csv('x_generated.csv', header=0).values;x_gen = x_gen[:, 1:(h + 1)]
y_gen = pd.read_csv('y_generated.csv', header=0).values;y_gen = y_gen[:, 1:(h + 1)]

gens = [Af_gen[0:6], Bf_gen[0:6], As_gen[0:6], Bs_gen[0:6], np.sqrt(s2eps_gen[0:6]), np.sqrt(s2eta_gen[0:6])]
coln = ["Af", "Bf", "As", "Bs", "sep", "set"]        # FORMAT ORGANIZATION for the parameters #

####################################################################################
# Reading Samples from Stan Together ###############################################
path = '/Applications/Diego Alejandro/2021-2 (Internship)/2ssm_stan/aem_2s_newPerturbation/SampleWithLoglikehood';os.chdir(path);os.getcwd()

Af_stan = pd.read_csv('Af_trace.csv', header=0).values;Af_stan = Af_stan[:, 1:(h + 1)]
Bf_stan = pd.read_csv('Bf_trace.csv', header=0).values;Bf_stan = Bf_stan[:, 1:(h + 1)]
As_stan = pd.read_csv('As_trace.csv', header=0).values;As_stan = As_stan[:, 1:(h + 1)]
Bs_stan = pd.read_csv('Bs_trace.csv', header=0).values;Bs_stan = Bs_stan[:, 1:(h + 1)]
sep_stan = pd.read_csv('sep_trace.csv', header=0).values;sep_stan = sep_stan[:, 1:(h + 1)]
set_stan = pd.read_csv('set_trace.csv', header=0).values;set_stan = set_stan[:, 1:(h + 1)]
loaded = np.loadtxt("Xf_trace.txt");Xf_stan = loaded.reshape(loaded.shape[0], loaded.shape[1] // h, h)
loaded = np.loadtxt("Xs_trace.txt");Xs_stan = loaded.reshape(loaded.shape[0], loaded.shape[1] // h, h)
X_stan = Xf_stan + Xs_stan

col = [Af_stan, Bf_stan, As_stan, Bs_stan, sep_stan, set_stan]
hdis = [np.zeros([h, 2]), np.zeros([h, 2]), np.zeros([h, 2]), np.zeros([h, 2]), np.zeros([h, 2]), np.zeros([h, 2])]
modes = [np.zeros([h, 1]), np.zeros([h, 1]), np.zeros([h, 1]), np.zeros([h, 1]), np.zeros([h, 1]), np.zeros([h, 1])]

import aus_newPerLik

for g in range(0, len(hdis)):
    modes[g] = pd.DataFrame(np.around(col[g], 3)).apply(aus_newPerLik.mode_estimation, 0)
    hdis[g] = az.hdi(col[g])
####################################################################################
# Reading Samples from Stan Individually ##########################################
pat_ = ['ind1/', 'ind2/', 'ind3/', 'ind4/', 'ind5/', 'ind6/']
Af_stan_ = []; Bf_stan_ = []; As_stan_ = []; Bs_stan_ = []; sep_stan_ = []; set_stan_ = []
for z in range(0, h):
    path = '/Applications/Diego Alejandro/2021-2 (Internship)/2ssm_stan/aem_2s_newPerturbation/SampleWithLoglikehood/IndividualStans/' + \
           pat_[z];os.chdir(path);os.getcwd()
    Af_stan_ind = pd.read_csv('Af_trace.csv', header=0).values;Af_stan_ind = Af_stan_ind[:, 1:2]
    Bf_stan_ind = pd.read_csv('Bf_trace.csv', header=0).values;Bf_stan_ind = Bf_stan_ind[:, 1:2]
    As_stan_ind = pd.read_csv('As_trace.csv', header=0).values;As_stan_ind = As_stan_ind[:, 1:2]
    Bs_stan_ind = pd.read_csv('Bs_trace.csv', header=0).values;Bs_stan_ind = Bs_stan_ind[:, 1:2]
    sep_stan_ind = pd.read_csv('sep_trace.csv', header=0).values;sep_stan_ind = sep_stan_ind[:, 1:2]
    set_stan_ind = pd.read_csv('set_trace.csv', header=0).values;set_stan_ind = set_stan_ind[:, 1:2]

    Af_stan_.append(Af_stan_ind)
    Bf_stan_.append(Bf_stan_ind)
    As_stan_.append(As_stan_ind)
    Bs_stan_.append(Bs_stan_ind)
    sep_stan_.append(sep_stan_ind)
    set_stan_.append(set_stan_ind)

col_ = [Af_stan_, Bf_stan_, As_stan_, Bs_stan_, sep_stan_, set_stan_]
hdis_ = [np.zeros([h, 2]), np.zeros([h, 2]), np.zeros([h, 2]), np.zeros([h, 2]), np.zeros([h, 2]), np.zeros([h, 2])]
modes_ = [np.zeros([h, 1]), np.zeros([h, 1]), np.zeros([h, 1]), np.zeros([h, 1]), np.zeros([h, 1]), np.zeros([h, 1])]

modes_[0]
col_[0][2]
for g_ in range(0, len(col_)):
    print(g_)
    temp = np.zeros([h, 1]);
    temp1 = np.zeros([h, 2])  # this was necessary otherwise python was just not overwritting the values
    for u_ in range(0, h):
        temp[u_] = aus_newPerLik.mode_estimation(np.around(col_[g_][u_], 3))
        temp1[u_, :] = az.hdi(col_[g_][u_])
    modes_[g_] = temp
    hdis_[g_] = temp1

hdis_[1]
hdis_[2]
####################################################################################
# Reading Samples from EM Individually #############################################
path = '/Applications/Diego Alejandro/2021-2 (Internship)/2ssm_stan/aem_2s_newPerturbation/SampleWithLoglikehood/EMmatlab/';os.chdir(path);os.getcwd()

ind1_He = pd.read_csv('ind1_He.txt', header=None).values;print(ind1_He)
ind2_He = pd.read_csv('ind2_He.txt', header=None).values;print(ind2_He)
ind3_He = pd.read_csv('ind3_He.txt', header=None).values;print(ind3_He)
ind4_He = pd.read_csv('ind4_He.txt', header=None).values;print(ind4_He)
ind5_He = pd.read_csv('ind5_He.txt', header=None).values;print(ind5_He)
ind6_He = pd.read_csv('ind6_He.txt', header=None).values;print(ind6_He)

from scipy.linalg import sqrtm

hes = [ind1_He, ind2_He, ind3_He, ind4_He, ind5_He, ind6_He]
isIt = []
for o in range(0, len(hes)):
    isIt.append(np.linalg.inv(sqrtm(hes[o])))

ind1_PE = pd.read_csv('ind1_PE.txt', header=None).values;print(ind1_PE)
ind2_PE = pd.read_csv('ind2_PE.txt', header=None).values;print(ind2_PE)
ind3_PE = pd.read_csv('ind3_PE.txt', header=None).values;print(ind3_PE)
ind4_PE = pd.read_csv('ind4_PE.txt', header=None).values;print(ind4_PE)
ind5_PE = pd.read_csv('ind5_PE.txt', header=None).values;print(ind5_PE)
ind6_PE = pd.read_csv('ind6_PE.txt', header=None).values;print(ind6_PE)

EP_matlab = np.concatenate((ind1_PE, ind2_PE, ind3_PE, ind4_PE, ind5_PE, ind6_PE), axis=0)
EP_matlab
EP_matlab.shape
# aS    aF   bS   bF   xS1   xF1   sigmax2    sigmau2  sigma12
# here we unconstrained them...
As_matlab_un = logit( EP_matlab[:, 0] )
Af_matlab_un = logit( EP_matlab[:, 1] )
Bs_matlab_un = logit( EP_matlab[:, 2] )
Bf_matlab_un = logit( EP_matlab[:, 3] )
# set_matlab_un = np.log( np.sqrt(EP_matlab[:, 6]) ) this is a mistake, you get the np.sqrt after computing the intervals
# sep_matlab_un = np.log( np.sqrt(EP_matlab[:, 7]) )
set_matlab_un = np.log( EP_matlab[:, 6] )
sep_matlab_un = np.log( EP_matlab[:, 7] )

col_mat_un = [Af_matlab_un, Bf_matlab_un, As_matlab_un, Bs_matlab_un, sep_matlab_un, set_matlab_un]
cis = [np.zeros([h, 2]), np.zeros([h, 2]), np.zeros([h, 2]), np.zeros([h, 2]), np.zeros([h, 2]), np.zeros([h, 2])]
cis_constrained = [np.zeros([h, 2]), np.zeros([h, 2]), np.zeros([h, 2]), np.zeros([h, 2]), np.zeros([h, 2]), np.zeros([h, 2])]

import scipy.stats

alpha = 0.975
zz = scipy.stats.norm.ppf(alpha)
for o in range(0, len(col_mat_un)):
    print(o)
    for q in range(0, len(col_mat_un[0])):
        print(q)
        cis[o][q, 0] = col_mat_un[o][q] - zz * isIt[q][o, o]
        cis[o][q, 1] = col_mat_un[o][q] + zz * isIt[q][o, o]
    if (o<4) : cis_constrained[o] = expit( cis[o] )
    else :
        cis_constrained[o] = np.sqrt( np.exp( cis[o] ) )
# cis_corrected = cis
# for o in range(0, len(col_mat_un)):
#     for q in range(0, len(col_mat_un)):
#         if (cis[o][q, 0] < 0): cis_corrected[o][q, 0] = 0.00001
#         if (cis[o][q, 1] > 1): cis_corrected[o][q, 1] = 0.99999

path = '/Applications/Diego Alejandro/2021-2 (Internship)/2ssm_stan/aem_2s_newPerturbation/SampleWithLoglikehood/';os.chdir(path);os.getcwd()
CI_Bootstrap1 = pd.read_csv('CIs_Bootstrap1.txt', header=None).values;print(CI_Bootstrap1)
CI_Bootstrap1 = np.delete(CI_Bootstrap1,[4,5],1) #1 means by cols
print(CI_Bootstrap1)
# hdis[0][0,:]
table_ = np.zeros([4,6*2])
# table_[0,0:2] = hdis[0][0,:]
CI_Bootstrap1[:,0]

# Af Bf As Bs sep set
o__ = [1,3,0,2,4,5]
# aS    aF   bS   bF   sigmax2    sigmau2  sigma12     check this order to add to the table and taking into account that xS1   xF1   were already deleter

for o in range(0, len(col_mat_un)):
    table_[0,o*2:o*2+2] = hdis[o][0, :]
    table_[1,o*2:o*2+2] = hdis_[o][0, :]
    table_[2,o*2:o*2+2] = cis_constrained[o][0, :]
    table_[3,o*2:o*2+2] = CI_Bootstrap1[:,o__[o] ]

# pd.DataFrame(table_).to_csv("table_CIs.csv")


# Plotting ########################################################################################################################################

############################################################################################################
#            Row Style           ###########################################################################
############################################################################################################

# from z=0-2
# hx = np.arange(logit(min(gens[z]))-0.3, logit(max(gens[z]))+0.3, 0.3);
# hy = np.arange(logit(min(gens[z]))-3.0, logit(max(gens[z]))+3.0, 0.3);
# for z = 3
# hx = np.arange(logit(min(gens[z]))-1.7, logit(max(gens[z]))+1.7, 0.3);
#     hy = np.arange(logit(min(gens[z]))-6.0, logit(max(gens[z]))+6.0, 0.3);

path = '/Applications/Diego Alejandro/2021-2 (Internship)/2ssm_stan/aem_2s_newPerturbation/SampleWithLoglikehood';os.chdir(path);os.getcwd()
for z in range(0, 4):  # the for just get until the previous one so its until 3
    plt.figure(figsize=(20, 20), dpi=1000);plt.tight_layout();    plt.title(coln[z])
    hx = np.arange(logit(min(gens[z])) - 0.3, logit(max(gens[z])) + 0.3, 0.3);
    hy = np.arange(logit(min(gens[z])) - 1.0, logit(max(gens[z])) + 1.0, 0.3);
    newhx = np.round(expit(hx), 3).astype(str).tolist();
    newhy = np.round(expit(hy), 3).astype(str).tolist()
    #####################  Stan - Together     ######
    ax1 = plt.subplot(1, 4, 1)
    ye = [abs(logit(modes[z]) - logit(hdis[z][:, 0])), abs(logit(hdis[z][:, 1]) - logit(modes[z]))]
    plt.errorbar(logit(gens[z]), logit(np.array(modes[z])), yerr=ye, fmt='o',
                 label='estimated posterior mode', color='k', markersize=8, capsize=5, linewidth=2)
    plt.axline([min(hx), min(hx)], [max(hx), max(hx)])
    plt.xlim([min(hx), max(hx)]);
    plt.ylim([min(hy), max(hy)])
    plt.xticks(hx, newhx, rotation=45);
    plt.yticks(hy, newhy)
    for i, txt in enumerate(np.round(gens[z], 3)):
        ax1.annotate('  ' + str(i), (logit(gens[z][i]), logit(modes[z][i])), color='b')
    plt.title(coln[z] + ' - Stan together');
    ax1.set_aspect('equal', adjustable='box')
    #####################  Stan - Individually ######
    ax2 = plt.subplot(1, 4, 2)
    ye_ = [abs(logit(modes_[z][:, 0]) - logit(hdis_[z][:, 0])), abs(logit(hdis_[z][:, 1]) - logit(modes_[z][:, 0]))]
    plt.errorbar(logit(gens[z]), logit(modes_[z][:,0]), yerr=ye_, fmt='o',
                 label='estimated posterior mode', color='k', markersize=8, capsize=5, linewidth=2)
    ax2.axline([min(hx), min(hx)], [max(hx), max(hx)])
    plt.xlim([min(hx), max(hx)]);
    plt.ylim([min(hy), max(hy)])
    plt.xticks(hx, newhx, rotation=45);
    plt.yticks(hy, newhy)
    plt.title(coln[z] + ' - Stan Individually');
    ax2.set_aspect('equal', adjustable='box')
    for i, txt in enumerate(np.round(gens[z], 3)):
        plt.annotate('  ' + str(i), (logit(gens[z][i]), logit(modes_[z][i])), color='b')
    # plt.savefig(coln[z]+"_fixed_.pdf",bbox_inches='tight')
    #####################  EM - Individually   ######
    ax3 = plt.subplot(1, 4, 3)
    plt.scatter(logit(gens[z]), col_mat_un[z] )
    ye_m = [ abs( col_mat_un[z]-cis[z][:,0] ) , abs( cis[z][:,1]-col_mat_un[z] ) ]
    # ye_m = [abs(col_mat_un[z] - logit(cis_corrected[z][:, 0])),
    #         abs(logit(cis_corrected[z][:, 1]) - col_mat_un[z]) ]
    # plt.errorbar(logit(gens[z]), np.array(col_mat_un[z]), yerr=ye_m, fmt='o',
    #              label='estimated posterior mode', color='k', markersize=8, capsize=5, linewidth=2)
    plt.errorbar(logit(gens[z]), col_mat_un[z], yerr=ye_m, fmt='o',
                 label='estimated posterior mode', color='k', markersize=8, capsize=5, linewidth=2)
    ax3.axline([min(hx), min(hx)], [max(hx), max(hx)])
    plt.xlim([min(hx), max(hx)]);
    plt.ylim([min(hy), max(hy)])
    plt.xticks(hx, newhx, rotation=45);
    plt.yticks(hy, newhy)
    plt.title(coln[z] + ' - EM Individually')
    ax3.set_aspect('equal', adjustable='box')
    for i, txt in enumerate(np.round(gens[z], 3)):
        plt.annotate('  ' + str(i), ( logit(gens[z][i]), col_mat_un[z][i] ), color='b')
    #####################  EM - Individually Bootstrap  #############
    ax4 = plt.subplot(1, 4, 4)
    plt.scatter(logit(gens[z]), col_mat_un[z])
    #!!!!!!!!!!!!!!!!!!!!!!!!!
    #here instead of using cis after getting the bootstrap confidence for the 6 individuals I should put those values
    ye_m = [abs(col_mat_un[z] - cis[z][:, 0]), abs(cis[z][:, 1] - col_mat_un[z])]
    plt.errorbar(logit(gens[z]), col_mat_un[z], yerr=ye_m, fmt='o',
                 label='estimated posterior mode', color='k', markersize=8, capsize=5, linewidth=2)
    ax3.axline([min(hx), min(hx)], [max(hx), max(hx)])
    plt.xlim([min(hx), max(hx)]);
    plt.ylim([min(hy), max(hy)])
    plt.xticks(hx, newhx, rotation=45);
    plt.yticks(hy, newhy)
    plt.title(coln[z] + ' - EM Individually')
    ax3.set_aspect('equal', adjustable='box')
    for i, txt in enumerate(np.round(gens[z], 3)):
        plt.annotate('  ' + str(i), (logit(gens[z][i]), col_mat_un[z][i]), color='b')

    plt.savefig(coln[z] + "_news.pdf", bbox_inches='tight')

# for z = 5 - set
# hx = np.arange( min(np.log(gens[z]))-0.5 , max(np.log(gens[z]))+0.5 , 0.3);
#     hy = np.arange( min(np.log(gens[z]))-0.5 , max(np.log(gens[z]))+0.5 , 0.3);


###### HERE CHECK WHY I DONT HAVE the scatter line for the points
for z in range(4, len(col)):  # Here for z = 4 and 5.
    z = 5;
    plt.figure(figsize=(20, 20), dpi=1000)
    hx = np.arange(min(np.log(gens[z])) - 0.5, max(np.log(gens[z])) + 0.5, 0.3);
    hy = np.arange(min(np.log(gens[z])) - 0.5, max(np.log(gens[z])) + 1.5, 0.3);
    #####################  Stan - Together
    ax1 = plt.subplot(1, 3, 1)
    ye = [abs(np.log(modes[z]) - np.log(hdis[z][:, 0])), abs(np.log(hdis[z][:, 1]) - np.log(modes[z]))]
    plt.errorbar(np.log(gens[z]), np.log(np.array(modes[z])), yerr=ye, fmt='o',
                 label='estimated posterior mode', color='k', markersize=8, capsize=5, linewidth=2)
    plt.axline([min(hx), min(hx)], [max(hx), max(hx)])
    plt.xlim([min(hx), max(hx)]);
    plt.ylim([min(hy), max(hy)])
    plt.title(coln[z] + ' - Stan together');
    ax1.set_aspect('equal', adjustable='box')
    for i, txt in enumerate(np.round(gens[z], 3)):
        ax1.annotate('  ' + str(i), (np.log(gens[z][i]), np.log(modes[z][i])), color='b')
    #####################  Stan - Individually
    ax2 = plt.subplot(1, 3, 2)
    ye_ = [abs(np.log(modes_[z][:, 0]) - np.log(hdis_[z][:, 0])), abs(np.log(hdis_[z][:, 1]) - np.log(modes_[z][:, 0]))]
    plt.errorbar(np.log(gens[z]), np.log(np.array(modes_[z][:,0])), yerr=ye_, fmt='o',
                 label='estimated posterior mode', color='k', markersize=8, capsize=5, linewidth=2)
    ax2.axline([min(hx), min(hx)], [max(hx), max(hx)])
    plt.xlim([min(hx), max(hx)]);
    plt.ylim([min(hy), max(hy)])
    plt.title(coln[z] + ' - Stan Individually');
    ax2.set_aspect('equal', adjustable='box')
    for i, txt in enumerate(np.round(gens[z], 3)):
        ax2.annotate('  ' + str(i), (np.log(gens[z][i]), np.log(modes_[z][i])), color='b')
    #####################  EM - Individually
    ax3 = plt.subplot(1, 3, 3)
    ye_m = [abs(col_mat_un[z] - cis[z][:, 0]),abs(cis[z][:, 1] - col_mat_un[z])]
    plt.errorbar(np.log(gens[z]), col_mat_un[z], yerr=ye_m, fmt='o',
                 label='estimated posterior mode', color='k', markersize=8, capsize=5, linewidth=2)
    # plt.scatter( gens[z] , col_mat[z] )
    ax3.axline([min(hx), min(hx)], [max(hx), max(hx)])
    plt.xlim([min(hx), max(hx)]);
    plt.ylim([min(hy), max(hy)])
    plt.title(coln[z] + ' - EM Individually');
    ax3.set_aspect('equal', adjustable='box')
    for i, txt in enumerate(np.round(gens[z], 3)):
        ax3.annotate('  ' + str(i), (np.log(gens[z][i]), col_mat_un[z][i] ), color='b')

    plt.savefig(coln[z] + "_news.pdf", bbox_inches='tight')







































