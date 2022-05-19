import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn
import scipy as sp

h = 60
url = 'https://raw.githubusercontent.com/dportelacassab/plotting/master/DataGeneration/'

####################################################################################
# Reading the Generated Samples from GitHub Ropository #############################
generatedParameters = pd.read_csv(url + 'generatedParameters.csv', header=0);
print(generatedParameters.head());generatedParameters = generatedParameters.values;generatedParameters = generatedParameters[:, 1:h + 1]

As_gen = generatedParameters[:, 0]
Bs_gen = generatedParameters[:, 1]
Af_gen = generatedParameters[:, 2]
Bf_gen = generatedParameters[:, 3]
s2eps_gen = generatedParameters[:, 4]
s2eta_gen = generatedParameters[:, 5]

gens = [Af_gen[0:h], Bf_gen[0:h], As_gen[0:h], Bs_gen[0:h], np.sqrt(s2eps_gen[0:h]), np.sqrt(s2eta_gen[0:h])]
coln = ["Af", "Bf", "As", "Bs", "sep", "set"]

####################################################################################################
#                 Reading Samples from Stan Together             ####################################
####################################################################################################
path = '/home/d/Documents/project/Samples_Grouped60/Bayesian_Together';os.chdir(path);os.getcwd()
path = '/home/d/Documents/project/Samples_Grouped60/togetherNew/together';os.chdir(path);os.getcwd()

Af_stan = pd.read_csv('Af_trace.csv', header=0).values;Af_stan = Af_stan[:, 1:(h + 1)]
Bf_stan = pd.read_csv('Bf_trace.csv', header=0).values;Bf_stan = Bf_stan[:, 1:(h + 1)]
As_stan = pd.read_csv('As_trace.csv', header=0).values;As_stan = As_stan[:, 1:(h + 1)]
Bs_stan = pd.read_csv('Bs_trace.csv', header=0).values;Bs_stan = Bs_stan[:, 1:(h + 1)]
sep_stan = pd.read_csv('sep_trace.csv', header=0).values;sep_stan = sep_stan[:, 1:(h + 1)]
set_stan = pd.read_csv('set_trace.csv', header=0).values;set_stan = set_stan[:, 1:(h + 1)]

#umm for ----Af_stan----- lets say I want to check the traces for a second
# ploting the pairplots for the parameters
for ind in range(0, 1):
    data = {'As': As_stan[:, ind], 'Bs': Bs_stan[:, ind], 'Af': Af_stan[:, ind], 'Bf': Bf_stan[:, ind],
            'sep_': sep_stan[:, ind], 'set_': set_stan[:, ind]}
    seaborn.pairplot(pd.DataFrame(data))
    plt.savefig("paiplot_" + str(ind) + ".pdf");
    plt.show()

# before going to the WAIC, check the traces, look that neff samples are still very low
# !!!!!!

As_stan.shape
As_stan[:, 4].shape
plt.plot(As_stan[:, 0])
plt.plot(As_stan[:, 1])
plt.plot(As_stan[:, 3])
plt.plot(As_stan[:, 2])
plt.plot(As_stan[:, 4])
plt.plot(As_stan[:, 5])

def loglike_(As_,Bs_,Af_,Bf_,sep_,set_,N_,xs_,xf_,p,y):
    sta = np.log( sp.stats.norm.pdf(xs_[0], 0,set_) ) + np.log( sp.stats.norm.pdf(xf_[0], 0,set_) )
    for i in range(0, N_-1):
        sta += ( np.log( sp.stats.norm.pdf(xs_[i+1], As_*xs_[i] - Bs_*y[i], set_) ) +
                 np.log( sp.stats.norm.pdf(xf_[i+1], Af_*xf_[i] - Bf_*y[i], set_) ) +
                 np.log( sp.stats.norm.pdf(y[i]    , xs_[i]+xf_[i]+p[i]   , sep_) ) );
    return sta



from multiprocessing import Pool
import time

for i in range(1,10):
    t0 = time.time()
    pool = Pool(8)
    result2 = pool.map(loglike_,  )
    pool.close()
    t1 = time.time()
    print(f"With multiprocessing we ran the function in {t1 - t0:0.4f} seconds")




plt.figure(figsize=(20, 20), dpi=1000);plt.tight_layout();plt.title('-')
for z in range(0,h-1):
    plt.subplot(12, 5, z+1);plt.title('ind'+str(z))
    plt.plot(Af_stan[:,z])
plt.savefig("Af_alls.pdf", bbox_inches='tight')


path = '/home/d/Documents/project/Samples_Grouped60/Bayesian_Together';os.chdir(path);os.getcwd()
path = '/home/d/Documents/project/Samples_Grouped60/togetherNew/together';os.chdir(path);os.getcwd()
# prior traces reading ####
A1mu_f_stan = pd.read_csv('A1mu_f_trace.csv', header=0).values;A1mu_f_stan = A1mu_f_stan[:, 1:(h + 1)]
A1s_f_stan = pd.read_csv('A1s_f_trace.csv', header=0).values;A1s_f_stan = A1s_f_stan[:, 1:(h + 1)]
B1mu_f_stan = pd.read_csv('B1mu_f_trace.csv', header=0).values;B1mu_f_stan = B1mu_f_stan[:, 1:(h + 1)]
B1s_f_stan = pd.read_csv('B1s_f_trace.csv', header=0).values;B1s_f_stan = B1s_f_stan[:, 1:(h + 1)]

A1mu_s_stan = pd.read_csv('A1mu_s_trace.csv', header=0).values;A1mu_s_stan = A1mu_s_stan[:, 1:(h + 1)]
A1s_s_stan = pd.read_csv('A1s_s_trace.csv', header=0).values;A1s_s_stan = A1s_s_stan[:, 1:(h + 1)]
B1mu_s_stan = pd.read_csv('B1mu_s_trace.csv', header=0).values;B1mu_s_stan = B1mu_s_stan[:, 1:(h + 1)]
B1s_s_stan = pd.read_csv('B1s_s_trace.csv', header=0).values;B1s_s_stan = B1s_s_stan[:, 1:(h + 1)]

sep_mode_stan = pd.read_csv('sep_mode_trace.csv', header=0).values;sep_mode_stan = sep_mode_stan[:, 1:(h + 1)]
sep_sd_stan = pd.read_csv('sep_sd_trace.csv', header=0).values;sep_sd_stan = sep_sd_stan[:, 1:(h + 1)]
set_mode_stan = pd.read_csv('set_mode_trace.csv', header=0).values;set_mode_stan = set_mode_stan[:, 1:(h + 1)]
set_sd_stan = pd.read_csv('set_sd_trace.csv', header=0).values;set_sd_stan = set_sd_stan[:, 1:(h + 1)]
# print( A1mu_f_stan.shape );plt.plot(A1mu_f_stan)

A1f_stan = pd.read_csv('A1f_trace.csv', header=0).values;A1f_stan = A1f_stan[:, 1:(h + 1)]
B1f_stan = pd.read_csv('B1f_trace.csv', header=0).values;B1f_stan = B1f_stan[:, 1:(h + 1)]
A1s_stan = pd.read_csv('A1s_trace.csv', header=0).values;A1s_stan = A1s_stan[:, 1:(h + 1)]
B1s_stan = pd.read_csv('B1s_trace.csv', header=0).values;B1s_stan = B1s_stan[:, 1:(h + 1)]

A1f_stan.flatten().shape
it_ = [A1f_stan,B1f_stan,A1s_stan,B1s_stan,sep_stan,set_stan,
       A1mu_f_stan,A1s_f_stan,B1mu_f_stan,B1s_f_stan,
       A1mu_s_stan,A1s_s_stan,B1mu_s_stan,B1s_s_stan,
       sep_mode_stan,sep_sd_stan,set_mode_stan,set_sd_stan];
itn_ = ["A1f","B1f","A1s","B1s","sep","set",
        'A1mu_f','A1s_f','B1mu_f','B1s_f',
        'A1mu_s','A1s_s','B1mu_s','B1s_s',
        'sep_mo','sep_sd','set_mo','set_sd']
plt.figure(figsize=(20, 20), dpi=1000);plt.tight_layout();plt.title('Overall Histograms')
for z in range(0, len(it_) ):
    plt.subplot(3, 6, z+1);plt.title(itn_[z])
    plt.hist( it_[z].flatten() )
plt.savefig("overallHistograms.pdf", bbox_inches='tight')

da_ = [A1f_stan.flatten(),B1f_stan.flatten(),A1s_stan.flatten(),B1s_stan.flatten(),sep_stan.flatten(),set_stan.flatten()]
seaborn.pairplot(pd.DataFrame( da_ ), kind='hist')
plt.savefig("paiplot_pdf");plt.show()

plt.hist( A1f_stan.flatten() )
plt.hist( B1f_stan.flatten() )
plt.hist( A1s_stan.flatten() )
plt.hist( B1s_stan.flatten() )
# plt.figure(figsize=(25, 25), dpi=1000);plt.tight_layout();plt.title('A1f_alls')
# for z in range(0,h):
#     plt.subplot(12, 5, z+1);plt.title('ind'+str(z))
#     plt.plot(A1f_stan[:,z])
# plt.savefig("A1f_alls.pdf", bbox_inches='tight')

