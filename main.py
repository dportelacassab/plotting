import pandas as pd
if __name__ ==  '__main__':
    import time
    import os
    from multiprocessing import Pool
    import time
    # import numpy as np
    # import pandas as pd


    def f(x):
        return x ** 2

    h = 60
    path = '/home/d/Documents/project/Samples_Grouped60/Bayesian_Together';os.chdir(path);os.getcwd()

    Af_stan = pd.read_csv('Af_trace.csv', header=0).values;
    Af_stan = Af_stan[:, 1:(h + 1)]
    Bf_stan = pd.read_csv('Bf_trace.csv', header=0).values;
    Bf_stan = Bf_stan[:, 1:(h + 1)]
    As_stan = pd.read_csv('As_trace.csv', header=0).values;
    As_stan = As_stan[:, 1:(h + 1)]
    Bs_stan = pd.read_csv('Bs_trace.csv', header=0).values;
    Bs_stan = Bs_stan[:, 1:(h + 1)]
    sep_stan = pd.read_csv('sep_trace.csv', header=0).values;
    sep_stan = sep_stan[:, 1:(h + 1)]
    set_stan = pd.read_csv('set_trace.csv', header=0).values;
    set_stan = set_stan[:, 1:(h + 1)]

    col = [Af_stan, Bf_stan, As_stan, Bs_stan, sep_stan, set_stan]

    path = '/home/d/Documents/project/';os.chdir(path);os.getcwd();import aus_newPerLik

    for i in len(col):
        list_length = i
        t0 = time.time()
        pool = Pool(8)
        result2 = pool.map( aus_newPerLik.mode_estimation, pd.DataFrame(np.around(col[i],3)) )
        print(result2+'\n')
        pool.close()
        t1 = time.time()
        print(f"With multiprocessing we ran the function in {t1 - t0:0.4f} seconds")
