######  PySpark ######################
import os
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('Dataframe').getOrCreate()
spark

path = '/home/d/Documents/project/Samples_Grouped60/Bayesian_Together';os.chdir(path);os.getcwd()
df_pyspark = spark.read.option('header','true').csv('Af_trace.csv')
df_pyspark.printSchema()
df_pyspark.columns
df_pyspark.select(['3','16']).show()

sc = spark.sparkContext
Xftrace_RDD = sc.textFile("/home/d/Documents/project/Samples_Grouped60/Bayesian_Together/Xf_trace.txt")
Xftrace_RDD.map(lambda x:len(x)).collect()
Xftrace_RDD.first().shape
Xftrace_RDD.getNumPartitions()

Xftrace_RDD.hist('3')

##### Dask ###########################3
from dask import dataframe  #import the datframe datastructure, behaves mostly like pandas
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')  # use a nicer default styling

paper_dk = dataframe.read_csv('/home/d/Documents/project/Samples_Grouped60/Bayesian_Together/Af_trace.csv',blocksize=4000000)  # point to csv
year_month_group = paper_dk.groupby(['3']).size()  # describe the computation to perform
year_month_group.compute()
roundedSer = paper_dk.groupby(['3']).DataFrame.round(decimals=3).compute()
ym_df = year_month_group.compute()  # actually read and compute the result.

plt.figure(figsize=(12,8))
ym_df.unstack().T.sum().plot()  # Plot publications by year
plt.title('Articles published per year')



# Pool
from multiprocessing import Pool
import time

def f(x):
    time.sleep(2)
    return x**2

for i in range(1,10):
    list_length = i
    t0 = time.time()
    pool = Pool(8)
    result2 = pool.map(f,list(range(list_length)))
    pool.close()
    t1 = time.time()
    print(f"With multiprocessing we ran the function in {t1 - t0:0.4f} seconds")