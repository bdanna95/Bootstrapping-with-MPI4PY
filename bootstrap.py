#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
from mpi4py import MPI
from sklearn.linear_model import LinearRegression
import math


# In[11]:


def B_chunks(B,p):
    B_chunks = B/p
    return int(B_chunks)


# In[12]:


def boot_chunk(chunk_size,df):
    #store final output as list
    listout = []
    #create empty dataframe with column names to store coefficients for each independent variable
    dfbeta = pd.DataFrame(columns=['beta1','beta2','beta3','beta4','beta5'])
    #loop creating random sample from dataframe with replacement
    
    for chunk in range(0,chunk_size):
        dfsample = df.sample(frac=1, replace=True)
        #create linear regression model of y = x1 + x2 + x3 + x4 + x5
        model = LinearRegression().fit(dfsample.loc[:, df.columns != 'y'], dfsample['y'])
        #append coefficients into coefficient dataframe
        dfbeta.loc[len(dfbeta)] = model.coef_.tolist()
        
    #get sample mean of each beta estimate
    dfbetabar = dfbeta.mean()
    #get sample std of each beta estimate
    dfbetastd = dfbeta.std()
    #get sum of squared beta estimates for each beta
    dfbeta2 = (dfbeta**2).sum()
    #concatenate into a df
    dfsumstat = pd.concat([dfbetabar,dfbetastd,dfbeta2],axis=1).reset_index().rename(columns={"index": "beta", 0: "betabar", 1: "betastd", 2: "beta2"})
    
    for i in range(len(dfsumstat)):
        listout.append(dfsumstat.loc[i,:].values.flatten().tolist())
    print(listout)
    return(listout)


# In[13]:


#CI calculation

def CI95Calc(betanum,df,n,z_score):
    betaCI = [betanum]
    betaCI.append((df[df['beta'] == betanum]['betabar'].mean())-(z_score * (math.sqrt(((df[df['beta'] == betanum]['beta2'].sum())-
      (n*(df[df['beta'] == betanum]['betabar'].mean())**2))/(n-1))/math.sqrt(n))))
    betaCI.append((df[df['beta'] == betanum]['betabar'].mean())+(z_score * (math.sqrt(((df[df['beta'] == betanum]['beta2'].sum())-
      (n*(df[df['beta'] == betanum]['betabar'].mean())**2))/(n-1))/math.sqrt(n))))
    return(betaCI)


# In[14]:


def run():
    #set mpi variables
    comm = MPI.COMM_WORLD
    myrank = comm.rank
    p = comm.size
    
    #set parameters for bootstrap iterations and alpha
    n = 10000
    z_score = 1.96
    filename = "/Users/brandondanna/Documents/AMS 598/Projects/Project2/project2_data.csv"
    
    if myrank == 0:
        df = pd.read_csv(filename)
    else:
        None
        
    comm.bcast(df,root=0)
    
    #run the bootstrap function on each processor for an equal chunk of the total and gather on the root
    mylist = comm.gather(boot_chunk(B_chunks(n,p),df),root=0)
    
    #ensure all iterations are complete before compiling results
    comm.Barrier()
    
    #on root node, read all lists into dataframe and use CI function to create confidence intervals and print
    if myrank == 0:
        myflatlist = [item for sublist in mylist for item in sublist]
        df = pd.DataFrame(columns=['beta','betabar','betastd','beta2'],data=myflatlist)
        beta1CI = CI95Calc('beta1',df,n,z_score)
        beta2CI = CI95Calc('beta2',df,n,z_score)
        beta3CI = CI95Calc('beta3',df,n,z_score)
        beta4CI = CI95Calc('beta4',df,n,z_score)
        beta5CI = CI95Calc('beta5',df,n,z_score)
        CITotal = [beta1CI,beta2CI,beta3CI,beta4CI,beta5CI]
        print([CITotal])
    else: None
        


# In[15]:


if __name__ == "__main__":
    run();


# In[ ]:




