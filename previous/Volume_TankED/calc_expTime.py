import sys
import glob
import numpy as np
import pandas as pd
import tempfile
import random
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from array import array
import math

infile ='VolumeTank_Texpected.csv'
df = pd.read_csv(infile)
print(df.head())
#print(df.tail())

# Read file with gridpoints vertices to calculate the expected hit time #
# Each row corresponds to the gridpoint number:
infile2 = 'gridpoint_coords.csv'
df2 = pd.read_csv(infile2, header = None)
print(df2.head())
coord = np.array(df2)
print('coord[0][:] ',coord[0][:])

hitx_cols = [f'X_{i}' for i in range(1,21)]
hity_cols = [f'Y_{i}' for i in range(1,21)]
hitz_cols = [f'Z_{i}' for i in range(1,21)]
hitT_cols = [f'T_{i}' for i in range(1,21)]

hitx = df[hitx_cols]
hity = df[hity_cols]
hitz = df[hitz_cols]
hitT = df[hitT_cols]

print(hitx.shape)
print('hitx ',hitx.head())
#cm = df[['xc', 'yc', 'zc']].values

c = 29.9792458  # in cm/ns

#function to select the index for the gridpoint with minimum dt for each hit:
def min_Texp(texp_arr, hittime):
    dt = abs(texp_arr-hittime)
    mintexp = np.amin(dt) #find min(dt)
    ind = np.argmin(dt)  #return its index
    return ind

# for each hit calculate the expected hit time from each grid point:
texp_foreachhit = []
hit_exp = []
MinGridpoint = []
for n in range(0, len(hitx)):
    for i in range(20):
        for j in range(1000):
            s = np.sqrt((hitx.iloc[n, i] - coord[j, 0])**2 + (hity.iloc[n, i] - coord[j, 1])**2 + (hitz.iloc[n, i] - coord[j, 2])**2)
            texp = s / c   #calculates the expected time for each hit for 1000 gridpoints
            texp_foreachhit.append(texp)
        index_DT = min_Texp(texp_foreachhit, hitT.iloc[n, i])
        #get the "winning" texpected with the index: index_DT from texp_foreachhit and store its coords. 
        MinGridpoint[i] = coord[index_DT]
            #check first element:
            if j==0:
               print("i: ",i," hit: ",hitx.iloc[n, i],",",hity.iloc[n, i],",",hitz.iloc[n, i])
               print("j ",j," coords: ",coord[j, 0],",",coord[j, 1],",",coord[j, 2]," s ", s," texp_row: ",texp_row, '\n')

#texp_cols = []
#for i in range(20):
#    s = np.sqrt((hitx.iloc[:, i] - coord[:, 0])**2 + (hity.iloc[:, i] - coord[:, 1])**2 + (hitz.iloc[:, i] - coord[:, 2])**2)
#    texp_row = s / c
#    texp_cols.append(texp_row)

#texp_cols = np.array(texp_cols)

# Add texp as new columns to the DataFrame
for i in range(20):
    df[f'texp_{i+1}'] = texp_cols[i]

df.to_csv('tankPMT_withonlyMRDcut_insidevolume_withTexp.csv', index=False, float_format = '%.3f')
print(df.head())
