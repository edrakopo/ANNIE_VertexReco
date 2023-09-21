import sys
import glob
import numpy as np
import pandas as pd
import tempfile
import random
import csv
from array import array
import math

# infile = '/home/evi/Desktop/ANNIE-THESIS-2/VolumeTank/ANNIE_VertexReco/Volume_Tank/VolumeTank_Texpected1.csv'
infile= '/home/evi/Desktop/ANNIE-THESIS-2/VolumeTank/ANNIE_VertexReco/Volume_Tank/VolumeTank_Texpected2407.csv'
df = pd.read_csv(infile)
print(df.head())

infile2 = '~/Desktop/ANNIE-THESIS-2/VolumeTank/ANNIE_VertexReco/Volume_Tank/gridpoint_coords.csv'
df2 = pd.read_csv(infile2, header=None)
print(df2.head())
coord = np.array(df2)
print('coord[0][:] ', coord[0][:])
print('coordlen', coord.shape)

hitx_cols = [f'X_{i}' for i in range(1, 21)]
hity_cols = [f'Y_{i}' for i in range(1, 21)]
hitz_cols = [f'Z_{i}' for i in range(1, 21)]
hitT_cols = [f'T_{i}' for i in range(1, 21)]

hitx = df[hitx_cols]
hity = df[hity_cols]
hitz = df[hitz_cols]
hitT = df[hitT_cols]

print(hitT.shape)
print('hitT ', hitT.head())

c = 29.9792458  # in cm/ns

def min_Texp(texp_arr, hittime):
    dt = abs(texp_arr - hittime)
    mindtexp = np.amin(dt)  # find min(dt)
    ind = np.argmin(dt)  # return its index
    return ind, mindtexp

def find_nextMin(index, texp_arr):
    new_texp_foreachhit = np.delete(texp_arr, index)
    return new_texp_foreachhit

# df['MinGrid_X'] = np.nan
# df['MinGrid_Y'] = np.nan
# df['MinGrid_Z'] = np.nan
# df['Dist'] = np.nan

for n in range(len(hitx)):
    if (n + 1) % 100 == 0:
        print(f"Processed {n+1} events.")
    event_texp = []
    event_mingrid_X = []
    event_mingrid_Y = []
    event_mingrid_Z = []
    
    for i in range(20):
        texp_foreachhit = []
        for j in range(1000):
            s = np.sqrt((hitx.iloc[n, i] - coord[j, 0])**2 + (hity.iloc[n, i] - coord[j, 1])**2 + (hitz.iloc[n, i] - coord[j, 2])**2)
            texp = s / c
            texp_foreachhit.append(texp)
        
        # Find the index and value of the expected time for each hit (out of 20 hits before median)
        index_DT, mindt = min_Texp(texp_foreachhit, hitT.iloc[n, i])
        min_texp = texp_foreachhit[index_DT]
        mingrid_X = coord[index_DT][0]
        mingrid_Y = coord[index_DT][1]
        mingrid_Z = coord[index_DT][2]
        
        # Store the minimum texp and corresponding grid point coordinates for each hit
        event_texp.append(min_texp)
        event_mingrid_X.append(mingrid_X)
        event_mingrid_Y.append(mingrid_Y)
        event_mingrid_Z.append(mingrid_Z)

        # Calculate distance between (xc, yc, zc) and (MinGrid_X, MinGrid_Y, MinGrid_Z)
        # dist = np.sqrt((df.loc[n, 'xc'] - mingrid_X)**2 + (df.loc[n, 'yc'] - mingrid_Y)**2 + (df.loc[n, 'zc'] - mingrid_Z)**2)
        # df.loc[n, 'Dist'] = dist

    # Append the lists for the current event to the DataFrame
    for i in range(20):
        df.loc[n, f'texp_{i+1}'] = event_texp[i]
    for i in range(20):
        df.loc[n, f'MinGrid_X_{i+1}'] = event_mingrid_X[i]
    for i in range(20):
        df.loc[n, f'MinGrid_Y_{i+1}'] = event_mingrid_Y[i]
    for i in range(20):
        df.loc[n, f'MinGrid_Z_{i+1}'] = event_mingrid_Z[i]

df.to_csv('tankPMT_withonlyMRDcut_insidevolume_withTexp2407.csv', index=False, float_format='%.3f')
print(df.head())
print("Saved the modified data to tankPMT_withonlyMRDcut_insidevolume_withTexp1907.csv.")
