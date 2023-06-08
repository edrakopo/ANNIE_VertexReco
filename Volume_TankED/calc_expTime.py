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
print(df.tail())

#read file with gridpoints vertices:
infile2 = 'gridpoint_coords.csv'
df2 = pd.read_csv(infile2, header = None)
print(df2.head())

hitx_cols = [f'X_{i}' for i in range(1,21)]
hity_cols = [f'Y_{i}' for i in range(1,21)]
hitz_cols = [f'Z_{i}' for i in range(1,21)]
hitT_cols = [f'T_{i}' for i in range(1,21)]

hitx = df[hitx_cols]
hity = df[hity_cols]
hitz = df[hitz_cols]
hitT = df[hitT_cols]

cm = df[['xc', 'yc', 'zc']].values

c = 29.9792458  # in cm/ns

texp_cols = []
for i in range(20):
    s = np.sqrt((hitx.iloc[:, i] - cm[:, 0])**2 + (hity.iloc[:, i] - cm[:, 1])**2 + (hitz.iloc[:, i] - cm[:, 2])**2)
    texp_row = s / c
    texp_cols.append(texp_row)

texp_cols = np.array(texp_cols)

# Add texp as new columns to the DataFrame
for i in range(20):
    df[f'texp_{i+1}'] = texp_cols[i]

df.to_csv('tankPMT_withonlyMRDcut_insidevolume_withTexp.csv', index=False, float_format = '%.3f')
print(df.head())
