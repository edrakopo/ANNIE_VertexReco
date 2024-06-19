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


df = pd.read_csv('tankPMT_withonlyMRDcut_insidevolume.csv')

print(df.shape)
print(df.head())



def find_tank(truevtxx_tank, truevtxy_tank, truevtxz_tank, step, Gridpoint):
    # make grid for 10cm, in volume 1x1m
    i = 1
    gridpoint = {}
    for z in range(10):
        for r in range(10):
            for c in range(10):
                gridpoint[i] = (z, r, c)
                i += 1

       # check if Gridpoint is a valid key in the gridpoint dictionary
    if Gridpoint not in gridpoint:
        print(f"Warning: {Gridpoint} is not a valid grid point")
        return np.nan, np.nan, np.nan

    # find vtx coordinates for given gridpoint
    z, r, c = gridpoint[int(Gridpoint)]
    vtx_z = -50 + z * step +5
    vtx_y = -50 + r * step +5
    vtx_x = -50 + c * step +5

    # print(f"Gridpoint: {Gridpoint}, z: {z}, r: {r}, c: {c}, vtx_x: {vtx_x}, vtx_y: {vtx_y}, vtx_z: {vtx_z}")
    
    
    return vtx_x, vtx_y, vtx_z


# check the length of the DataFrame
print(df.shape)

# apply find_tank function to each row
new_columns = df['Gridpoint'].astype(int).apply(lambda gp: find_tank(0, 0, 0, 10, gp))
# new_columns = df['Predicted_Gridpoint'].astype(int).apply(lambda gp: find_tank(df['truevtxX'], df['truevtxY'], df['truevtxZ'], 10, gp))


# check the length of the new columns
print(len(new_columns))
print(len(new_columns[0]))

# create new columns for vtx_x, vtx_y, and vtx_z
# df[['vtx_x', 'vtx_y', 'vtx_z']] = new_columns
df[['xc', 'yc', 'zc']] = pd.DataFrame(new_columns.tolist(), index=df.index)


# check the length of the DataFrame again
print(df.shape)

print(df.head())

df.to_csv('tankPMT_withonlyMRDcut_insidevolume2.csv', index=False)

