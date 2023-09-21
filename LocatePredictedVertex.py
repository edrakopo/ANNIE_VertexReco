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


# infile = "predictionsVertexDRfor10cm1304a.csv"
df = pd.read_csv('FinalpredictionCSV2804.csv')
# infile='FinalpredictionCSV2804.csv'
print(df.shape)
print(df.head())



def find_tank(truevtxx_tank, truevtxy_tank, truevtxz_tank, step, Gridpoint):
    # make grid for 10cm
    i = 1
    gridpoint = {}
    for z in range(30):
        for r in range(40):
            for c in range(30):
                gridpoint[i] = (z, r, c)
                i += 1

       # check if Gridpoint is a valid key in the gridpoint dictionary
    if Gridpoint not in gridpoint:
        print(f"Warning: {Gridpoint} is not a valid grid point")
        return np.nan, np.nan, np.nan

    # find vtx coordinates for given gridpoint
    z, r, c = gridpoint[int(Gridpoint)]
    vtx_z = -140 + z * step
    vtx_y = -190 + r * step
    vtx_x = -140 + c * step

    # print(f"Gridpoint: {Gridpoint}, z: {z}, r: {r}, c: {c}, vtx_x: {vtx_x}, vtx_y: {vtx_y}, vtx_z: {vtx_z}")
    
    
    return vtx_x, vtx_y, vtx_z


# check the length of the DataFrame
print(df.shape)

# apply find_tank function to each row
# new_columns = df['Predicted_Gridpoint'].apply(lambda gp: find_tank(0, 0, 0, 10, gp))
new_columns = df['Reco_Gridpoint'].astype(int).apply(lambda gp: find_tank(0, 0, 0, 10, gp))
# new_columns = df['Predicted_Gridpoint'].astype(int).apply(lambda gp: find_tank(df['truevtxX'], df['truevtxY'], df['truevtxZ'], 10, gp))


# check the length of the new columns
print(len(new_columns))
print(len(new_columns[0]))

# create new columns for vtx_x, vtx_y, and vtx_z
# df[['vtx_x', 'vtx_y', 'vtx_z']] = new_columns
# df[['Predicted_vtx_x', 'Predicted_vtx_y', 'Predicted_vtx_z']] = pd.DataFrame(new_columns.tolist(), index=df.index)
df[['reco_pred_vtx_x', 'reco_pred_vtx_y', 'reco_pred_vtx_z']] = pd.DataFrame(new_columns.tolist(), index=df.index)


# check the length of the DataFrame again
print(df.shape)

print(df.head())

# df.to_csv('CSVwithLocatedVtx2804.csv', index=False)
df.to_csv('CSVwithLocatedRecoVtx2804.csv', index=False)

