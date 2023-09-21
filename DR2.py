import sys
import glob
import numpy as np
import pandas as pd
import tempfile
import random
import csv
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from array import array

infile = 'CSVwithLocatedVtx2804.csv'
# infile = 'CSVwithLocatedRecoVtx2804.csv'


print( "--- opening file with input variables!")

filein = open(str(infile), "r")
print("evts in: ",filein)
Dataset=pd.read_csv(filein)

truevtx, rest, predictedVtx = np.split(Dataset,[3,6],axis=1)

print("rest1 :", rest)
print("truevtx:", truevtx)
print("predicted:", predictedVtx)


DR2 = np.empty(len(Dataset))

import math


# Split the dataset into truevtx and predictedvtx arrays
truevtx = Dataset[['truevtxX', 'truevtxY', 'truevtxZ']].values
predictedvtx = Dataset[['Predicted_vtx_x', 'Predicted_vtx_y', 'Predicted_vtx_z']].values

# Calculate the Euclidean distance between the true and predicted vertices
DR2 = np.sqrt(np.sum((predictedvtx - truevtx)**2, axis=1))
print('DR2', DR2)

# Create a new DataFrame that contains the predictedvtx, truevtx, and DR2 arrays
output_df = pd.DataFrame({'predictedVtx_x': predictedvtx[:, 0],
                          'predictedVtx_y': predictedvtx[:, 1],
                          'predictedVtx_z': predictedvtx[:, 2],
                          'truevtxX': truevtx[:, 0],
                          'truevtxY': truevtx[:, 1],
                          'truevtxZ': truevtx[:, 2],
                          'DR2': DR2})

print('output', output_df.head())

# Write the output DataFrame to a CSV file
output_df.to_csv('output2504.csv', index=False)





#ADD DR2 TO EXISTING CSV FILE AND PLOT




# Read the first CSV file
df1 = pd.read_csv('CSVwithLocatedVtx2504.csv')


# Read the second CSV file and extract the DR2 column
df2 = pd.read_csv('output2504.csv')
dr2_col = df2['DR2']

# Add the DR2 column to the first CSV file
df1['DR2'] = dr2_col

# Save the updated first CSV file
# df1.to_csv('plotDR.csv', index=False)

print(df1.head)


data = df1['DR'] 
dataprev = df1['DR2'] 
nbins=np.arange(-50,200,5)
fig,ax=plt.subplots(ncols=1, sharey=False)#, figsize=(8, 6))
f0=ax.hist(data, nbins, histtype='step', fill=False, color='blue',alpha=0.75, label='DR') 
f1=ax.hist(dataprev, nbins, histtype='step', fill=False, color='red',alpha=0.75, label='DR2')
#ax.set_xlim(0.,200.)
ax.set_xlabel('Distance [cm]')
ax.set_ylabel('Events')
# ax.set_yscale('log')
ax.legend( loc='best')
#ax.xaxis.set_ticks(np.arange(0., 500., 50))
ax.tick_params(axis='x', which='minor', bottom=False)
plt.show()
# fig.savefig('DR_DR23.png')
plt.close(fig)