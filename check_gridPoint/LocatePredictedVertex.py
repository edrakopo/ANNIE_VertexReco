import sys
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import tempfile
import random
import csv
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from array import array

infile= '/Users/edrakopo/Desktop/Evi_scripts/FinalpredictionCSV2804.csv'
filein = open(str(infile))
df = pd.read_csv(filein)
print(df.head())

# filein = open(str(infile))
# print("evts for training in: ",filein)

# Dataset=np.array(pd.read_csv(filein))
# vtx,grid,predicted_grid,rest=np.split(Dataset, [2,3,4], axis=1)

import pandas as pd

def find_tank(truevtxx_tank, truevtxy_tank, truevtxz_tank, step, Gridpoint):
    # make grid for 10cm
    i = 1
    gridpoint = {}
    for z in range(30):
        for r in range(40):
            for c in range(30):
                gridpoint[i] = (z, r, c)
                i += 1

    # find vtx coordinates for given gridpoint
    if Gridpoint<=36000:
       z, r, c = gridpoint[int(Gridpoint)]
       vtx_z = -140 + z * step
       vtx_y = -190 + r * step
       vtx_x = -140 + c * step
    else: 
       vtx_z = -999
       vtx_y = -999
       vtx_x = -999

    return vtx_x, vtx_y, vtx_z


# check the length of the DataFrame
print(df.shape)
#print(df.describe())

# apply find_tank function to each row
#print("Max gridpoint:",df['Gridpoint'].max())
new_columns = df['Predicted_Gridpoint'].apply(lambda gp: find_tank(0, 0, 0, 10, gp))

print("checking vertex for gridpoint 15044 : ", find_tank(0, 0, 0, 10, 15044.44))
print("checking vertex for gridpoint 15078 : ", find_tank(0, 0, 0, 10, 15078))
print("checking vertex for gridpoint 15079 : ", find_tank(0, 0, 0, 10, 15079))

# check the length of the new columns
print(len(new_columns))
print(len(new_columns[0]))

# create new columns for vtx_x, vtx_y, and vtx_z
# df[['vtx_x', 'vtx_y', 'vtx_z']] = new_columns
df[['vtx_x', 'vtx_y', 'vtx_z']] = pd.DataFrame(new_columns.tolist(), index=df.index)

#calculate DR:
import math
DR_DNN = np.empty(len(np.array(df['Gridpoint'])))
DR_reco = np.empty(len(np.array(df['Gridpoint'])))
 
for i in range (0,len(np.array(df['Gridpoint']))):
    DR_reco[i] = math.sqrt(((df['truevtxX'][i] - df['recovtxX'][i])**2 + (df['truevtxY'][i] - df['recovtxY'][i])**2 + (df['truevtxZ'][i] - df['recovtxZ'][i])**2))
    DR_DNN[i] = math.sqrt(((df['truevtxX'][i] - df['vtx_x'][i])**2 + (df['truevtxY'][i] - df['vtx_y'][i])**2 + (df['truevtxZ'][i] - df['vtx_z'][i])**2))

df1 = pd.DataFrame(DR_reco, columns=['DR_reco'])
df2 = pd.DataFrame(DR_DNN, columns=['DR_DNN'])
df_f = pd.concat((df,df1),axis=1)
df_final = pd.concat((df_f,df2),axis=1)

# check the length of the DataFrame again
print(df_final.shape)
print(df_final.head())
#check the DR,DR_DNN again: 
print("max value for reco DR: ",df_final['DR_reco'].max(),",",df_final['DR'].max())
print("max value for DNN DR: ",df_final['DR_DNN'].max())
df_final.to_csv("predictionsWithGridPoint.csv", float_format = '%.3f')


#------------------------------------------------------
#plot DR:
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (9, 7),
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

data = df_final['DR_DNN']
dataprev = df_final['DR_reco']
#print("data ",data)
#print("dataprev ",dataprev)
nbins=np.arange(0,400,4)
fig,ax=plt.subplots(ncols=1, sharey=False)#, figsize=(8, 6))
f0=ax.hist(data, nbins, histtype='step', fill=False, color='blue',alpha=0.75)
f1=ax.hist(dataprev, nbins, histtype='step', fill=False, color='red',alpha=0.75)
#ax.set_xlim(0.,200.)
ax.set_xlabel('$\Delta R$ [cm]')
ax.legend(('DNN','Prev.Reco'))
#ax.xaxis.set_ticks(np.arange(0., 500., 50))
ax.tick_params(axis='x', which='minor', bottom=False)
title = "mean = %.2f, std = %.2f, Prev: mean = %.2f, std = %.2f " % (data.mean(), data.std(),dataprev.mean(), dataprev.std())
plt.title(title)
plt.show()
fig.savefig('GridPoint_resol_DR.png')
plt.close(fig)

#plot Gridpoint difference histos:
data0 = df_final['Gridpoint']-df['Predicted_Gridpoint']
dataprev0 = df_final['Gridpoint']-df['Reco_Gridpoint']
print("data0 ",data0)
print("dataprev0 ",dataprev0)
nbins=np.arange(0,1000,100)
fig,ax=plt.subplots(ncols=1, sharey=False)#, figsize=(8, 6))
f0=ax.hist(data0, nbins, histtype='step', fill=False, color='blue',alpha=0.75)
f1=ax.hist(dataprev0, nbins, histtype='step', fill=False, color='red',alpha=0.75)
#ax.set_xlim(0.,200.)
ax.set_xlabel('grispoint Difference')
ax.legend(('DNN-True gridpoint','Reco-True gridpoint'))
#ax.xaxis.set_ticks(np.arange(0., 500., 50))
ax.tick_params(axis='x', which='minor', bottom=False)
title = "mean = %.2f, std = %.2f, Prev: mean = %.2f, std = %.2f " % (data0.mean(), data0.std(),dataprev0.mean(), dataprev0.std())
plt.title(title)
plt.show()
fig.savefig('GridPoint_resol.png')
plt.close(fig)

#plot true vs DNN predicted gridpoint:
import ROOT
canvas = ROOT.TCanvas()
canvas.cd(1)
th2f = ROOT.TH2F("DNN_GridPoint", "; GridPoint; DNN GridPoint prediction", 150, 5000, 20000., 150, 5000., 20000.)
for i in range(len(df['Gridpoint'])):
    th2f.Fill(df['Gridpoint'][i], df['Predicted_Gridpoint'][i])
#line = ROOT.TLine(20000.,20000.,20000.,20000.)
th2f.SetStats(0)
th2f.Draw("ColZ")
#line.SetLineColor(2)
canvas.Draw()
#line.Draw("same")
canvas.SaveAs("DNN_true_Gridpoint.png")


#plot true vs reco gridpoint:
import ROOT
canvas = ROOT.TCanvas()
canvas.cd(1)
th2f = ROOT.TH2F("Reco_GridPoint", "; GridPoint; GridPoint from Reco.vertex", 150, 5000, 20000., 150, 5000., 20000.)
for i in range(len(df['Gridpoint'])):
    th2f.Fill(df['Gridpoint'][i], df['Reco_Gridpoint'][i])
#line = ROOT.TLine(20000.,20000.,20000.,20000.)
th2f.SetStats(0)
th2f.Draw("ColZ")
#line.SetLineColor(2)
canvas.Draw()
#line.Draw("same")
canvas.SaveAs("reco_true_Gridpoint.png")
