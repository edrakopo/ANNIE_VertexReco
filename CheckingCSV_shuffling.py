import sys
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import tempfile
import random
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from array import array
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

#--------- File with events for reconstruction:
#infile = "tankPMT_forVetrexReco.csv"
infile = "tankPMT_forVetrexReco_withRecoV.csv"

# Set TF random seed to improve reproducibility
#seed = 170
#np.random.seed(seed)

print( "--- opening file with input variables!")
#--- events for training - MC events
filein = open(str(infile))
print("evts for training in: ",filein)
#print((pd.read_csv(filein)).head())
Dataset=np.array(pd.read_csv(filein))

#features, rest, recovertex, labels, gridpoint = np.split(Dataset,[4400,4402,4405,4408],axis=1)
hitx, hity, hitz, hitT, totalPMTs, totalLAPPDs, recovertex, labels, gridpoint = np.split(Dataset,[1100,2200,3300,4400,4401,4402,4405,4408],axis=1)
print("hitz: ",hitz[0])
print("hitT: ",hitT[0])
print("recovertex: ", recovertex[0])
print("labels: ", labels)
print("totalPMTs:", totalPMTs[0]," min: ", np.amin(totalPMTs)," with index:",np.argmin(totalPMTs) ," max: ",np.amax(totalPMTs))
print("np.mean(totalPMTs): ",np.mean(totalPMTs)," np.median(totalPMTs): ",np.median(totalPMTs))

#checking shape:
print(type(hitT)," len(hitT): ",len(hitT))
print("hitT.shape: ", hitT.shape)
print("hitx.shape: ",hitx.shape)
#print(hitT)
assert(len(hitx)==len(hitT))

#for one event try to convert to  df: 
def hits_medianT(dt,dx,dy,dz): 
    d00 = pd.concat([dt, dx, dy, dz],axis=1)
    d0 = d00[d00['T'] != 0]
    print(d0)
    #calculate median and its index: 
    medianT = d0['T'].median()
    posmedian = d0.loc[d0['T']==d0['T'].median()]
    print("posmedian: ",posmedian," with value: ",medianT)

    d0.sort_values(by='T', inplace=True)
    print("d0 after sorting: \n", d0)
    pos1 = d0[d0['T'] > d0['T'].median()].iloc[0]
    pos2 = d0[d0['T'] < d0['T'].median()].iloc[-1]
    print("pos1: ", pos1," pos2: ", pos2)

    df_short = d0[d0['T']<medianT]
    print(df_short)
    #select the 20 first hits: 
    df_ToUse = df_short[:20]
    #df_ToUse = df_short.sample(n=20) #randomly select 20 hits
    print(df_ToUse)
    return df_ToUse


print("hitT[i][:].shape: ",hitT[0][:].shape)
Dt = pd.DataFrame(hitT[0][:], columns = ['T'])
Dx = pd.DataFrame(hitx[0][:], columns = ['X'])
Dy = pd.DataFrame(hity[0][:], columns = ['Y'])
Dz = pd.DataFrame(hitz[0][:], columns = ['Z'])
print(hits_medianT(Dt, Dx, Dy, Dz))

#call the function for all events:



#np.random.shuffle(Dataset) #shuffling the data sample to avoid any bias in the training
df = pd.DataFrame(Dataset)
#print(df.head())
#df.to_csv("shuffledevents_Timebeforemedian.csv", float_format = '%.3f')

