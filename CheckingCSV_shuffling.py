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
assert(len(hitz)==len(hitT))
assert(len(hitx)==len(hitz))
assert(len(hitx)==len(hity))
'''
#Find median timevalue for each event: 
t = hitT[hitT != 0]
for i in range(0,2):
    print(np.median(t[i]))

#med = lambda hits: [np.median(t[i]) for i in range(0,2)]
med = lambda hits: [np.median(t[i]) for i in range(0,5)]
print(med(t))
median_index = np.where(t==8.21572) #med(t[0])[0])
print(median_index)
#median_hitT = lambda medianT: [np.median(t) for t in hitT]
#print("median_hitT: ",median_hitT(t[0]))
'''


np.random.shuffle(Dataset) #shuffling the data sample to avoid any bias in the training
df = pd.DataFrame(Dataset)
print(df.head())
df.to_csv("shuffledevents_Timebeforemedian.csv", float_format = '%.3f')

