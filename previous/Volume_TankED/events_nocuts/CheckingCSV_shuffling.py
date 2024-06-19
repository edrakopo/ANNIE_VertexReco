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
#infile = "tankPMT_forVetrexReco_withRecoV.csv"
infile='tankPMT_nocut_insidevolume.csv'

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
hitx, hity, hitz, hitT, totalPMTs, recovertex, labels, gridpoint, nhits, recoVtxFOM, TrueTrackLengthInMRD = np.split(Dataset,[1100,2200,3300,4400,4401,4404,4407,4408,4409,4410],axis=1)
#hits, hitT, nhits, labels, gridpoint, cm, texp, mingridx, mingridy, mingridz = np.split(Dataset, [60, 80,81,84,85,88,108, 128,148 ], axis=1)

print('nhits', nhits," totalPMTs: ",totalPMTs)
print('hitT',hitT)
print("labels: ", labels)
print("gridpoint ", gridpoint)
print("recovertex ",recovertex[0]," labels: ",labels[0])
#checking shape:
print(type(hitT)," len(hitT): ",len(hitT))
print("hitT.shape: ", hitT.shape)
print("hitx.shape: ",hitx.shape)
print("recoVtxFOM ",recoVtxFOM," TrueTrackLengthInMRD ",TrueTrackLengthInMRD)

np.random.shuffle(Dataset) #shuffling the data sample to avoid any bias in the training
df = pd.DataFrame(Dataset)
print(df.head())
df.to_csv("shuffled_NoCuts.csv", float_format = '%.3f')

