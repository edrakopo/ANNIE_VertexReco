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
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.utils import shuffle

#--------- File with events for reconstruction:
#--- evts for training:
#infile = "tankPMT_forVetrexReco_withRecoV.csv" #unshuffled 
infile = "shuffled.csv"                        #shuffled

# Set TF random seed to improve reproducibility
seed = 170
np.random.seed(seed)

print( "--- opening file with input variables!")
#--- events for training - MC events
filein = open(str(infile))
print("evts for training in: ",filein)
#print((pd.read_csv(filein)).head())
Dataset=np.array(pd.read_csv(filein))

#features, rest, recovertex, labels, gridpoint, gridpointpmt = np.split(Dataset,[5500,5502,5505,5508,5509],axis=1)
features, rest, recovertex, labels, gridpoint = np.split(Dataset,[4400,4402,4405,4408],axis=1)
print("rest :", rest[0])
print("features: ",features[0])
print("recovertex: ", recovertex)
print("labels: ", labels)
print("gridpoint ", gridpoint)
# np.random.shuffle(Dataset) #shuffling the data sample to avoid any bias in the training
# df = pd.DataFrame(Dataset)
# print(df.head())



