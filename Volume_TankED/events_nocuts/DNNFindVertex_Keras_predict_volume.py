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
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint

#--------- File with events for reconstruction:
#--- evts for prediction:
infile = "shuffled_NoCuts.csv"
#

# Set TF random seed to improve reproducibility
seed = 170
np.random.seed(seed)

print( "--- opening file with input variables!")
#--- events for training - MC events
filein = open(str(infile))
print("evts for training in: ",filein)
df = pd.read_csv(filein, index_col=0)
print(df.head())

#processing feautures 
Dataset=np.array(df)
#hits, hitT, nhits, labels, gridpoint, cm, texp, mingridx, mingridy, mingridz = np.split(Dataset, [60, 80,81,84,85,88,108, 128,148 ], axis=1)
hitx, hity, hitz, hitT, totalPMTs, recovertex0, labels, gridpoint, nhits, recoVtxFOM, TrueTrackLengthInMRD = np.split(Dataset,[1100,2200,3300,4400,4401,4404,4407,4408,4409,4410],axis=1)

#split events in train/test samples:
np.random.seed(0)
test_x = np.hstack((hitx[18000:], hity[18000:], hitz[18000:], hitT[18000:]))
test_y = labels[18000:]
recovertex = recovertex0[18000:]
print("test sample features shape: ", test_x.shape," test sample label shape: ", test_y.shape)
print("recovertex[0][0] ", recovertex[0][0]," recovertex[0][1]: ",recovertex[0][1])

def custom_loss_function(y_true, y_pred):
    dist = tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), 1))
    return dist

# create model
model = Sequential()
model.add(Dense(100, input_dim=4400, kernel_initializer='normal', activation='relu'))
model.add(Dense(20, kernel_initializer='normal', activation='relu'))
#model.add(Dense(6, kernel_initializer='normal', activation='relu'))
model.add(Dense(9, kernel_initializer='normal', activation='relu'))
model.add(Dense(3, kernel_initializer='normal', activation='relu'))

# load weights
#print("Created model and loaded weights from file")
#model.load_weights("weights_bets.hdf5")

# Compile model
#model.compile(loss=custom_loss_function, optimizer='ftrl', metrics=custom_loss_function)
model.compile(loss=custom_loss_function, optimizer='Adamax', metrics=['mean_absolute_error'])
print("Created model and loaded weights from file")

# load weights
print("Created model and loaded weights from file")
model.load_weights("weights_bets.hdf5")

## Predict.
print('predicting...')
# Scale data (test set) to 0 mean and unit standard deviation.
#scaler = preprocessing.StandardScaler()
#train_x = scaler.fit_transform(train_x)
#x_transformed = scaler.transform(test_x)
x_transformed = test_x

#make predictions:
y_predicted = model.predict(x_transformed)
print("shapes: ", test_y.shape, ", ", y_predicted.shape)
print("test_y: ",test_y," y_predicted: ",y_predicted)

assert(len(test_y)==len(y_predicted))
assert(len(test_y)==len(recovertex))
DR = np.empty(len(test_y))
DR_reco = np.empty(len(test_y))

import math
#print("DR0 : ", math.sqrt(((y_predicted[0][0] - test_y[0][0])**2 + (y_predicted[0][1] - test_y[0][1])**2 + (y_predicted[0][2] - test_y[0][2])**2)))

for i in range (0,len(y_predicted)):
     DR[i] = math.sqrt(((y_predicted[i][0] - test_y[i][0])**2 + (y_predicted[i][1] - test_y[i][1])**2 + (y_predicted[i][2] - test_y[i][2])**2))
     DR_reco[i] = math.sqrt(((recovertex[i][0] - test_y[i][0])**2 + (recovertex[i][1] - test_y[i][1])**2 + (recovertex[i][2] - test_y[i][2])**2))
     #print("DR: ", DR)

scores = model.evaluate(x_transformed, test_y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]))
print(scores)

# Score with sklearn.
score_sklearn = metrics.mean_squared_error(y_predicted, test_y)
print('MSE (sklearn): {0:f}'.format(score_sklearn))

#-----------------------------
nbins=np.arange(0,400,10)
fig,ax=plt.subplots(ncols=1, sharey=False)#, figsize=(8, 6))
f0=ax.hist(DR, nbins, histtype='step', fill=False, color='blue',alpha=0.75) 
f1=ax.hist(DR_reco, nbins, histtype='step', fill=False, color='red',alpha=0.75)
#ax.set_xlim(0.,200.)
ax.set_xlabel('$\Delta R$ [cm]')
#ax.legend(('NEW','Previous'))
#ax.xaxis.set_ticks(np.arange(0., 500., 50))
ax.tick_params(axis='x', which='minor', bottom=False)
#title = "mean = %.2f, std = %.2f," % (DR.mean(), DR.std())
title = "mean = %.2f, std = %.2f, Prev: mean = %.2f, std = %.2f " % (DR.mean(), DR.std(),DR_reco.mean(), DR_reco.std())
plt.title(title)
plt.show()
fig.savefig('resol_DR.png')
plt.close(fig)

nhits2=nhits[18000:]
recoVtxFOM2 =recoVtxFOM[18000:]
TrueTrackLengthInMRD2 = TrueTrackLengthInMRD[18000:]
data = np.concatenate((test_y, y_predicted,recovertex, nhits2,recoVtxFOM2,TrueTrackLengthInMRD2),axis=1)
print(data)
df = pd.DataFrame(data, columns=['trueX','trueY','trueZ','DNNX','DNNY','DNNZ','recoX','recoY','recoZ','nhits','recoVtxFOM','TrueTrackLengthInMRD'])
print(df.head())
df1 = pd.DataFrame(DR_reco, columns=['DR_reco'])
df2 = pd.DataFrame(DR, columns=['DR'])
df_f = pd.concat((df,df1),axis=1)
df_final = pd.concat((df_f,df2),axis=1)
#df_final = pd.concat((df,df2),axis=1)
print(df_final.head())

print(" saving .csv file with predicted variables..")
df_final.to_csv("predictionsVertexFV.csv", float_format = '%.3f')
