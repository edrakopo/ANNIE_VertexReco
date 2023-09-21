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
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import ModelCheckpoint

#--------- File with events for reconstruction:
#--- evts for prediction:
#infile = "tankPMT_forVetrexReco.csv"
#infile = "tankPMT_forVetrexReco_withRecoV.csv"
infile = "10cmRecoGridpoint_shuffled.csv"

#

# Set TF random seed to improve reproducibility
seed = 170
np.random.seed(seed)


print( "--- opening file with input variables!")
#--- events for training - MC events
filein = open(str(infile))
print("evts for training in: ",filein)
Dataset=np.array(pd.read_csv(filein))
# features, rest, recovertex, labels, gridpoint, gridpointpmt = np.split(Dataset,[5500,5502,5505,5508,5509],axis=1)
# features, rest, recovertex, labels, gridpoint, gridpointreco, gridpointpmt = np.split(Dataset,[5500,5502,5505,5508,5509, 5510],axis=1)

features, rest1, rest2, recovertex, labels, gridpoint, gridpointreco, gridpointpmt = np.split(Dataset,[4400,5500,5502,5505,5508,5509, 5510],axis=1)

print("rest1 :", rest1)
print("rest2 :", rest2[0])
# print("rest :", rest)
print("features: ",features)
print("recovertex: ",recovertex)
print("labels: ", labels)
print("gridpoint ", gridpoint)
print("gridpoint reco", gridpointreco)
print("gridpoint pmt", gridpointpmt)
#split events in train/test samples:
num_events, num_pixels = features.shape
print(num_events, num_pixels)
np.random.seed(0)
train_x = features[:2000]
train_y = gridpoint[:2000]
test_x = features[2000:]
test_y = gridpoint[2000:]
recoVtx_y = recovertex[2000:]
trueVtx= labels[2000:]
gridpointRECO= gridpointreco[2000:]
print("test sample features shape: ", test_x.shape," test sample label shape: ", test_y.shape)

# create model
model = Sequential()
model.add(Dense(50, input_dim=4400, kernel_initializer='normal', activation='relu'))
#model.add(Dense(50, kernel_initializer='normal', activation='relu'))
model.add(Dense(20, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal', activation='relu'))
#model.add(Dense(25, input_dim=4400, kernel_initializer='normal', activation='relu'))
#model.add(Dense(25, kernel_initializer='normal', activation='relu'))
#model.add(Dense(3, kernel_initializer='normal', activation='relu'))

# load weights
model.load_weights("weights25_bets.hdf5")

# Compile model
model.compile(loss='mean_squared_error', optimizer='Adamax', metrics=['accuracy'])
print("Created model and loaded weights from file")

## Predict.
print('predicting...')
# Scale data (test set) to 0 mean and unit standard deviation.
scaler = preprocessing.StandardScaler()
train_x = scaler.fit_transform(train_x)
x_transformed = scaler.transform(test_x)
y_predicted = model.predict(x_transformed)
print("shapes: ", test_y.shape, ", ", y_predicted.shape)
print("test_y: ",test_y,"\n y_predicted: ",y_predicted)

assert(len(test_y)==len(y_predicted))
assert(len(test_y)==len(recoVtx_y))
assert(len(test_y)==len(trueVtx))
DR = np.empty(len(test_y))
# DR_reco = np.empty(len(test_y))
# DR_true = np.empty(len(test_y))

import math
# print("DR0 : ", math.sqrt(((y_predicted[0][0] - test_y[0][0])**2 + (y_predicted[0][1] - test_y[0][1])**2 + (y_predicted[0][2] - test_y[0][2])**2)))


# for i in range (0,len(y_predicted)):
#      DR[i] = math.sqrt(((y_predicted[i][0] - test_y[i][0])**2 + (y_predicted[i][1] - test_y[i][1])**2 + (y_predicted[i][2] - test_y[i][2])**2))
#      DR_reco[i] = math.sqrt(((recoVtx_y[i][0] - test_y[i][0])**2 + (recoVtx_y[i][1] - test_y[i][1])**2 + (recoVtx_y[i][2] - test_y[i][2])**2))


for i in range (0,len(y_predicted)):
    
     DR[i] = math.sqrt(((recoVtx_y[i][0] - trueVtx[i][0])**2 + (recoVtx_y[i][1] - trueVtx[i][1])**2 + (recoVtx_y[i][2] - trueVtx[i][2])**2))
    #  print("DR: ", DR)
 
# print(DR[i])

scores = model.evaluate(x_transformed, test_y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]))
print(scores)

# Score with sklearn.
score_sklearn = metrics.mean_squared_error(y_predicted, test_y)
print('MSE (sklearn): {0:f}'.format(score_sklearn))

#------------------------------------------------------------------------

print(" saving .csv file with predicted variables..")

#data = np.concatenate((test_y[0], test_y[1], test_y[2], y_predicted[0], y_predicted[1], y_predicted[2]),axis=1)
# data = np.concatenate((test_y, y_predicted,recoVtx_y),axis=1)

data = np.concatenate((test_y, y_predicted, gridpointRECO),axis=1)
print(data)
df = pd.DataFrame(data, columns=['Gridpoint','Predicted_Gridpoint', 'Reco_Gridpoint'])
print(df.head())
df1 = pd.DataFrame(DR, columns=['DR'])
df_f = pd.concat((df,df1),axis=1)
print(df_f.head())


df_true= pd.DataFrame(trueVtx, columns=['truevtxX', 'truevtxY', 'truevtxZ'])
df_reco= pd.DataFrame(recoVtx_y, columns=['recovtxX', 'recovtxY', 'recovtxZ'])
print(df_true.head())
print(df_reco.head())
df_ff=pd.concat((df_true,df_reco),axis=1)
df_final = pd.concat((df_ff,df_f),axis=1)
print(df_final.head())
# df1 = pd.DataFrame(DR_reco, columns=['DR_reco'])
# df2 = pd.DataFrame(DR_true, columns=['DR_true'])
# df_f = pd.concat((df,df1),axis=1)
# df_final = pd.concat((df_f,df2),axis=1)
# print(df_final.head())
df_f.to_csv("predictionsVertexDRfor10cm2804.csv", float_format = '%.3f', index=False)
df_final.to_csv("FinalpredictionCSV2804.csv", float_format = '%.3f', index=False)
