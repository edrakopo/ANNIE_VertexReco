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
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from numpy import asarray
from sklearn.preprocessing import MinMaxScaler


#--------- File with events for reconstruction:
#--- evts for prediction:
#infile = "tankPMT_forVetrexReco.csv"
#infile = "tankPMT_forVetrexReco_withRecoV.csv"
#infile = "tankPMT_forVetrexReco_withRecoV.csv"
# infile = "shuffledtankPMT_forVetrexReco_withRecoV.csv"
# infile = "dataBootcampShuffle.csv"
# infile = "/home/evi/Desktop/ANNIE-THESIS/10cmRecoGridpoint_shuffled.csv"
# infile= '/home/evi/Desktop/ANNIE-THESIS/300hits_shuffled.csv'
infile='shuffledevents_Timebeforemedian.csv'


#

# Set TF random seed to improve reproducibility
seed = 170
np.random.seed(seed)


print( "--- opening file with input variables!")
#--- events for training - MC events
filein = open(str(infile))
print("evts for training in: ",filein)
Dataset=np.array(pd.read_csv(filein))
# define min max scaler
scaler = MinMaxScaler()
# transform data
scaled = scaler.fit_transform(Dataset)
print(scaled)
# features, recovertex, labels= np.split(Dataset,[4400,4403],axis=1)
# features, rest1, nhits0, rest2, recovertex, labels, gridpoint, gridpointreco, gridpointpmt = np.split(Dataset,[4400,5500,5501,5502,5505,5508,5509, 5510],axis=1)

# features, nhits0, rest2, recovertex, labels, gridpoint, gridpointreco = np.split(Dataset,[1200,1201,1202,1205,1208,1209],axis=1)
#with 20 hits
features, nhits0, rest2, recovertex, labels, gridpoint, gridpointreco = np.split(Dataset,[80,81,82,85,88,89],axis=1)


print("features: ",features)
print("recovertex: ",recovertex)
print('nhits', nhits0)
print("labels: ", labels)
#split events in train/test samples:
num_events, num_pixels = features.shape
print(num_events, num_pixels)
np.random.seed(0)
train_x = features[:2000]
train_y = labels[:2000]
test_x = features[2000:]
test_y = labels[2000:]
recoVtx_y = recovertex[2000:]
nhits=nhits0[2000:]
print(test_x.shape)
print("test sample features shape: ", test_x.shape," test sample label shape: ", test_y.shape)

# create model
model = Sequential()
model.add(Dense(60, input_dim=80, kernel_initializer='normal', activation='relu'))
# model.add(Dense(50, kernel_initializer='normal', activation='relu'))
model.add(Dense(45, kernel_initializer='normal', activation='relu'))
model.add(Dense(30, kernel_initializer='normal', activation='relu'))
model.add(Dense(10, kernel_initializer='normal', activation='relu'))
model.add(Dense(3, kernel_initializer='normal', activation='relu'))

#model.add(Dense(25, input_dim=4400, kernel_initializer='normal', activation='relu'))
#model.add(Dense(25, kernel_initializer='normal', activation='relu'))
#model.add(Dense(3, kernel_initializer='normal', activation='relu'))

# load weights
model.load_weights("weights_bets20.hdf5")

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
print("test_y: ",test_y," y_predicted: ",y_predicted)

assert(len(test_y)==len(y_predicted))
assert(len(test_y)==len(recoVtx_y))
DR = np.empty(len(test_y))
DR_reco = np.empty(len(test_y))
import math
print("DR0 : ", math.sqrt(((y_predicted[0][0] - test_y[0][0])**2 + (y_predicted[0][1] - test_y[0][1])**2 + (y_predicted[0][2] - test_y[0][2])**2)))
for i in range (0,len(y_predicted)):
     DR[i] = math.sqrt(((y_predicted[i][0] - test_y[i][0])**2 + (y_predicted[i][1] - test_y[i][1])**2 + (y_predicted[i][2] - test_y[i][2])**2))
     DR_reco[i] = math.sqrt(((recoVtx_y[i][0] - test_y[i][0])**2 + (recoVtx_y[i][1] - test_y[i][1])**2 + (recoVtx_y[i][2] - test_y[i][2])**2))
     #print("DR: ", DR)

scores = model.evaluate(x_transformed, test_y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]))
print(scores)

# Score with sklearn.
score_sklearn = metrics.mean_squared_error(y_predicted, test_y)
print('MSE (sklearn): {0:f}'.format(score_sklearn))

#-----------------------------

print(" saving .csv file with predicted variables..")

# Open the CSV file for writing
with open("predicted_features.csv", "w", newline="") as csvfile:
   
    # Define the headers for the CSV file
    fieldnames = ["hitX_" + str(i) for i in range(1, 21)] + ["hitY_" + str(i) for i in range(1, 21)] + ["hitZ_" + str(i) for i in range(1, 21)] + ["hitT_" + str(i) for i in range(1, 21)]

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    # Write the column headers to the CSV file
    writer.writeheader()
    
    # Loop over the rows in the test_x array
    for row in test_x:
        # Extract the hitX, hitY, hitZ, and hitT values for this row
        hitX = row[:20]
        hitY = row[20:40]
        hitZ = row[40:60]
        hitT = row[60:]
        
        # Write these values to the CSV file
        writer.writerow(dict(zip(fieldnames, np.concatenate((hitX, hitY, hitZ, hitT)))))  

#data = np.concatenate((test_y[0], test_y[1], test_y[2], y_predicted[0], y_predicted[1], y_predicted[2]),axis=1)
data = np.concatenate((test_y, recoVtx_y),axis=1)
print(data)
df = pd.DataFrame(data, columns=['trueX','trueY','trueZ','recoX','recoY','recoZ'])
print(df.head())
df1 = pd.DataFrame(DR_reco, columns=['DR_reco'])
df2 = pd.DataFrame(DR, columns=['DR'])
df_f = pd.concat((df,df1),axis=1)
df_ff = pd.concat((df_f,df2),axis=1)
df3= pd.DataFrame(nhits, columns=['nhits'])
df_final = pd.concat((df_ff,df3),axis=1)
print(df_final.head())
df_final.to_csv("predictionsVertex.csv", float_format = '%.3f', index=False)

#Concat the 2 csv with predicted vertex and the others variables

dfa = pd.read_csv('predicted_features.csv')
dfb = pd.read_csv("predictionsVertex.csv")
# dfa=dfa.reset_index(drop=True)
# dfb=dfb.reset_index(drop=True)

final_df = pd.concat([dfa, dfb], axis=1)
print(final_df.head())
# Write the combined dataframe to a new CSV file
# final_df.to_csv("predictionsVertex_3vars.csv", index=False)
final_df.to_csv("predictionsVertex_20hits.csv", index=False)
