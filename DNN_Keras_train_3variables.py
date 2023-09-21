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
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor

#--------- File with events for reconstruction:
#--- evts for training:
#infile = "tankPMT_forVetrexReco.csv"
# infile = "dataBootcampShuffle.csv"
# infile = "/home/evi/Desktop/ANNIE-THESIS/10cmRecoGridpoint_shuffled.csv"
# infile= '/home/evi/Desktop/ANNIE-THESIS/300hits_shuffled.csv'
# infile2 = "../LocalFolder/data_forRecoLength_9.csv"
infile='shuffledevents_Timebeforemedian.csv'

# Set TF random seed to improve reproducibility
seed = 170
np.random.seed(seed)

print( "--- opening file with input variables!")
#--- events for training - MC events
filein = open(str(infile))
print("evts for training in: ",filein)
Dataset=np.array(pd.read_csv(filein))
# example of a normalization
from numpy import asarray
from sklearn.preprocessing import MinMaxScaler

# define min max scaler
scaler = MinMaxScaler()
# transform data
scaled = scaler.fit_transform(Dataset)
print(scaled)
# np.random.shuffle(Dataset)#shuffling the data sample to avoid any bias in the training
#print(Dataset)
# features, recovertex, labels= np.split(Dataset,[4400,4403],axis=1)
# features, rest1, rest2, recovertex, labels, gridpoint, gridpointreco, gridpointpmt = np.split(Dataset,[4400,5500,5502,5505,5508,5509, 5510],axis=1)
# features, rest1, rest2, recovertex, labels, gridpoint, gridpointreco, gridpointpmt = np.split(Dataset,[4400,5500,5502,5505,5508,5509, 5510],axis=1)

#for 300 hits
# features, nhits, rest2, recovertex, labels, gridpoint, gridpointreco = np.split(Dataset,[1200,1201,1202,1205,1208,1209],axis=1)

#with 20 hits
features, nhits, rest2, recovertex, labels, gridpoint, gridpointreco = np.split(Dataset,[80,81,82,85,88,89],axis=1)

print("totalPMTs:", nhits[0]," min: ", np.amin(nhits)," with index:",np.argmin(nhits) ," max: ",np.amax(nhits))
print("np.mean(totalPMTs): ",np.mean(nhits)," np.median(totalPMTs): ",np.median(nhits))
print("features: ",features[0])
print("recovertex: ",recovertex)
print("labels: ", labels)
#split events in train/test samples:
num_events, num_pixels = features.shape
print(num_events, num_pixels)
np.random.seed(0)
train_x = features[:2000]
train_y = labels[:2000]



print("train sample features shape: ", train_x.shape," train sample label shape: ", train_y.shape)

# Scale data (training set) to 0 mean and unit standard deviation.
scaler = preprocessing.StandardScaler()
train_x = scaler.fit_transform(train_x)

def custom_loss_function(y_true, y_pred):
    #dist = math.dist(y_true, y_pred)
    dist = tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), 1))
    return dist
   #squared_difference = tf.square(y_true - y_pred)
   #return tf.reduce_mean(squared_difference, axis=-1)

def create_model():
    # create model
    model = Sequential()
    model.add(Dense(60, input_dim=80, kernel_initializer='normal', activation='relu'))
#    model.add(Dense(50, kernel_initializer='normal', activation='relu'))
    model.add(Dense(45, kernel_initializer='normal', activation='relu'))
    model.add(Dense(30, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(3, kernel_initializer='normal', activation='relu'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='Adamax', metrics=['accuracy'])
    return model

estimator = KerasRegressor(build_fn=create_model, epochs=20, batch_size=1, verbose=0)

# checkpoint
filepath="weights_bets20.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
callbacks_list = [checkpoint]
# Fit the model
history = estimator.fit(train_x, train_y, validation_split=0.33, epochs=20, batch_size=1, callbacks=callbacks_list, verbose=0)
#-----------------------------
# summarize history for loss
f, ax2 = plt.subplots(1,1)
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('Model Loss')
ax2.set_ylabel('Performance')
ax2.set_xlabel('Epochs')
ax2.set_xlim(0.,20.)
ax2.legend(['training loss', 'validation loss'], loc='upper left')
plt.savefig("keras_train_test_20medianThits.pdf")
#n, bins, patches = plt.hist(lambdamax, 50, density=1, facecolor='r', alpha=0.75)
#plt.savefig("TrueTrackLengthLambdamaxhist.pdf")
#plt.savefig("TrueTrackLengthhist.pdf")
#plt.scatter(lambdamax,labels)
#plt.savefig("ScatterplotTrueTrackLengthwithlambdamax.pdf")