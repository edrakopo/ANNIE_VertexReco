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
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score

# infile='tankPMT_withonlyMRDcut_insidevolume_withTexp.csv'
infile='/home/evi/Desktop/ANNIE-THESIS-2/VolumeTank/ANNIE_VertexReco/Volume_Tank/shuffled_tankPMT_withonlyMRDcut_insidevolume_withTexp2407.csv'
# infile= '/home/evi/Desktop/ANNIE-THESIS-2/VolumeTank/ANNIE_VertexReco/Volume_Tank/Final_Tank_IncideVolume.csv'
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


# features,hitT, nhits, labels, gridpoint, cm, texp = np.split(Dataset,[60, 80, 81,84,85,88],axis=1)
# features, nhits, labels, gridpoint, cm, mingrid, Dist, texp = np.split(Dataset, [80,81,84,85,88,91,92], axis=1)
# hits, hitT, nhits, labels, gridpoint, cm, texp, mingridx, mingridy, mingridz = np.split(Dataset, [60, 80,81,84,85,88,108, 128,148 ], axis=1)
hits, hitT, nhits, labels, gridpoint, cm, texp, mingridx, mingridy, mingridz, recovtx, recofom = np.split(Dataset, [60, 80,81,84,85,88,108, 128,148,168,171 ], axis=1)


print('nhits', nhits)
# print("features: ",features)
print('hitT',hitT)
print("labels: ", labels)
print("gridpoint ", gridpoint)
print('texp', texp)
print('cm', cm)
print('mingrid', mingridx)
# print('dist', Dist)
# print('dist', dist)

#split events in train/test samples:
# num_events, num_pixels = hitT.shape
# print(num_events, num_pixels)
np.random.seed(0)
train_x = np.hstack((hitT[:3000], mingridx[:3000], mingridy[:3000], mingridz[:3000], texp[:3000] ))
train_y = cm[:3000]
# train_y =texp[:3000]
# train_y = np.hstack((cm[:3000], texp[:3000]))
num_events, num_pixels = train_x.shape
print(num_events, num_pixels)

print("train sample features shape: ", train_x.shape," train sample label shape: ", train_y.shape)

# Scale data (training set) to 0 mean and unit standard deviation.
scaler = preprocessing.StandardScaler()
train_x = scaler.fit_transform(train_x)

def custom_loss_function(y_true, y_pred):
    # Calculate the mean squared error (MSE)
    mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
    return mse

# def custom_loss_function(y_true, y_pred):
#     #dist = math.dist(y_true, y_pred)
#     dist = tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), 1))
#     return dist
   #squared_difference = tf.square(y_true - y_pred)
   #return tf.reduce_mean(squared_difference, axis=-1)

def create_model():
    # create model
    model = Sequential()
    model.add(Dense(50, input_dim=100, kernel_initializer='normal', activation='relu'))
#    model.add(Dense(20, kernel_initializer='normal', activation='relu'))
    model.add(Dense(30, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(3, kernel_initializer='normal', activation='relu')) 
    # optimizer = optimizers.Ftrl(learning_rate=0.001)
    # Compile model
    model.compile(loss=custom_loss_function, optimizer='ftrl', metrics=custom_loss_function)
    # model.compile(loss='mse', optimizer='ftrl', metrics=['mse'])
    return model

# estimator = KerasRegressor(build_fn=create_model, epochs=30, batch_size=4, verbose=0)

# Initialize the KerasRegressor with the create_model function
estimator = KerasRegressor(build_fn=create_model, epochs=30, batch_size=4, verbose=1)

# Fit the model and evaluate with cross-validation
scores = cross_val_score(estimator, train_x, train_y, cv=5, scoring='neg_mean_squared_error')
print("Cross-Validation Scores: ", scores)
'''
kfold = KFold(n_splits=10)#, random_state=seed)
results = cross_val_score(estimator, train_x, train_y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
'''
# checkpoint
filepath="/home/evi/Desktop/ANNIE-THESIS-2/VolumeTank/ANNIE_VertexReco/Volume_Tank/weights_bets.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
callbacks_list = [checkpoint]

# Fit the model
#history = estimator.fit(train_x, train_y, validation_split=0.33, epochs=22, batch_size=1, callbacks=callbacks_list, verbose=0)
#kfold = KFold(n_splits=4)
#results = cross_val_score(estimator, train_x, train_y, cv=kfold)
#print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
history = estimator.fit(train_x, train_y, validation_split=0.33,batch_size=1, callbacks=callbacks_list, verbose=1)
#-----------------------------
# summarize history for loss
f, ax2 = plt.subplots(1,1)
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('Model Loss')
ax2.set_ylabel('Performance')
ax2.set_xlabel('Epochs')
ax2.set_xlim(0.,30.)
ax2.legend(['training loss', 'validation loss'], loc='upper left')
plt.savefig("/home/evi/Desktop/ANNIE-THESIS-2/VolumeTank/ANNIE_VertexReco/Volume_Tank/keras_train_test_volume2407.pdf")