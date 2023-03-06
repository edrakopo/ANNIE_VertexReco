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
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

#--------- File with events for reconstruction:
#--- evts for training:
#infile = "tankPMT_forVetrexReco.csv"
infile = "shuffled.csv"
#

# Set TF random seed to improve reproducibility
seed = 170
np.random.seed(seed)

print( "--- opening file with input variables!")
#--- events for training - MC events
filein = open(str(infile))
print("evts for training in: ",filein)
df = pd.read_csv(filein)

#processing feautures 


Dataset=np.array(df)
#np.random.shuffle(Dataset)#shuffling the data sample to avoid any bias in the training
#print(Dataset)
#features, totalPMTs, recovertex, labels, gridpoint = np.split(Dataset,[4401,4403,4406,4409],axis=1)
features, totalPMTs, totalLAPPDs, recovertex, labels, gridpoint = np.split(Dataset,[4400,4401,4402,4405,4408],axis=1)
print("totalPMTs:", totalPMTs[0]," min: ", np.amin(totalPMTs)," with index:",np.argmin(totalPMTs) ," max: ",np.amax(totalPMTs))
print("np.mean(totalPMTs): ",np.mean(totalPMTs)," np.median(totalPMTs): ",np.median(totalPMTs))
'''
#printing all array elements for specific selection
with np.printoptions(threshold=np.inf):
    print(features[2615][3301:3330])
'''
print("features: ",features[0])
print("recovertex: ",recovertex)
print("labels: ", labels)

#split events in train/test samples:
num_events, num_pixels = features.shape
print(num_events, num_pixels)
np.random.seed(0)
train_x = features[:3000]
train_y = labels[:3000]

print("train sample features shape: ", train_x.shape," train sample label shape: ", train_y.shape)

# Scale data (training set) to 0 mean and unit standard deviation.
scaler = preprocessing.StandardScaler()
train_x = scaler.fit_transform(train_x)

def create_model():
    # create model
    model = Sequential()
    model.add(Dense(20, input_dim=4400, kernel_initializer='normal', activation='relu'))
#    model.add(Dense(20, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(3, kernel_initializer='normal', activation='relu'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='Adamax', metrics=['accuracy'])
    return model

estimator = KerasRegressor(build_fn=create_model, epochs=22, batch_size=10, verbose=0)
'''
kfold = KFold(n_splits=10)#, random_state=seed)
results = cross_val_score(estimator, train_x, train_y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
'''
# checkpoint
filepath="weights_bets.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
callbacks_list = [checkpoint]

# Fit the model
history = estimator.fit(train_x, train_y, validation_split=0.33, epochs=22, batch_size=1, callbacks=callbacks_list, verbose=0)
#-----------
'''
estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=100, verbose=False)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X, y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

estimator.fit(X, y)
prediction = estimator.predict(X)
accuracy_score(y, prediction)
'''
#-----------
#-----------------------------
# summarize history for loss
f, ax2 = plt.subplots(1,1)
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('Model Loss')
ax2.set_ylabel('Performance')
ax2.set_xlabel('Epochs')
ax2.set_xlim(0.,50.)
ax2.legend(['training loss', 'validation loss'], loc='upper left')
plt.savefig("keras_train_test.pdf")

