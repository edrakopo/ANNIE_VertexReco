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
infile='/home/evi/Desktop/ANNIE-THESIS-2/VolumeTank/ANNIE_VertexReco/Volume_Tank/shuffled_tankPMT_withonlyMRDcut_insidevolume_withTexp2407.csv'


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
hits, hitT, nhits, labels, gridpoint, cm, texp, mingridx, mingridy, mingridz, recovtx, recofom = np.split(Dataset, [60, 80,81,84,85,88,108, 128,148,168,171 ], axis=1)


#calculate tien difference between expected and hit time
hitDt = hitT-texp
print(hitT[0],"-",texp[0]," = ",hitDt[0] )
print(hitT[1000],"-",texp[100]," = ",hitDt[100])
print("checking gridpoints: ", mingridx[0],",",mingridy[0],",",mingridz[0])
#split events in train/test samples:
np.random.seed(0)
#test_x = np.hstack((hitDt[3000:], mingridx[3000:], mingridy[3000:], mingridz[3000:]))
test_x = np.hstack((hitT[3000:], mingridx[3000:], mingridy[3000:], mingridz[3000:], texp[3000:] ))
test_y = cm[3000:]

print("test sample features shape: ", test_x.shape," test sample label shape: ", test_y.shape)

def custom_loss_function(y_true, y_pred):
    dist = tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), 1))
    return dist

# create model
model = Sequential()
model.add(Dense(50, input_dim=100, kernel_initializer='normal', activation='relu'))
model.add(Dense(30, kernel_initializer='normal', activation='relu'))
model.add(Dense(10, kernel_initializer='normal', activation='relu'))
model.add(Dense(3, kernel_initializer='normal', activation='relu')) 
# load weights
#print("Created model and loaded weights from file")
#model.load_weights("weights_bets.hdf5")

# Compile model
model.compile(loss=custom_loss_function, optimizer='ftrl', metrics=custom_loss_function)
print("Created model and loaded weights from file")

# load weights
print("Created model and loaded weights from file")
model.load_weights("/home/evi/Desktop/ANNIE-THESIS-2/VolumeTank/ANNIE_VertexReco/Volume_Tank/weights_bets.hdf5")

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
test_y_resized = np.resize(test_y, y_predicted.shape)
assert(len(test_y)==len(y_predicted))
#assert(len(test_y)==len(recoVtx_y))
DR = np.empty(len(test_y))
#DR_reco = np.empty(len(test_y))

import math
#print("DR0 : ", math.sqrt(((y_predicted[0][0] - test_y[0][0])**2 + (y_predicted[0][1] - test_y[0][1])**2 + (y_predicted[0][2] - test_y[0][2])**2)))

for i in range (0,len(y_predicted)):
     DR[i] = math.sqrt(((y_predicted[i][0] - test_y[i][0])**2 + (y_predicted[i][1] - test_y[i][1])**2 + (y_predicted[i][2] - test_y[i][2])**2))
     #DR_reco[i] = math.sqrt(((recoVtx_y[i][0] - test_y[i][0])**2 + (recoVtx_y[i][1] - test_y[i][1])**2 + (recoVtx_y[i][2] - test_y[i][2])**2))
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
#f1=ax.hist(dataprev, nbins, histtype='step', fill=False, color='red',alpha=0.75)
#ax.set_xlim(0.,200.)
ax.set_xlabel('$\Delta R$ [cm]')
#ax.legend(('NEW','Previous'))
#ax.xaxis.set_ticks(np.arange(0., 500., 50))
ax.tick_params(axis='x', which='minor', bottom=False)
title = "mean = %.2f, std = %.2f," % (DR.mean(), DR.std())
#title = "mean = %.2f, std = %.2f, Prev: mean = %.2f, std = %.2f " % (data.mean(), data.std(),dataprev.mean(), dataprev.std())
plt.title(title)
plt.show()
fig.savefig('/home/evi/Desktop/ANNIE-THESIS-2/VolumeTank/ANNIE_VertexReco/Volume_Tank/GRIDpointresol_DR.png')
plt.close(fig)
'''
data = np.concatenate((test_y, y_predicted,recoVtx_y),axis=1)
print(data)
df = pd.DataFrame(data, columns=['trueX','trueY','trueZ','DNNX','DNNY','DNNZ','recoX','recoY','recoZ'])
print(df.head())
#df1 = pd.DataFrame(DR_reco, columns=['DR_reco'])
df2 = pd.DataFrame(DR, columns=['DR'])
df_f = pd.concat((df,df1),axis=1)
df_final = pd.concat((df_f,df2),axis=1)
print(df_final.head())

print(" saving .csv file with predicted variables..")
df_final.to_csv("predictionsVertex.csv", float_format = '%.3f')
'''
