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
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor

#--------- File with events for reconstruction:
#--- evts for training:
#infile = "tankPMT_forVetrexReco.csv"
infile = "/home/evi/Desktop/ANNIE-THESIS/tankPMT_forVetrexReco_withRecoQ.csv"


# read the original CSV into a pandas dataframe
df = pd.read_csv("/home/evi/Desktop/ANNIE-THESIS/tankPMT_forVetrexReco_withRecoQ.csv")

# shuffle the dataframe rows
df_shuffled = df.sample(frac=1)

# save the shuffled dataframe to a new CSV file
df_shuffled.to_csv('shuffled.csv', index=False)

