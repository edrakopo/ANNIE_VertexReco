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

infile = '/home/evi/Desktop/ANNIE-THESIS/tankPMT_trueGridpoint.csv'
infile2 = '/home/evi/Desktop/ANNIE-THESIS/predictionsVertexDRfor10cm.csv'

  
# reading two csv files
df1 = pd.read_csv('/home/evi/Desktop/ANNIE-THESIS/tankPMT_trueGridpoint.csv')
df2 = pd.read_csv('/home/evi/Desktop/ANNIE-THESIS/predictionsVertexDRfor10cm.csv')


# merge the dataframes based on a common column
merged_df = pd.merge(df1, df2, on='Gridpoint')

# write the merged dataframe to a new CSV file
merged_df.to_csv('merged_file.csv', index=False)

  

  



