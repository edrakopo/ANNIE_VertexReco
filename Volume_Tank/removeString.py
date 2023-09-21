import pandas as pd
import numpy as np
import math
import csv
import ast

infile='tankPMT_withonlyMRDcut_insidevolume_withTexp.csv'



# Read the CSV file into a DataFrame
df = pd.read_csv('tankPMT_withonlyMRDcut_insidevolume_withTexp.csv')

# df = df.drop('Min_dt_values', axis=1)

columns_to_remove = ['Min_dt_values', 'MinGrid_dt_values']
df = df.drop(columns_to_remove, axis=1)

# Save the modified DataFrame to a new CSV file
df.to_csv('modified_csv_file.csv', index=False)

#MinGrid_dt_values