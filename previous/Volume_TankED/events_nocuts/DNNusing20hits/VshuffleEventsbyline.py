import numpy as np
import pandas as pd

# infile='tankPMT_withonlyMRDcut_insidevolume_withTexp.csv'
# infile='/home/evi/Desktop/ANNIE-THESIS-2/VolumeTank/ANNIE_VertexReco/Volume_Tank/tankPMT_withonlyMRDcut_insidevolume_withTexp2407_new.csv'
#infile="/home/evi/Desktop/ANNIE-THESIS-2/VolumeTank/ANNIE_VertexReco/Volume_Tank/nocuts/tankPMT_nocut_insidevolume.csv"
infile ="../tankPMT_nocut_insidevolume.csv"

#df=pd.read_csv('/home/evi/Desktop/ANNIE-THESIS-2/VolumeTank/ANNIE_VertexReco/Volume_Tank/nocuts/tankPMT_nocut_insidevolume.csv')
df=pd.read_csv(infile)
# shuffle the dataframe rows
df_shuffled = df.sample(frac=1)

# save the shuffled dataframe to a new CSV file
# df_shuffled.to_csv('shuffled_tankPMT_withonlyMRDcut_insidevolume_withTexp2407.csv', index=False)

df_shuffled.to_csv('shuffled_NoCuts_20hits.csv', index=False)
