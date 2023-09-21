import sys
import glob
import numpy as np
import pandas as pd

infile='/home/evi/Desktop/ANNIE-THESIS-2/VolumeTank/ANNIE_VertexReco/Volume_Tank/shuffled_tankPMT_withonlyMRDcut_insidevolume_withTexp2407.csv'

print( "--- opening file with input variables!")

filein = open(str(infile))
print("evts for training in: ",filein)

Dataset=np.array(pd.read_csv(filein))

hits, hitT, nhits, labels, gridpoint, cm, texp, mingridx, mingridy, mingridz, recovtx, recofom = np.split(Dataset, [60, 80,81,84,85,88,108, 128,148,168,171 ], axis=1)


print('nhits', nhits)
# print("features: ",features)
print('hitT',hitT)
print("labels: ", labels)
print("gridpoint ", gridpoint)
print('texp', texp)
print('cm', cm)
print('mingrid', mingridz)
print('reco', recovtx)
print('recofom', recofom)
# print('dist', Dist)
# print('dist', dist)

