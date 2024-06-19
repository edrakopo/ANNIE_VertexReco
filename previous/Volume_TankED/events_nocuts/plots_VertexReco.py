import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#import ROOT
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10, 7),
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

infile = "predictionsVertexFV.csv"

filein = open(str(infile))
print("number of events: ",filein)
df00=pd.read_csv(filein, index_col=0)
#df00 = df000[df000['recoVtxFOM']>87.] #strict cut to avoid mis-reconstructed events
print("df00.head() \n",df00.head())

#--- all events
#data = df00['DR']
#dataprev = df00['DR_reco']

#--- selecting events with recoVtxFOM>0
#data = df00[df00['recoVtxFOM']>0.]['DR']
#dataprev = df00[df00['recoVtxFOM']>0.]['DR_reco']
'''
#--- selecting events with TrueTrackLengthInMRD>0
data = df00[df00['TrueTrackLengthInMRD']>0.]['DR']
#dataprev = df00[[df00['TrueTrackLengthInMRD']>0.]['DR_reco']
#compare with reco events with ['recoVtxFOM']>0.   
dataprev1 = df00[df00['recoVtxFOM']>0.]
dataprev = dataprev1[dataprev1['TrueTrackLengthInMRD']>0.]['DR_reco']
'''
#--- check events with DR<40. 
data = df00[df00['DR']<40.]['DR']
dataprev = df00[df00['DR']<40.]['DR_reco']
print("df00[df00['DR']<40.]: \n",df00[df00['DR']<40.].head(10))

nbins=np.arange(0,400,10)
fig,ax=plt.subplots(ncols=1, sharey=False)#, figsize=(8, 6))
f0=ax.hist(data, nbins, histtype='step', fill=False, color='blue',alpha=0.75) 
f1=ax.hist(dataprev, nbins, histtype='step', fill=False, color='red',alpha=0.75)
#ax.set_xlim(0.,200.)
ax.set_xlabel('$\Delta R$ [cm]')
ax.legend(('NEW','Previous'))
#ax.xaxis.set_ticks(np.arange(0., 500., 50))
ax.tick_params(axis='x', which='minor', bottom=False)
title = "entries: %.1f, mean = %.2f, std = %.2f, Prev: entries: %.1f, mean = %.2f, std = %.2f " % (len(data), data.mean(), data.std(), len(dataprev), dataprev.mean(), dataprev.std())
ax.set_yscale('log') #set logarithmic scale
plt.title(title)
plt.show()
#fig.savefig('resol_DR_FV.png')
#fig.savefig('resol_DR_FV_recoVtxFOM>0.png')
#fig.savefig('resol_DR_FV_TrueTrackLengthInMRD>0.png')
fig.savefig('resol_DR_<40.png')
plt.close(fig)

