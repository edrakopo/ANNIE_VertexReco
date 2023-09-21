import sys
import glob
import numpy as np
import pandas as pd
import tempfile
import random
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from array import array

infile='tankPMT_withonlyMRDcut_insidevolume2.csv'

print( "--- opening file with input variables!")
#--- events for training - MC events
filein = open(str(infile))
print("evts for training in: ",filein)
#print((pd.read_csv(filein)).head())
Dataset=np.array(pd.read_csv(filein))

hitx, hity, hitz, hitT, totalPMTs, labels, gridpoint, cm  = np.split(Dataset,[1100,2200,3300,4400,4401,4404, 4405],axis=1)

print("hitz: ",hitz[0])
print("hitT: ",hitT[0])
print("labels: ", labels)
print("gridpoint ", gridpoint)
print('cm',cm)
print("totalPMTs:", totalPMTs[0]," min: ", np.amin(totalPMTs)," with index:",np.argmin(totalPMTs) ," max: ",np.amax(totalPMTs)) #nhits
print("np.mean(totalPMTs): ",np.mean(totalPMTs)," np.median(totalPMTs): ",np.median(totalPMTs))

#checking shape:
print(type(hitT)," len(hitT): ",len(hitT))
print("hitT.shape: ", hitT.shape)
print("hitx.shape: ",hitx.shape)
#print(hitT)
assert(len(hitx)==len(hitT))

#for one event try to convert to  df: 
def hits_medianT(dt,dx,dy,dz): 
    
    d00 = pd.concat([dt, dx, dy, dz],axis=1)
    print('d00', d00.head())
    d0 = d00[d00.columns[d00.columns.str.startswith('T_')]]  # Select columns starting with 'T_'
    d0.columns = d0.columns.astype(str) 
    print(d0)
    #calculate median and its index: 
    for col in d0.columns:
        medianT = d0[col].median()
        posmedian = d0.loc[d0[col] == medianT]
        print("posmedian: ", posmedian, " with value: ", medianT)

        d0.sort_values(by=col, inplace=True)
        print("d0 after sorting: \n", d0)
        pos1 = d0[d0[col] > medianT].iloc[0]
        pos2 = d0[d0[col] < medianT].iloc[-1]
        print("pos1: ", pos1, " pos2: ", pos2)
        df_short = d0[d0[col] < medianT]
        print(df_short)

        df_ToUse = df_short.iloc[:20]  # Select the first 20 rows
        print('df to use', df_ToUse)

        # Create new DataFrame with 20 columns each for X, Y, Z, and T_i
        new_df = pd.DataFrame()
        for c in ['X', 'Y', 'Z']:
            for i in range(len(hitx)):
                new_col = f'{c}_{i+1}'
                new_df[new_col] = df_ToUse[c]

        # Copy the rest of the columns from the original DataFrame
        for col2 in d00.columns:
            if col2 not in ['X', 'Y', 'Z']:
                new_df[col2] = d00[col2]
        print('newdf', new_df)
        return new_df


    # for i, row in Dataset.iterrows():
    #     hitT = row.loc["hitT_1":"hitT_1100"]
    #     hitt = hitT[hitT != 0]

df_x=pd.DataFrame(hitx)
df_y=pd.DataFrame(hity)
df_z=pd.DataFrame(hitz)
df_t=pd.DataFrame(hitT)

# print("hitT[i][:].shape: ",hitT[0][:].shape)
# Dt = pd.DataFrame(hitT[0][:], columns = ['T'])
# Dx = pd.DataFrame(hitx[0][:], columns = ['X'])
# Dy = pd.DataFrame(hity[0][:], columns = ['Y'])
# Dz = pd.DataFrame(hitz[0][:], columns = ['Z'])

# Filter hits based on median T and select the first 20 hits
# modified_df = hits_medianT(pd.DataFrame(hitT), pd.DataFrame(hitx), pd.DataFrame(hity), pd.DataFrame(hitz))


# new_df_x = pd.DataFrame(hitx[:, :20], columns=[f'X_{i+1}' for i in range(20)])
# new_df_y = pd.DataFrame(hity[:, :20], columns=[f'Y_{i+1}' for i in range(20)])
# new_df_z = pd.DataFrame(hitz[:, :20], columns=[f'Z_{i+1}' for i in range(20)])
# new_df_t = pd.DataFrame(hitT[:, :20], columns=[f'T_{i+1}' for i in range(20)])


# print(hits_medianT(Dt, Dx, Dy, Dz))

# Create the modified DataFrame
modified_df = hits_medianT(df_t, df_x,df_y,df_z)

output_df = pd.DataFrame({'nhits': totalPMTs[:,0],
                          'truevtxX': labels[:, 0],
                          'truevtxY': labels[:, 1],
                          'truevtxZ': labels[:, 2],
                          'Gridpoint': gridpoint[:,0],
                          'xc': cm[:, 0],
                          'yc': cm[:, 1],
                          'zc': cm[:, 2]
                          
})

# print(output_df)
# final_df = pd.concat([new_df_x, new_df_y, new_df_z, new_df_t, output_df], axis=1)
# final_df = pd.concat([new_df_t, output_df], axis=1)

# final_df = pd.concat([modified_df, output_df], axis=1)
final_df = pd.concat([modified_df, output_df], axis=1)
print(final_df)

# Save the modified DataFrame to a new CSV file
# final_file = 'shuffledevents_Timebeforemedian.csv'

final_file = '/home/evi/Desktop/ANNIE-THESIS-2/VolumeTank/ANNIE_VertexReco/Volume_Tank/VolumeTank_Texpected1.csv'
final_df.to_csv(final_file, float_format = '%.3f', index=False)
print(f'Saved the modified data to {final_file}.')
