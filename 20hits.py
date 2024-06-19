import pandas as pd
import numpy as np

# Read the original CSV file
infile = 'shuffled_NoCuts_20hits.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(infile)

# Initialize empty lists to store the selected values
selected_x = []
selected_y = []
selected_z = []
selected_t = []

# Iterate over each row
for index, row in df.iterrows():
    # Get the T columns for the current row
    t_cols = [col for col in df.columns if col.startswith('T_')]
    
    # Filter T columns with values not equal to 0
    nonzero_t_cols = [col for col in t_cols if row[col] != 0]
    
    if len(nonzero_t_cols) > 0:
        # Calculate the median of non-zero T values
        median_t = np.median(row[nonzero_t_cols])
         
        # Filter T columns with values below the median
        selected_t_cols = [col for col in nonzero_t_cols if row[col] < median_t][:20]
        num_values = len(selected_t_cols)
        if num_values > 20:        

           # Select corresponding X, Y, and Z columns for the selected T columns
           selected_xyz_cols = [col.replace('T', 'X') for col in selected_t_cols]
           selected_xyz_cols += [col.replace('T', 'Y') for col in selected_t_cols]
           selected_xyz_cols += [col.replace('T', 'Z') for col in selected_t_cols]
        
           # Append the selected values to the lists
           num_selected_t = len(selected_t_cols)
           selected_x.extend(row[selected_xyz_cols[:num_selected_t]].values)
           selected_y.extend(row[selected_xyz_cols[num_selected_t:2*num_selected_t]].values)
           selected_z.extend(row[selected_xyz_cols[2*num_selected_t:3*num_selected_t]].values)
           selected_t.extend(row[selected_t_cols].values)
        else:
           #print("Number of hits: ", num_values)
           #print("selected_t_cols: ",selected_t_cols)
           #if nhits<20 fill in the rest values with random hits above the median
           selected_t_cols1 = [col for col in nonzero_t_cols if row[col] < median_t][:num_values]
           #print("selected_t_cols 1: ",selected_t_cols1) 
           selected_t_cols2 = [col for col in nonzero_t_cols if row[col] >= median_t][:20-num_values] 
           #print("selected_t_cols 2: ",selected_t_cols2)
           selected_t_cols = selected_t_cols1+selected_t_cols2
           #print("selected_t_cols: ",selected_t_cols)
           #print("len(selected_t_cols): ",len(selected_t_cols))
           # Select corresponding X, Y, and Z columns for the selected T columns
           selected_xyz_cols = [col.replace('T', 'X') for col in selected_t_cols]
           selected_xyz_cols += [col.replace('T', 'Y') for col in selected_t_cols]
           selected_xyz_cols += [col.replace('T', 'Z') for col in selected_t_cols]
           #print("len(selected_t_cols): ",len(selected_t_cols))
           # Append the selected values to the lists
           num_selected_t = len(selected_t_cols)
           selected_x.extend(row[selected_xyz_cols[:num_selected_t]].values)
           selected_y.extend(row[selected_xyz_cols[num_selected_t:2*num_selected_t]].values)
           selected_z.extend(row[selected_xyz_cols[2*num_selected_t:3*num_selected_t]].values)
           selected_t.extend(row[selected_t_cols].values)
          
        # Pad the lists with NaN values if there are fewer than 20 values
        num_values = len(selected_t_cols)
        if num_values < 20:
           num_pad_values = 20 - num_values
           selected_x.extend([np.nan] * num_pad_values)
           selected_y.extend([np.nan] * num_pad_values)
           selected_z.extend([np.nan] * num_pad_values)
           selected_t.extend([np.nan] * num_pad_values)           
        '''   
           selected_x.extend([np.nan] * num_pad_values)
           selected_y.extend([np.nan] * num_pad_values)
           selected_z.extend([np.nan] * num_pad_values)
           selected_t.extend([np.nan] * num_pad_values)
        '''
# Create a DataFrame from the selected values
final_df = pd.DataFrame()
for i in range(1, 21):
    final_df[f'X_{i}'] = selected_x[(i-1)::20]

for i in range(1, 21):
    final_df[f'Y_{i}'] = selected_y[(i-1)::20]

for i in range(1, 21):
    final_df[f'Z_{i}'] = selected_z[(i-1)::20]

for i in range(1, 21):
    final_df[f'T_{i}'] = selected_t[(i-1)::20]


# Save the final DataFrame to a new CSV file
#final_df.to_csv('/home/evi/Desktop/ANNIE-THESIS-2/VolumeTank/ANNIE_VertexReco/Volume_Tank/VolumeTank_Texpected2407.csv', index=False)

#ADD COLUMNS FOR NHITS, GRIDPOINT, AND TRUE GRID COORDS

# Extract the desired columns from the extra DataFrame
#add_columns = ['totalPMTs', 'truevtxX', 'truevtxY', 'truevtxZ', 'Gridpoint','recovtxX','recovtxY','recovtxZ','recoVtxFOM']
add_columns = ['totalPMTs','recovtxX','recovtxY','recovtxZ','truevtxX', 'truevtxY', 'truevtxZ','Gridpoint', 'nhits','recoVtxFOM','TrueTrackLengthInMrd']
extra_columns = df[add_columns]

# Concatenate the existing DataFrame with the extra columns
combined_df = pd.concat([final_df, extra_columns], axis=1)

# Save the combined DataFrame to a new CSV file
combined_df.to_csv('shuffledcsv_20hits_perevt.csv')

