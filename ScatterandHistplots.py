import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

# Load the data from the CSV file
infile = "predictionsVertex_3vars_with_dr.csv"
df = pd.read_csv(infile)
print(df.head())
#Get initial conditions
cuts_df = df[(df["DR"] > 100) & (df["DR_reco"] > 100)] 
# cuts_df = df[(df["DR"] <30) & (df["DR_reco"] <30)] 
# cuts_df=df
print(cuts_df["nhits"])
print('cuts:', cuts_df)

# Loop over the rows in cuts_df
# for i, row in cuts_df.iterrows():

#     # Extract the hitY and dr values for the current row
#     hitY = row.loc["hitY_1":"hitY_1100"]
#     dr = row.loc["dr_1":"dr_1100"]
#     trueY = row["trueY"]
#     truedr = row["truedr"]


#   # Calculate mean and median of hitY and dr
#     mean_hitY = np.mean(hitY)
#     median_hitY = np.median(hitY)
#     mean_dr = np.mean(dr)
#     median_dr = np.median(dr)


#     # Create a scatter plot for the current row
#     plt.scatter(dr, hitY)
#     plt.scatter(truedr, trueY)

#     # Add axis labels and a title for the current plot
#     # plt.xlabel("hit X-Z")
#     # plt.ylabel("hit Y")
#     # plt.title(f"Scatter plot for event {i+1}")
#     # plt.savefig(f"1005plots/B/event_{i+1}.png")
#     # plt.clf()
#     # plt.show()


# # Loop over the T columns and create a histogram for each column
for i, row in cuts_df.iterrows():

    hitT = row.loc["hitT_1":"hitT_1100"]
    hitt = hitT[hitT != 0]
    
    # Create a histogram for T
    mean_hitT = np.mean(hitt)
    median_hitT = np.median(hitt)

    plt.axvline(x=mean_hitT, color='r', linestyle='-', label='Mean hitT')
    plt.axvline(x=median_hitT, color='g', linestyle='-', label='Median hitT')

    # Add mean and median values to plot
    plt.text(0.95, 0.5, f"Mean hitT = {mean_hitT:.2f}\nMedian hitT = {median_hitT:.2f}",
             transform=plt.gca().transAxes, ha='right', va='center', bbox=dict(facecolor='white', edgecolor='black'))
  
    plt.legend()
    plt.hist(hitT, bins=50, range=(1,50))
    plt.xlabel("Time")
    # plt.ylabel("time')
    plt.title(f"Histogram for event {i+1}")
    plt.legend()
    plt.savefig(f"1105plots/B/hist time event_{i+1}.png")
    plt.clf()
    # plt.show()

