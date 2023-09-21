import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

# Load the data from the CSV file
infile = "predictionsVertex_3vars_with_dr.csv"
df = pd.read_csv(infile)
print(df.head())
#Get initial conditions
# cuts_df = df[(df["DR"] > 100) & (df["DR_reco"] < 100)] 
cuts_df = df[(df["DR"] <30) & (df["DR_reco"] >30)] 
# cuts_df=df
print(cuts_df["nhits"])
print('cuts:', cuts_df)

# Loop over the rows in cuts_df
for i, row in cuts_df.iterrows():

    # Extract the hitY and dr values for the current row
    hitY = row.loc["hitY_1":"hitY_1100"]
    dr = row.loc["dr_1":"dr_1100"]
    trueY = row["trueY"]
    truedr = row["truedr"]

    # Create a scatter plot for the current row
    plt.scatter(dr, hitY)
    plt.scatter(truedr, trueY)

    # Add axis labels and a title for the current plot
    plt.xlabel("hit X-Z")
    plt.ylabel("hit Y")
    # plt.xlim(-50, 250)
    # plt.ylim(-50, 250)
    plt.title(f"Scatter plot for event {i+1}")
    plt.savefig(f"1005plots/D/event_{i+1}.png")
    plt.clf()
    # plt.show()