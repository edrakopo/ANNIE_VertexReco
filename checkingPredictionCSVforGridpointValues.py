import csv
import pandas as pd

import pandas as pd

# read the CSV file
df = pd.read_csv('predictionsVertexDRfor10cm1304a.csv')

# select the rows where the column has values greater than 36000
mask = df['Predicted_Gridpoint'] > 36000
result = df[mask]

# print the selected rows
print(result)
