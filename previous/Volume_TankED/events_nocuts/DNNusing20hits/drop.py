#script to remove lines with nan values corresponding to events with hits<20
import csv

input_filename = 'shuffledcsv_20hits_perevt.csv'
output_filename = 'newshuffledcsv_20hits_perevt.csv'
expected_columns = 20
missing_rows = []

# Find rows with missing values
with open(input_filename, 'r') as input_file:
    reader = csv.reader(input_file)
    for row_number, row in enumerate(reader, start=1):
        x_columns = row[:expected_columns]
        y_columns = row[expected_columns:expected_columns*2]
        z_columns = row[expected_columns*2:expected_columns*3]
        t_columns = row[expected_columns*3:expected_columns*4]

        missing_x = any(value.strip() == '' for value in x_columns)
        missing_y = any(value.strip() == '' for value in y_columns)
        missing_z = any(value.strip() == '' for value in z_columns)
        missing_t = any(value.strip() == '' for value in t_columns)

        if missing_x or missing_y or missing_z or missing_t:
            missing_rows.append(row_number)

# Write new CSV file without missing rows
with open(input_filename, 'r') as input_file, open(output_filename, 'w', newline='') as output_file:
    reader = csv.reader(input_file)
    writer = csv.writer(output_file)
    for row_number, row in enumerate(reader, start=1):
        if row_number not in missing_rows:
            writer.writerow(row)

print(f"New CSV file '{output_filename}' created without the rows containing missing values.")
 
