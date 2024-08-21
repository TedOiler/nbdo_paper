import pandas as pd

# Define the paths to the input CSV files
file1 = 'cordex_discrete_results.csv'
file2 = 'cordex_continuous_results.csv'
file3 = 'nbdo_results.csv'

# Read the CSV files into DataFrames
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)

# Concatenate the DataFrames
consolidated_df = pd.concat([df1, df2, df3], ignore_index=True)

# Define the path for the output CSV file
output_file = 'consolidated_results.csv'

# Save the consolidated DataFrame to a new CSV file
consolidated_df.to_csv(output_file, index=False)

print(f"Consolidated results saved to {output_file}")
