import pandas as pd

# File paths
csv_file1 = r"D:\Project Work\AFTER COMPRY MODEL REFINMENT\FDA approve\main data\fda_approve_des.csv"
csv_file2 = r"D:\Project Work\AFTER COMPRY MODEL REFINMENT\FDA approve\main data\for_pred_1998.csv"

# Read the CSV files
df1 = pd.read_csv(csv_file1)
df2 = pd.read_csv(csv_file2)

# Extract columns from the second CSV file that are present in the first CSV file
common_columns = list(set(df1.columns) & set(df2.columns))

# Reorder columns in the second DataFrame to match the order in the first DataFrame
result_df2 = df2[common_columns].reindex(columns=df1.columns)

# Save the resulting DataFrame to a new CSV file
result_df2.to_csv(r"D:\Project Work\AFTER COMPRY MODEL REFINMENT\FDA approve\main data\fda_used_for_pred.csv", index=False)

# Display the resulting DataFrame
print(result_df2)
