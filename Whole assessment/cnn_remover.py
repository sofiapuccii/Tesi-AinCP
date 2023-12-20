import os
import pandas as pd

# Set the path to the Iterations folder
iterations_folder_path = 'Iterations'

# Iterate through each iteration folder
for i in range(10):
    iteration_folder = f'Iteration_{i}'
    csv_file_path = os.path.join(iterations_folder_path, iteration_folder, 'best_estimators_stats.csv')

    # Check if the CSV file exists
    if os.path.exists(csv_file_path):
        # Read the CSV file
        df = pd.read_csv(csv_file_path)

        # Filter out rows where 'model_type' contains 'cnn'
        df_filtered = df[~df['model_type'].str.contains('cnn', case=False, na=False)]

        # Save the filtered DataFrame back to CSV
        df_filtered.to_csv(csv_file_path, index=False)
        print(f'Processed {csv_file_path}')
    else:
        print(f'File not found: {csv_file_path}')
