import os
import csv

# Define the parent directory
parent_dir = '/home/scerramarchi/AInCP-ML/Whole assessment/'

n = 1
threshold = 0.85

# Define the folder names
folders = ['Iterations incriminated '+ str(n)]

# Iterate over the folders
for folder in folders:
    folder_path = os.path.join(parent_dir, folder)
    
    # Iterate over the iteration_x folders
    for i in range(10):
        iteration_folder = f'Iteration_{i}'
        iteration_folder_path = os.path.join(folder_path, iteration_folder)
        
        # Read the best_estimators_results.csv file
        csv_file_path = os.path.join(iteration_folder_path, 'best_estimators_results.csv')
        
        with open(csv_file_path, 'r') as file:
            reader = csv.DictReader(file)
            count = 0
            
            # Count the rows with "mean_test_score" over certain threshold
            for row in reader:
                if float(row['mean_test_score']) > threshold and row['window_size'] == '300':
                    count += 1
            
            print(f'In {iteration_folder}, there are {count} classifiers with "mean_test_score" over {threshold}')