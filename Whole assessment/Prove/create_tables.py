import pandas as pd
import json

import matplotlib.pyplot as plt

folder = 'Iterations/'
results = []
iteration_stats = {}

for iteration in range(10):
    save_folder = folder + 'Iteration_' + str(iteration) + '/'

    best_estimators_results = pd.read_csv(save_folder + 'best_estimators_results.csv')
    best_estimators_results = best_estimators_results[['model_type','window_size','method', 'params', 'mean_test_score', 'std_test_score']]
    filtered_results = best_estimators_results[(best_estimators_results['mean_test_score'] >= 0.85) & (best_estimators_results['window_size'] == 300)].copy()
    filtered_results['mean_test_score'] = filtered_results['mean_test_score'].round(4)
    filtered_results['std_test_score'] = filtered_results['std_test_score'].round(4)

    filtered_results.rename(columns={'model_type': 'Model type', 'window_size': 'Window size', 'method': 'Method', 'params': 'Parameters', 'mean_test_score': 'MTS', 'std_test_score': 'STD'}, inplace=True)

    filtered_results['Iteration'] = iteration  # Add iteration column
    results.append(filtered_results)

    # Update iteration stats
    for model_type in filtered_results['Model type'].unique():
        if model_type in iteration_stats:
            iteration_stats[model_type].append(iteration)
        else:
            iteration_stats[model_type] = [iteration]

# Concatenate all dataframes
all_results = pd.concat(results)

# Save concatenated dataframe to CSV
#all_results.to_csv(folder + 'all_results.csv', index=False)

# Group by configuration and count the number of iterations
configurations = all_results.groupby(['Model type', 'Method', 'Parameters']).agg({'Iteration': 'nunique'}).reset_index()

# Order the rows of the dataframe by the values of the 'Iteration' column
configurations = configurations.sort_values('Iteration', ascending=False)

# Convert configurations dataframe to JSON
configurations_json = configurations.to_json(orient='records', indent=4)

# Print JSON
print(configurations_json)
    
# Convert configurations JSON to dataframe
configurations_df = pd.read_json(configurations_json)

# Save configurations dataframe to CSV
configurations_df.to_csv(folder + 'configurations.csv', index=False)