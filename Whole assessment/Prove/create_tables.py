import pandas as pd
import matplotlib.pyplot as plt

folder = 'Iterations/'
results = []

for iteration in range(10):

    save_folder = folder + 'Iteration_' + str(iteration) + '/'

    best_estimators_results = pd.read_csv(save_folder + 'best_estimators_results.csv')
    best_estimators_results = best_estimators_results[['model_type','window_size','method', 'params', 'mean_test_score', 'std_test_score']]
    filtered_results = best_estimators_results[(best_estimators_results['mean_test_score'] >= 0.85) & (best_estimators_results['window_size'] == 300)].copy()
    filtered_results['mean_test_score'] = filtered_results['mean_test_score'].round(4)
    filtered_results['std_test_score'] = filtered_results['std_test_score'].round(4)

    filtered_results.rename(columns={'model_type': 'Model type', 'window_size': 'Window size', 'method': 'Method', 'params': 'Parameters', 'mean_test_score': 'MTS', 'std_test_score': 'STD'}, inplace=True)

    filtered_results.to_csv(save_folder + 'iter'+str(iteration)+'_dataframe.csv', index=False)



# Concatenate all dataframes
#all_results = pd.concat(results)

# Save to txt
#all_results.to_csv('all_results.txt', index=False, sep='\t')


