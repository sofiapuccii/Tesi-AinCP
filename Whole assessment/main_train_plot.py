import os
import json
import multiprocessing
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from train_select_classifiers import train_select_classifiers
from train_regressor import train_regressor
from test_classifier_regressor import test_classifier_regressor

# Cambio la directory di esecuzione in quella dove si trova questo file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#data_folder = 'C:/Users/david/Documents/University/Borsa di Studio - REDCap/only_AC-80_patients/'
data_folder = '../../only_AC-80_patients/'

min_mean_test_score = 0.85
window_size = 300

# New
if not os.path.exists('Best_model/Classifiers/'):

    print(' ----- TRAINING CLASSIFIERS ----- ')

    metadata = pd.read_excel(data_folder + 'metadata2023_08.xlsx')
    subjects_indexes = list(range(len(metadata)))
    np.random.shuffle(subjects_indexes)
    train_select_classifiers(data_folder, save_folder='Best_model/Classifiers/', subjects_indexes=subjects_indexes)
    
if not os.path.exists('Best_model/Regressor/'):

    print(' ----- TRAINING REGRESSOR ----- ')
    
    train_regressor(data_folder, save_folder='Best_model/Regressor/', subjects_indexes=subjects_indexes, min_mean_test_score=min_mean_test_score, window_size=window_size)

if not os.path.exists('Iterations/Iteration_0/combined_test_stats.json'):

    print(' ----- TESTING CLASSIFIER AND REGRESSOR ----- ')
    processes = []

    for iteration in range(number_of_iterations):

        save_folder = 'Iterations/Iteration_' + str(iteration) + '/'

        # Reading from a JSON file and accessing data
        with open(save_folder + 'iteration_data.json', 'r') as file:
            data = json.load(file)
        retrieved_test_indexes = data['Test Indexes']

        p = multiprocessing.Process(target=test_classifier_regressor, args=(data_folder, save_folder, retrieved_test_indexes, min_mean_test_score, window_size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

folder_prefix= "Iterations/Iteration_"
r2_list = []
corrcoef_list = []

for i in range(number_of_iterations):
    folder_name = f"{folder_prefix}{i}"
    json_file_path = os.path.join(folder_name, "combined_test_stats.json")

    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
            r2_list.append(data['Regressor Stats']['R2 Score'])
            corrcoef_list.append(data['Best Classifier Stats']['Correlation Coefficient'])

average_r2_score = np.mean(r2_list)
average_corr_score = np.mean(corrcoef_list)

print(f"The average r2 score for the regressor is: {average_r2_score}")
print(f"The average correlation CPI-AHA is: {average_corr_score}")

results = {
    "R2 Score List":r2_list,
    "Correlation List":corrcoef_list,
    "Average R2 Score": average_r2_score,
    "Average CPI-AHA Correlation": average_corr_score
}

# Writing to a JSON file
with open('Iterations/test_results.json', 'w') as file:
    json.dump(results, file, indent=4)


print(' ----- ESECUZIONE DEL MAIN TERMINATA ----- ')


"""
1. Allenare classificatori                                      (train-AHA)

2. Allenare regessore                                           (train-week)

3. Testare miglior classificatore (corfcoef) e regressore (r2)  (test-week)
    4.1 Predict samples sui best estimators (est1, est2, est3)  ->  hp_tot_list_list  
    4.2 Corrcoef tra la prima colonna di hp_tot_list_list e y
    4.3 Calcolare r2 tra regressor(hp_tot_list_list) e y

Iterazione -> corrcoef e r2 / [1, 2, 3][55, 60, 68]
Iterazione -> corrcoef e r2 / [4, 5, 6][55, 60, 68]
Iterazione -> corrcoef e r2 / [7, 8, 9][55, 60, 68]

[1, 2, 34, 4, 5, 6, 7, 8, 9]

best estimators
    est1
    est2
    est3

"""