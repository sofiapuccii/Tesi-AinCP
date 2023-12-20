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

number_of_iterations = 10
min_mean_test_score = 0.85
window_size = 300

# New
if not os.path.exists('Iterations/'):

    print(' ----- CREATING ITERATIONS FOLDERS AND TRAINING CLASSIFIERS ----- ')
    processes = []
    iteration = 0
    metadata = pd.read_excel(data_folder + 'metadata2023_08.xlsx')
    labels = metadata['hemi'].values
    rskf = RepeatedStratifiedKFold(n_splits=number_of_iterations, n_repeats=1)

    for train_indexes, test_indexes in rskf.split(np.empty(metadata.shape[0]), labels):

        # Creating a structured dictionary
        data = {
            "Iteration": iteration,
            "Train Indexes": train_indexes.tolist(),
            "Test Indexes": test_indexes.tolist()
        }

        save_folder = 'Iterations/Iteration_' + str(iteration) + '/'
        os.makedirs(save_folder)
        iteration += 1
        
        # Writing to a JSON file
        with open(save_folder + 'iteration_data.json', 'w') as file:
            json.dump(data, file, indent=4)
        
        p = multiprocessing.Process(target=train_select_classifiers, args=(data_folder, save_folder, train_indexes))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if not os.path.exists('Iterations/Iteration_0/Regressors/'):

    print(' ----- TRAINING REGRESSORS ----- ')
    processes = []

    for iteration in range(number_of_iterations):

        save_folder = 'Iterations/Iteration_' + str(iteration) + '/'

        # Reading from a JSON file and accessing data
        with open(save_folder + 'iteration_data.json', 'r') as file:
            data = json.load(file)
        retrieved_train_indexes = data['Train Indexes']
        retrieved_test_indexes = data['Test Indexes']

        p = multiprocessing.Process(target=train_regressor, args=(data_folder, save_folder, retrieved_train_indexes, min_mean_test_score, window_size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if not os.path.exists('Iterations/Iteration_0/combined_test_stats.json'):

    print(' ----- TESTING CLASSIFIER AND REGRESSOR ----- ')
    processes = []

    for iteration in range(number_of_iterations):

        save_folder = 'Iterations/Iteration_' + str(iteration) + '/'

        # Reading from a JSON file and accessing data
        with open(save_folder + 'iteration_data.json', 'r') as file:
            data = json.load(file)
        retrieved_train_indexes = data['Train Indexes']
        retrieved_test_indexes = data['Test Indexes']

        p = multiprocessing.Process(target=test_classifier_regressor, args=(data_folder, save_folder, retrieved_test_indexes, min_mean_test_score, window_size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()



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