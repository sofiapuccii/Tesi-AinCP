import os
import json
import multiprocessing
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from train_classifiers import train_classifiers
from test_classifiers import test_classifiers
from test_regressor import test_regressor

# Cambio la directory di esecuzione in quella dove si trova questo file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#data_folder = 'C:/Users/david/Documents/University/Borsa di Studio - REDCap/only_AC-80_patients/'
data_folder = '../../only_AC-80_patients/'
metadata = pd.read_excel(data_folder + 'metadata2023_08.xlsx')
labels = metadata['hemi'].values

# Repeated Stratified K-Fold cross-validator
rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=1)
iteration = 0
processes = []

# Iterazioni
for train_index, test_index in rskf.split(np.empty(metadata.shape[0]), labels):

    save_folder = 'Iterations/Iteration_' + str(iteration) + '/'
    
    if not os.path.exists(save_folder):

        os.makedirs(save_folder)
        
        # Creating a structured dictionary
        data = {
            "Iteration": iteration,
            "Train Indexes": train_index.tolist(),
            "Test Indexes": test_index.tolist()
        }

        # Writing to a JSON file
        with open(save_folder + 'iteration_data.json', 'w') as file:
            json.dump(data, file, indent=4)

        #Chiamata alla funzione per allenare i migliori modelli e selezionarli
        p = multiprocessing.Process(target=train_classifiers, args=(data_folder, save_folder, train_index))
        p.start()
        processes.append(p)

    iteration += 1

for p in processes:
    p.join()

processes = []

for iter in range(iteration):

    save_folder = 'Iterations/Iteration_' + str(iter) + '/'

    # Reading from a JSON file and accessing data
    with open(save_folder + 'iteration_data.json', 'r') as file:
        data = json.load(file)
    retrieved_train_indexes = data['Train Indexes']
    retrieved_test_indexes = data['Test Indexes']

    p = multiprocessing.Process(target=test_classifiers, args=(data_folder, save_folder, retrieved_test_indexes))
    p.start()
    processes.append(p)

    p = multiprocessing.Process(target=test_regressor, args=(data_folder, save_folder, retrieved_train_indexes, retrieved_test_indexes, 0.85, 300))
    p.start()
    processes.append(p)

    


    
for p in processes:
    p.join()