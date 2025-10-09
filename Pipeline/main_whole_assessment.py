import os
import json
import multiprocessing
import numpy as np
import pandas as pd
import joblib as jl
from sklearn.model_selection import RepeatedStratifiedKFold
from train_select_classifiers import train_select_classifiers
from train_regressor import train_regressor
from test_classifier_regressor import test_classifier_regressor
from plotting import plot_dashboards, plot_corrcoeff, create_timestamps_list
from AI_plot import plot_AI_raw

# Cambio la directory di esecuzione in quella dove si trova questo file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#data_folder = 'C:/Users/david/Documents/University/Borsa di Studio - REDCap/only_AC-80_patients/'
data_folder = '../../../Dati_RAW/'

number_of_iterations = 5 #8 pazienti per test ad ogni iterazione, standard per dataser medi (30-100 campioni)
min_mean_test_score = 0.7 #soglia minima di accuratezza che un classificatore deve avere per essere considerato valido
window_size = 6400 #dimensione della finestra temporale in campioni (frequenza_campionamento(26.67)*durata_finestra_s)
#una finestra ottimale si aggira intorno ai 2-6 min e cattura pattern motori significativi, riduce rumore e computazionalmente efficiente
folder = 'Iterations/' #cartella dove salvare i risultati delle iterazioni

# Setup cross-validation
if not os.path.exists(folder): #se la cartella non esiste, la crea

    print(' ----- CREATING ITERATIONS FOLDERS AND TRAINING CLASSIFIERS ----- ')
    processes = [] # Lista per tenere traccia dei processi
    iteration = 0  # Contatore per le iterazioni
    metadata = pd.read_excel(data_folder + 'metadata2022_04.xlsx') # Lettura del file excel con i metadati
    labels = metadata['hemi'].values # Estrazione delle etichette per la stratificazione dalla colonna 'hemi'
    rskf = RepeatedStratifiedKFold(n_splits=number_of_iterations, n_repeats=1, random_state=42) # Inizializzazione del cross-validation

    for train_indexes, test_indexes in rskf.split(np.empty(metadata.shape[0]), labels): # Generazione degli indici di train e test

        # Creazione della struttura dati da salvare
        data = {
            "Iteration": iteration, 
            "Train Indexes": train_indexes.tolist(),
            "Test Indexes": test_indexes.tolist()
        }

        save_folder = folder + 'Iteration_' + str(iteration) + '/' # Creazione del percorso della cartella per l'iterazione corrente
        os.makedirs(save_folder) # Creazione della cartella
        iteration += 1 # Incremento del contatore delle iterazioni
        
        with open(save_folder + 'iteration_data.json', 'w') as file: # Scrittura dei dati in un file JSON nella cartella dell'iterazione
            json.dump(data, file, indent=4) # Salvataggio del file JSON
        
        p = multiprocessing.Process(target=train_select_classifiers, args=(data_folder, save_folder, train_indexes, [window_size])) # Creazione del processo
        p.start() # Avvio del processo
        processes.append(p) # Aggiunta del processo alla lista

    for p in processes: # Attesa del completamento di tutti i processi
        p.join()

# Controlla che non esistano già i regressori
if not os.path.exists(folder + 'Iteration_0/Regressors/'):

    print(' ----- TRAINING REGRESSORS ----- ')
    processes = [] # Lista per tenere traccia dei processi

    for iteration in range(number_of_iterations): # Iterazione attraverso il numero di iterazioni

        save_folder = folder + 'Iteration_' + str(iteration) + '/' # Creazione del percorso della cartella per l'iterazione corrente

        # Lettura da un file JSON e accesso ai dati diverso da prima
        with open(save_folder + 'iteration_data.json', 'r') as file: 
            data = json.load(file) # Caricamento dei dati dal file JSON
        retrieved_train_indexes = data['Train Indexes'] # Estrazione degli indici di train
        
        p = multiprocessing.Process(target=train_regressor, args=(data_folder, save_folder, retrieved_train_indexes, min_mean_test_score, window_size)) # Creazione del processo per l'addestramento del regressore
        # si passano la cartella con i dati raw, la cartella dove salvare i risultati, gli indici di train che sono gli stessi dei classificatori, la soglia minima di accuratezza e la dimensione della finestra
        p.start() # Avvio del processo
        processes.append(p) # Aggiunta del processo alla lista

    for p in processes:
        p.join() # Attesa del completamento di tutti i processi

if not os.path.exists(folder + 'Iteration_0/combined_test_stats.json'): # se non esistono già i risultati dei test

    print(' ----- TESTING CLASSIFIER AND REGRESSOR ----- ')
    processes = [] 

    for iteration in range(number_of_iterations): # Iterazione attraverso il numero di iterazioni

        save_folder = folder + 'Iteration_' + str(iteration) + '/' # Creazione del percorso della cartella per l'iterazione corrente

        # Lettura da un file JSON e accesso ai dati
        with open(save_folder + 'iteration_data.json', 'r') as file:
            data = json.load(file) # Caricamento dei dati dal file JSON
        retrieved_test_indexes = data['Test Indexes'] # Estrazione degli indici di test

        p = multiprocessing.Process(target=test_classifier_regressor, args=(data_folder, save_folder, retrieved_test_indexes, min_mean_test_score, window_size)) # Creazione del processo per il test del classificatore e del regressore
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

folder_prefix= folder + "Iteration_" # prefisso per le cartelle delle iterazioni
r2_list = [] # lista per i punteggi R2
corrcoef_list = [] # lista per i coefficienti di correlazione

for i in range(number_of_iterations): # iterazione attraverso il numero di iterazioni
    folder_name = f"{folder_prefix}{i}" # nome della cartella per l'iterazione corrente
    json_file_path = os.path.join(folder_name, "combined_test_stats.json") # percorso del file JSON con i risultati dei test

    if os.path.exists(json_file_path): # verifica che il file esista
        with open(json_file_path, 'r') as json_file: # apertura del file JSON
            data = json.load(json_file) # caricamento dei dati dal file JSON
            r2_list.append(data['Regressor Stats']['R2 Score']) # aggiunta del punteggio R2 alla lista
            corrcoef_list.append(data['Best Classifier Stats']['Correlation Coefficient']) # aggiunta del coefficiente di correlazione alla lista

average_r2_score = np.mean(r2_list) # calcolo del punteggio R2 medio
average_corr_score = np.mean(corrcoef_list) # calcolo del coefficiente di correlazione medio

print(f"The average r2 score for the regressor is: {average_r2_score}")
print(f"The average correlation CPI-AHA is: {average_corr_score}")

results = { 
    "R2 Score List":r2_list,
    "Correlation List":corrcoef_list,
    "Average R2 Score": average_r2_score,
    "Average CPI-AHA Correlation": average_corr_score
} # dizionario con i risultati

# Writing to a JSON file
with open(folder + 'test_results.json', 'w') as file: # apertura del file JSON per la scrittura
    json.dump(results, file, indent=4) # scrittura dei risultati nel file JSON

if not os.path.exists(folder + 'Iteration_0/Week_stats/predictions_dataframe.csv'): # se non esiste già il file con le predizioni

    print(' ----- PLOTTING PREDICTIONS ----- ')
    processes = []

    # Creazione lista timestamps
    if not os.path.exists('timestamps_list'):
        timestamps = create_timestamps_list(data_folder)
        jl.dump(timestamps, 'timestamps_list') # salvo la lista per non doverla ricreare ogni volta


    for iteration in range(number_of_iterations): 

        save_folder = folder + 'Iteration_' + str(iteration) + '/' # Creazione del percorso della cartella per l'iterazione corrente

        with open(save_folder + 'iteration_data.json', 'r') as file:
            data = json.load(file)
        retrieved_test_indexes = data['Test Indexes']

        p = multiprocessing.Process(target=plot_dashboards, args=(data_folder, save_folder, retrieved_test_indexes, min_mean_test_score, window_size)) # Creazione del processo per il plotting delle dashboard
        
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if not os.path.exists(folder + 'Scatter_AHA_CPI_Home-AHA.png'): # se non esiste già il file con il plot delle correlazioni

    print(' ----- PLOTTING CORRELATIONS ----- ')
    iterations_folders = []

    for iteration in range(number_of_iterations):

        iterations_folders.append(folder + 'Iteration_' + str(iteration) + '/')

    plot_corrcoeff(iterations_folders=iterations_folders, save_folder=folder) # chiama la funzione per plottare le correlazioni e salvare il file

if not os.path.exists(folder + 'AI_plots/'): # se non esiste già la cartella per i plot Asymmetry index 
    
    os.makedirs(folder + 'AI_plots/') # la crea

    print(' ----- PLOTTING AI ----- ')

    save_folder = folder + 'AI_plots/' # cartella dove salvare i plot 

    # Legge gli indici di test dalla prima iterazione (sono gli stessi per tutte le iterazioni)
    with open(folder + 'Iteration_0/iteration_data.json', 'r') as file:
        data = json.load(file)
    retrieved_test_indexes = data['Test Indexes'] # indici di test
    plot_AI_raw(data_folder, save_folder, retrieved_test_indexes) # chiama la funzione per plottare gli AI e salvare i file

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