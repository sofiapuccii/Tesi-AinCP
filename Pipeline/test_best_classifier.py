import os
import hashlib
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sktime.base import BaseEstimator
from sklearn.metrics import r2_score
from predict_samples import predict_samples

def test_best_classifier(data_folder, save_folder, subjects_indexes): 

    best_classifier = pd.read_csv(save_folder + 'best_estimators_results.csv', index_col=0).sort_values(by=['mean_test_score', 'std_test_score'], ascending=False).iloc[0] # carica i risultati da train_select_classifiers ordina per accuratezza e stabilità e seleziona il primo cioè il migliore
    # esempio di riga selezionata: best_classifier={'method':'concat', 'window_size':600, 'model_type':'TimeSeriesKMeans', 'gridsearch_hash':'a1b2c3d4e5','mean_test_score':0.87, 'std_test_score':0.02}
    metadata = pd.read_excel(data_folder + 'metadata2023_08.xlsx').iloc[subjects_indexes] # carica il file excel con i metadati e seleziona solo le righe corrispondenti agli indici dei soggetti passati come argomento, rimuove le colonne non necessarie per alleggerire
    metadata.drop(['age_aha', 'gender', 'dom', 'date AHA', 'start AHA', 'stop AHA'], axis=1, inplace=True) # colonne da rimuovere, quelle che rimangono solo subject, AHA, MACS, hemi

    #estimators_specs_list = [row for index, row in best_estimators_df[best_estimators_df['mean_test_score'] >= 0.5].iterrows()]

    estimators_list = [] # lista di dizionari, ogni dizionario contiene un estimatore addestrato e le sue specifiche (metodo, window_size, hemi_cluster)
    model_params_concat = '' # stringa che concatena i parametri di tutti gli stimatori caricati, usata per creare un hash unico per il regressore

    #for estimators_specs in estimators_specs_list:
    estimator_dir = save_folder + "Trained_models/" + best_classifier['method'] + "/" + str(best_classifier['window_size']) + "_points/" + best_classifier['model_type'].split(".")[-1] + "/gridsearch_" + best_classifier['gridsearch_hash']  + "/" # percorso della cartella che contiene il modello addestrato

    with open(estimator_dir + 'GridSearchCV_stats/best_estimator_stats.json', "r") as stats_f: # carica i parametri del modello migliore trovato durante la ricerca a griglia
        grid_search_best_params = json.load(stats_f) # carica il file json con i parametri del modello migliore
 
    estimator = BaseEstimator().load_from_path(estimator_dir + 'best_estimator.zip') # carica il modello addestrato
    estimators_list.append({'estimator': estimator, 'method': best_classifier['method'], 'window_size': best_classifier['window_size'], 'hemi_cluster': grid_search_best_params['Hemi cluster']}) # aggiunge il modello e le sue specifiche alla lista
    print('Loaded -> ', estimator_dir + 'best_estimator.zip')
    model_params_concat = model_params_concat + str(estimator.get_params()) # concatena i parametri del modello alla stringa

    #reg_path = 'Regressors/regressor_'+ (hashlib.sha256((model_params_concat).encode()).hexdigest()[:10])

    x = [] # lista dei valori CPI previsti dai modelli
    y = metadata['AHA'] # lista dei valori AHA reali dai metadati

    for index in subjects_indexes: # per ogni soggetto nell'elenco degli indici dei soggetti
        _, hp_tot, _, _ = predict_samples(data_folder, estimators_list, index+1) # predice il valore CPI usando i modelli caricati, index+1 perchè gli indici dei soggetti partono da 1 mentre gli indici delle liste da 0
        x.append(hp_tot[0]) # aggiunge il valore CPI previsto alla lista x


    # Alternative 1
    #for i in range (metadata.shape[0]):
    #    _, hp_tot, _, _ = predict_samples(data_folder, estimators_list, metadata['subject'].iloc[i])
    #    x.append(hp_tot[0])
    #    y.append(metadata['AHA'].iloc[i])

    # Alternative 2
    #for subj in subjects_indexes:
    #    _, hp_tot, _, _ = predict_samples(data_folder, estimators_list, subj)
    #    x.append(hp_tot[0])
    #    y.append(metadata[metadata['subject'] == subj]['AHA'].iloc[0])

        #   hp_tot_list_list =                 y =
        #   [[ 95.0, 90.0, 80.0],              [56,
        #    [ 95.0, 90.0, 80.0],               70,
        #    [ 95.0, 90.0, 80.0],               80,
        #    [ 95.0, 90.0, 80.0]]               34]

    # Corcoef CPI (X) - AHA (y)
    corcoef = np.corrcoef(x, y) # calcola la matrice di correlazione tra i valori CPI previsti e i valori AHA reali
    print(' - corrcoef: ', corcoef) # la matrice di correlazione è una matrice 2x2, gli elementi diagonali sono 1 (correlazione perfetta con se stessi), gli elementi fuori diagonale sono la correlazione tra x e y

    # Organizing data into a dictionary
    data = { 
        "Method": best_classifier['method'],
        "Window Size": best_classifier['window_size'],
        "Model Type": best_classifier['model_type'],
        "Gridsearch Hash": best_classifier['gridsearch_hash'],
        "Correlation Coefficient": corcoef
    }

    # Writing to a JSON file
    with open(save_folder + 'best_classifier_stats.json', 'w') as file:
        json.dump(data, file, indent=4)