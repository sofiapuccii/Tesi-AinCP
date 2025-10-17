import numpy as np
import pandas as pd
from elaborate_magnitude import elaborate_magnitude
from create_windows import decimate_df

def predict_samples(data_folder, estimators, patient):

    if not estimators:
        print('You have selected zero estimators to predict the samples with')
        exit(1)

    if len(set(es['window_size'] for es in estimators)) != 1: # tutti gli estimatori devono avere la stessa window_size
        print('You have selected estimators that operates on different window sizes')
        exit(1)

    df = pd.read_csv(data_folder + 'data/week/' + str(patient) + '_week_RAW.csv') # Caricamento del file CSV dei dati grezzi del paziente
    df = decimate_df(df, 3) # Decimazione delle time series a 26.67Hz
    #costruzione di due serie temporali che rappresentano l'intensità di movimento per ciascun braccio.
    magnitude_D = np.sqrt(np.square(np.array(df['x_D'])) + np.square(np.array(df['y_D'])) + np.square(np.array(df['z_D']))) 
    magnitude_ND = np.sqrt(np.square(np.array(df['x_ND'])) + np.square(np.array(df['y_ND'])) + np.square(np.array(df['z_ND']))) 

    #fase di chunking e preprocessing
    to_discard = [] # Indici delle finestre da scartare (finestre con solo valori a zero)

    window_size = estimators[0]['window_size'] # Tutti gli estimatori hanno la stessa window_size, quindi prendo quella del primo

    for es in estimators:
        es['series'] = []   # lista che conterrà le finestre processate

    # Fase di chunking
    for j in range (0, len(magnitude_D), window_size): 

        chunk_D = magnitude_D[j:j + window_size] # finestra lato dominante
        chunk_ND = magnitude_ND[j:j + window_size] # finestra lato non dominante

        if chunk_D.size == window_size and chunk_ND.size == window_size: # Se la finestra è completa (uguale alla window_size)
            
            for es in estimators:
                es['series'].append(elaborate_magnitude(es['method'], chunk_D, chunk_ND)) # Elaborazione della finestra in base al metodo e aggiunta alla lista delle serie dell'estimatore

            if np.all(chunk_D == 0) and np.all(chunk_ND == 0):  # Vengono scartate le finestre con solo valori a zero
                to_discard.append(int(j/window_size)) # Aggiunta dell'indice della finestra alla lista delle finestre da scartare

    y_list = [] # Lista dei valori predetti da ogni classificatore
    hp_tot_list = [] # Lista dei CPI calcolati per questo paziente

    # Fase di predizione
    for es in estimators:
        #print(np.array(es['series']).shape)
        y = es['estimator'].predict(np.array(es['series'])) # Predizione delle etichette per tutte le finestre usando l'estimatore
        #print(es['method'])

        for index in to_discard:
            y[index] = -1 # Le finestre scartate vengono etichettate con -1

        hemi_cluster = es['hemi_cluster'] # Cluster associato agli emiplegici

        cluster_healthy_samples = 0     # Sani
        cluster_hemiplegic_samples = 0  # Emiplegici

        for k in range(len(y)): # Per ogni etichetta predetta
            if y[k] == hemi_cluster: # Se l'etichetta corrisponde al cluster emiplegico
                cluster_hemiplegic_samples += 1 # Incremento del conteggio degli emiplegici
                y[k] = -1 # Viene etichettato come -1
            elif y[k] != -1: # Se l'etichetta non è -1 (cioè non è una finestra scartata)
                cluster_healthy_samples += 1   # Incremento del conteggio dei non emiplegici
                y[k] = 1 # Viene etichettato come 1
            else:
                y[k] = 0 # Viene etichettato come 0 (finestra scartata)
        
        y_list.append(y)

        # Calcolo del CPI
        hp_tot = np.nan # Valore di CPI iniziale
        if (cluster_healthy_samples != 0 or cluster_hemiplegic_samples != 0):
            hp_tot = (cluster_healthy_samples / (cluster_hemiplegic_samples + cluster_healthy_samples)) * 100 # Calcolo del CPI come percentuale di finestre non emiplegiche sul totale delle finestre non scartate

        hp_tot_list.append(hp_tot)

    return y_list, hp_tot_list, magnitude_D, magnitude_ND # Ritorna le liste delle etichette predette, dei CPI calcolati e delle magnitudo D e ND