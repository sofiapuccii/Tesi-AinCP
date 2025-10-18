import pandas as pd
import math
import numpy as np
from elaborate_magnitude import elaborate_magnitude
from scipy.signal import decimate

# Importa preprocessing semplice se disponibile
try:
    from simple_preprocessing import improved_preprocessing
    IMPROVED_PREPROCESSING_AVAILABLE = True
except ImportError:
    IMPROVED_PREPROCESSING_AVAILABLE = False
    print("Preprocessing migliorato non disponibile, uso metodo tradizionale")

def decimate_df(data, factor): # Funzione per il downsampling del DataFrame
    if factor <= 1: 
        return data
    # Isolare le colonne numeriche
    df_axis = data[['x_D', 'y_D', 'z_D', 'x_ND', 'y_ND', 'z_ND']]
    # Eseguire il downsampling
    df_decimated = pd.DataFrame(decimate(df_axis, factor, axis=0, ftype='fir', zero_phase=True), columns=df_axis.columns).reset_index(drop=True)
    #axis=0 specifica su quale asse applicare la decimazione e axis=0 sono le righe (asse temporale); ftype='fir' filtro FIR finite impulse response , zero_phase true evita sfasamenti temporali nel segnale
    #columns =df_axis.columns mantiene i nomi delle colonne originali, .reset_index(drop=true) resetta gli indici del DataFrame decimato
    timestamps = pd.DataFrame(data['datetime'].iloc[::factor].reset_index(drop=True)) # Prende ogni factor-esimo timestamp e resetta gli indici
    # Verifica che le lunghezze combacino
    assert timestamps.shape[0] == df_decimated.shape[0], "Mismatch in lunghezza dopo decimation."
    # Creare un nuovo DataFrame con i dati decimati e i timestamps
    return pd.concat([timestamps, df_decimated], axis=1)

def create_windows(data, window_size, config=None):

    # Se c'è configurazione migliorata e modulo disponibile, usala
    if config is not None and IMPROVED_PREPROCESSING_AVAILABLE and config.get('use_advanced_preprocessing', False):
        print("Usando preprocessing migliorato...")
        result = improved_preprocessing(data, config)
        
        if result['windows'] is not None:
            return result['windows'], result['activity_scores'], result['stats']
        else:
            # Se non usa smart windowing, crea finestre tradizionali sui dati processati
            processed_data = result['processed_data']
            traditional_windows = create_windows_traditional(processed_data, window_size)
            return traditional_windows, [], result['stats'] # Restituisci anche le statistiche 
    
    else:
        # Usa metodo tradizionale
        print("Usando metodo tradizionale...")
        return create_windows_traditional(data, window_size)

#applica una decimazione se i dati sono troppo numerosi poi suddivide la matrice in finestre di dimensione prefissata
def create_windows_traditional(data, window_size):
    """
    Metodo tradizionale per creare finestre.
    """
    windows = []
    
    # Decimazione semplice se necessario 
    if len(data) > window_size * 100:  # Se troppi dati
        decimation_factor = 3  # 80Hz -> 26.67Hz

        data = data[::decimation_factor, :]
        print(f"Decimazione applicata: fattore {decimation_factor}")
    
    # Crea finestre non sovrapposte
    for i in range(0, len(data) - window_size + 1, window_size):
        window = data[i:i + window_size, :]
        windows.append(window)
    
    print(f"Create {len(windows)} finestre tradizionali") 
    return windows

'''def create_windows_legacy(data_folder, subjects_indexes, operation_type, WINDOW_SIZE): # Funzione per creare le finestre di dati
    series = [] # Lista per memorizzare le serie di dati elaborate
    y_AHA = [] 
    y_MACS =[]
    y = [] 
    metadata = pd.read_excel(data_folder + 'metadata2022_04.xlsx').iloc[subjects_indexes] # Legge il file Excel con i metadati e seleziona solo le righe corrispondenti agli indici dei soggetti specificati

    for index in range (metadata.shape[0]): # Per ogni soggetto
        # Lettura del file CSV corrispondente al soggetto
        df = pd.read_csv(data_folder + 'data/AHA/' + str(metadata['subject'].iloc[index]) + '_AHA_RAW.csv')

        # Si fa il downsampling della time series, prendendo un campione ogni 3 (80 Hz -> 26.67 Hz)
        df = decimate_df(df, 3)

        # Nel caso in cui non bastasse una duplicazione dell'intera time series questa verrà scartata
        if df.shape[0]<WINDOW_SIZE: # Se la lunghezza del DataFrame è minore della finestra
            df_concat = pd.concat([df, df.iloc[:WINDOW_SIZE-df.shape[0]]], ignore_index = True, axis = 0) # Concateno il DataFrame con una sua porzione per raggiungere la lunghezza minima
            df = df_concat # Aggiorno il DataFrame
            #print('MODIFICATO Paziente ' + str(j) + ' -> df.shape[0] = ' + str(df.shape[0]))

        scart = (df.shape[0] % WINDOW_SIZE)/2 # Calcolo di quanti campioni scartare per avere un numero di campioni multiplo della finestra
        
        df_cut = df.iloc[math.ceil(scart):df.shape[0]-math.floor(scart)] # Scarto i campioni calcolati all'inizio e alla fine del DataFrame
        Dati: 10.000 campioni, finestra: 3.200 campioni
        10.000 ÷ 3.200 = 3,125 finestre
        Scarto: 400 campioni (200 dall'inizio + 200 dalla fine)
        Risultato: 9.600 campioni = esattamente 3 finestre

        x_D = np.array(df_cut['x_D']) # Converto le colonne del DataFrame in array 
        y_D = np.array(df_cut['y_D'])
        z_D = np.array(df_cut['z_D'])

        x_ND = np.array(df_cut['x_ND'])
        y_ND = np.array(df_cut['y_ND'])
        z_ND = np.array(df_cut['z_ND'])

        # Calculating magnitude
        magnitude_D = np.sqrt(np.square(x_D) + np.square(y_D) + np.square(z_D)) # Calcolo della magnitudo del vettore tridimensionale
        magnitude_ND = np.sqrt(np.square(x_ND) + np.square(y_ND) + np.square(z_ND)) # Calcolo della magnitudo del vettore tridimensionale
        for i in range (0, len(magnitude_D), WINDOW_SIZE): # Per ogni finestra
            chunk_D = magnitude_D[i:i + WINDOW_SIZE] # Prendo la finestra corrente di dati
            chunk_ND = magnitude_ND[i:i + WINDOW_SIZE] 
            series.append(elaborate_magnitude(operation_type, chunk_D, chunk_ND)) # Elaboro la magnitudo e la aggiungo alla lista delle serie
            y_AHA.append(metadata['AHA'].iloc[index]) # Aggiungo l'AHA score del paziente
            y_MACS.append(metadata['MACS'].iloc[index]) #e anche il MACS
            y.append(metadata['hemi'].iloc[index]-1) # Lateralità (0=destra o 1=sinistra)
    
    return np.array(series), y_AHA, y_MACS, np.array(y) # Converto la lista delle serie in un array numpy e restituisco insieme alle etichette'''

    #return np.array(copy.deepcopy(series)), y_AHA, y_MACS, np.array(copy.deepcopy(y))
    # create a list of dictionaries
    #dicts = [{"column": lst} for lst in series]
    #return pd.DataFrame(dicts).copy(), y_AHA, y_MACS, np.array(y)   