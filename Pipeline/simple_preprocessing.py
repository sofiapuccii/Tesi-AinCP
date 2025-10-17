import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, filtfilt
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

#Applica un filtro passa-banda per isolare le frequenze in un range specifico
def apply_bandpass_filter(data, fs=80, low_freq=0.5, high_freq=12.0, order=4):
    """
    Applica filtro passa-banda semplice.
    """
    nyquist = fs / 2 #calcolo della frequenza di Nyquist
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    # Butter filter
    b, a = butter(order, [low, high], btype='band') # progetta un filtro Butterworth passa-banda
    
    filtered_data = np.zeros_like(data) # Inizializza array per dati filtrati (preallocazione memoria per efficienza)
    for channel in range(data.shape[1]): # Itera su ciascuna colonna
        filtered_data[:, channel] = filtfilt(b, a, data[:, channel]) # Applica filtro passa-banda alla colonna corrente

    return filtered_data

# Rimozione della gravità con filtro passa-alto 
def remove_gravity_simple(data, fs=80, cutoff=0.1):
    """
    Rimuove gravità con filtro passa-alto semplice.
    """
    nyquist = fs / 2
    cutoff_norm = cutoff / nyquist # Frequenza di taglio normalizzata
    
    b, a = butter(4, cutoff_norm, btype='high') 
    
    detrended_data = np.zeros_like(data)
    for channel in range(data.shape[1]):
        detrended_data[:, channel] = filtfilt(b, a, data[:, channel])
    
    return detrended_data

def calculate_activity_simple(window):
    """
    Calcola attività in modo semplice usando varianza della magnitude.
    """
    magnitude = np.sqrt(np.sum(window**2, axis=1)) # magnitude = √(X² + Y² + Z²)
    return np.var(magnitude) # varianza della magnitude come misura di attività

#creazione di finestre intelligenti con filtro attività
def create_smart_windows(data, window_size_seconds=240, overlap=0.5, fs=26.67, 
                        filter_inactive=True, activity_threshold=0.1):
    """
    Crea finestre intelligenti con filtro attività.
    """
    window_size = int(fs * window_size_seconds) # 6400 campioni per finestra
    stride = int(window_size * (1 - overlap)) # overlap del 50%
    
    windows = []
    activity_scores = []
    
    # Crea finestre con sovrapposizione
    for start in range(0, len(data) - window_size + 1, stride):
        end = start + window_size 
        window = data[start:end, :] #estrae la finestra corrente
        
        # Calcola attività
        activity = calculate_activity_simple(window)
        activity_scores.append(activity)
        
        # Filtra se richiesto
        if filter_inactive and activity < activity_threshold:
            continue  # Salta finestra inattiva
        
        windows.append(window)
    
    print(f"Create {len(windows)} finestre attive su {len(activity_scores)} totali")
    
    return windows, activity_scores

def improved_preprocessing(raw_data, config):
    data = raw_data.copy() # Copia dati originali per evitare modifiche in-place
    
    print(f"Preprocessing - Shape originale: {data.shape}")
    
    # 1. Filtro passa-banda (se abilitato)
    if config.get('bandpass_filter', False):
        data = apply_bandpass_filter(
            data, 
            fs=config['original_fs'],
            low_freq=config['bandpass_low'],
            high_freq=config['bandpass_high']
        )
        print("Filtro passa-banda applicato")
    
    # 2. Rimozione gravità (se abilitato)
    if config.get('remove_gravity', False):
        data = remove_gravity_simple(data, fs=config['original_fs'])
        print("Gravità rimossa")
    
    # 3. Ricampionamento semplice
    if config['original_fs'] != config['target_fs']:
        # Ricampionamento con scipy
        original_length = data.shape[0]
        target_length = int(original_length * config['target_fs'] / config['original_fs']) # Calcola nuova lunghezza
        
        resampled_data = np.zeros((target_length, data.shape[1]))
        for channel in range(data.shape[1]):
            resampled_data[:, channel] = signal.resample(data[:, channel], target_length) # Ricampiona ogni canale
        
        data = resampled_data # Aggiorna la copia locale con versione ricampionata
        print(f"Ricampionato a {config['target_fs']} Hz - Nuova shape: {data.shape}")
    
    # 4. Creazione finestre intelligenti (se abilitato)
    if config.get('smart_windowing', False):
        windows, activity_scores = create_smart_windows(
            data,
            window_size_seconds=config['window_size_seconds'], # crea finestre da 4 minuti 
            overlap=config['window_overlap'],
            fs=config['target_fs'],
            filter_inactive=config['filter_inactive_windows'],
            activity_threshold=config['activity_threshold']
        )
        
        return {
            'processed_data': data,
            'windows': windows,
            'activity_scores': activity_scores,
            'stats': {
                'original_shape': raw_data.shape,
                'processed_shape': data.shape,
                'total_windows': len(activity_scores),
                'active_windows': len(windows)
            }
        }
    else:
        # Usa finestre tradizionali, qui restituisce solo dati processati, le finestre saranno create dal metodo tradizionale create_windows
        return {
            'processed_data': data,
            'windows': None,
            'activity_scores': [],
            'stats': {
                'original_shape': raw_data.shape,
                'processed_shape': data.shape
            }
        }

#trasformazione delle finestre in vettori features numeriche 
def extract_improved_features(windows):
   
    if not windows:
        return np.array([]) # Ritorna array vuoto se nessuna finestra disponibile
    
    features_list = []
    
    for window in windows:
        features = [] # array per contenere le features della finestra corrente
        
        for ch in range(window.shape[1]): # 
            channel_data = window[:, ch] 
            
            # Features statistiche base
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.min(channel_data),
                np.max(channel_data),
                np.median(channel_data)
            ])
            
            # Features avanzate
            features.extend([
                np.percentile(channel_data, 25),
                np.percentile(channel_data, 75),
                np.var(channel_data),
                skew(channel_data) if len(channel_data) > 3 else 0, # per calcolare l'asimmetria controlla che ci siano più di 3 dati
                kurtosis(channel_data) if len(channel_data) > 3 else 0 #parametro che misura lo spessore delle code di distribuzione di probabilità che può essere platicurtica, leptocurtica e normocurtica
            ])
            
            # Features di movimento
            features.extend([
                np.sum(np.abs(np.diff(channel_data))),  # Variazione totale= quanto si muove complessivamente
                np.mean(np.abs(np.diff(channel_data))),  # Mean absolute difference= velocità media di cambiamento (alto valore->movimenti rapidi e frequenti; basso valore->movimenti lenti e stabili)
                len(np.where(np.abs(np.diff(channel_data)) > np.std(channel_data))[0])  # Activity count, numero di movimenti significativi, conta quanti cambiamenti superano la variabilità normale del segnale
            ])
        
        # Features cross-channel
        if window.shape[1] >= 3: #controlla se la finestra ha almeno 3 colonne (AccX,AccY,AccZ)
            # Magnitude
            magnitude = np.sqrt(np.sum(window**2, axis=1))
            features.extend([
                np.mean(magnitude),
                np.std(magnitude),
                np.max(magnitude)
            ])
            
            # Correlazioni tra canali
            for i in range(window.shape[1]):
                for j in range(i+1, window.shape[1]): # evita duplicati
                    corr = np.corrcoef(window[:, i], window[:, j])[0, 1]
                    features.append(corr if not np.isnan(corr) else 0)
        
        features_list.append(features)
    
    return np.array(features_list)