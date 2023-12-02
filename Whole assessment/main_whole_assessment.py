import os
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from timeit import default_timer 
from train_classifiers import train_classifiers

# Cambio la directory di esecuzione in quella dove si trova questo file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#data_folder = 'C:/Users/david/Documents/University/Borsa di Studio - REDCap/only_AC-80_patients/'
data_folder = '../../only_AC-80_patients/'
metadata = pd.read_excel(data_folder + 'metadata2023_08.xlsx')
labels = metadata['hemi'].values

# Repeated Stratified K-Fold cross-validator
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5)

# Iterazioni
for train_index, test_index in rskf.split(np.empty(metadata.shape[0]), labels):

    # Stampa degli indici
    #print("TRAIN:", train_index, "TEST:", test_index)

    #Chiamata alla funzione per allenare i migliori modelli e slezionarli
    start = default_timer()
    train_classifiers(data_folder, train_index)
    print("Time needed:", default_timer()-start)    
    
    #TODO: Chiamata alla funzione per allenare regressore e fare assessment