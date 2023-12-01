from sklearn.model_selection import RepeatedStratifiedKFold
import numpy as np
import pandas as pd
import os



from train_classifiers import train_classifiers

# Dataset
#X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
#y = np.array([0, 1, 0, 1, 0, 1])

# Cambio la directory di esecuzione in quella dove si trova questo file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Repeated Stratified K-Fold cross-validator
rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=42)

#data_folder = 'C:/Users/david/Documents/University/Borsa di Studio - REDCap/only_AC-80_patients/'
data_folder = '../../only_AC-80_patients/'
metadata = pd.read_excel(data_folder + 'metadata2023_08.xlsx')

print('Finished')
print(metadata.shape[0])
exit(0)

labels = metadata['hemi'].values

# Iterazioni
for train_index, test_index in rskf.split(np.empty(metadata.shape[0]), labels):

    #X_train, X_test = X[train_index], X[test_index]
    #y_train, y_test = y[train_index], y[test_index]

    # Stampa degli indici
    #print("TRAIN:", train_index, "TEST:", test_index)

    #Chiamata alla funzione per allenare i migliori modelli e slezionarli
    train_classifiers("folder", train_index)

    #TODO: Chiamata alla funzione per allenare regressore e fare assessment