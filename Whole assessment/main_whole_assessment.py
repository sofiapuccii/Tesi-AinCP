from sklearn.model_selection import RepeatedStratifiedKFold
import numpy as np

# Dataset
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
y = np.array([0, 1, 0, 1, 0, 1])

# Repeated Stratified K-Fold cross-validator
rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=42)

# Iterazioni
for train_index, test_index in rskf.split(X, y):

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Stampa degli indici
    print("TRAIN:", train_index, "TEST:", test_index)

    #TODO: Chiamata alla funzione per allenare i migliori modelli e slezionarli

    #TODO: Chiamata alla funzione per allenare regressore e fare assessment