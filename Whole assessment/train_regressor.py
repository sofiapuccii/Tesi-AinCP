import numpy as np
import joblib as jl
import pandas as pd
from sklearn.linear_model import LinearRegression
from predict_samples import predict_samples
import os

def train_regressor(data_folder, estimators, reg_path, train_indexes):

    hp_tot_lists = []

    metadata = pd.read_excel(data_folder + 'metadata2023_08.xlsx').iloc[train_indexes]

    for i in range (1, metadata.shape[0]+1):
        print('REGRESSOR: PATIENT ', i, 'BEGIN')
        _,hp_tot_list,_,_ = predict_samples(data_folder, estimators, i)
        hp_tot_lists.append(hp_tot_list)
        print('REGRESSOR: PATIENT ', i, 'END')

    lin_reg = LinearRegression()

    X = np.array(hp_tot_lists)
    y = np.array(metadata['AHA'].values)

    print(X)
    print(y)

    # Generate a permutation index and use it to shuffle both arrays
    permutation_idx = np.random.permutation(len(X))
    X_shuffled = X[permutation_idx]
    y_shuffled = y[permutation_idx]
    print('REGRESSOR: START FIT')
    lin_reg.fit(X_shuffled, y_shuffled)
    print('REGRESSOR: END FIT')

    os.makedirs('Regressors', exist_ok = True)
    jl.dump(lin_reg, reg_path)