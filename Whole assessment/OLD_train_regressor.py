import numpy as np
import joblib as jl
import pandas as pd
from sklearn.linear_model import LinearRegression
from predict_samples import predict_samples
import os

def train_regressor(data_folder, estimators, reg_path, train_indexes):

    hp_tot_lists = []

    metadata = pd.read_excel(data_folder + 'metadata2023_08.xlsx').iloc[train_indexes]

    for index in train_indexes:
        print('REGRESSOR: PATIENT ', i, 'BEGIN')
        _,hp_tot_list,_,_ = predict_samples(data_folder, estimators, index+1)
        hp_tot_lists.append(hp_tot_list)
        print('REGRESSOR: PATIENT ', i, 'END')

    X = np.array(hp_tot_lists)
    y = np.array(metadata['AHA'].values)


    model = LinearRegression()
    print('REGRESSOR: START FIT')
    model.fit(X, y)
    print('REGRESSOR: END FIT')
    os.makedirs('Regressors', exist_ok = True)
    jl.dump(model, reg_path)