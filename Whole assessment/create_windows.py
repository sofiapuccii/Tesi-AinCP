import pandas as pd
import math
import numpy as np
from elaborate_magnitude import elaborate_magnitude

def create_windows(data_folder, subjects_indexes, operation_type, WINDOW_SIZE):
    series = []
    y_AHA = []
    y_MACS =[]
    y = []
    metadata = pd.read_excel(data_folder + 'metadata2023_08.xlsx').iloc[subjects_indexes]

    for index in range (metadata.shape[0]):
        df = pd.read_csv(data_folder + 'data/' + str(metadata['subject'].iloc[index]) + '_AHA_1sec.csv')

        # Nel caso in cui non bastasse una duplicazione dell'intera time series questa verrà scartata
        if df.shape[0]<WINDOW_SIZE:
            df_concat = pd.concat([df, df.iloc[:WINDOW_SIZE-df.shape[0]]], ignore_index = True, axis = 0)
            df = df_concat
            #print('MODIFICATO Paziente ' + str(j) + ' -> df.shape[0] = ' + str(df.shape[0]))

        scart = (df.shape[0] % WINDOW_SIZE)/2
        
        df_cut = df.iloc[math.ceil(scart):df.shape[0]-math.floor(scart)]

        # Calculating magnitude
        magnitude_D = np.sqrt(np.square(df_cut['x_D']) + np.square(df_cut['y_D']) + np.square(df_cut['z_D']))
        magnitude_ND = np.sqrt(np.square(df_cut['x_ND']) + np.square(df_cut['y_ND']) + np.square(df_cut['z_ND']))
        for i in range (0, len(magnitude_D), WINDOW_SIZE):
            chunk_D = magnitude_D.iloc[i:i + WINDOW_SIZE]
            chunk_ND = magnitude_ND.iloc[i:i + WINDOW_SIZE]
            series.append(elaborate_magnitude(operation_type, chunk_D, chunk_ND))
            y_AHA.append(metadata['AHA'].iloc[index])
            y_MACS.append(metadata['MACS'].iloc[index])
            y.append(metadata['hemi'].iloc[index]-1)
    
    return np.array(series), y_AHA, y_MACS, np.array(y)

    #return np.array(copy.deepcopy(series)), y_AHA, y_MACS, np.array(copy.deepcopy(y))
    # create a list of dictionaries
    #dicts = [{"column": lst} for lst in series]
    #return pd.DataFrame(dicts).copy(), y_AHA, y_MACS, np.array(y)