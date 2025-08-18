import os
import numpy as np
import pandas as pd
from train_select_classifiers import train_select_classifiers
from train_regressor import train_regressor
from plotting import plot_dashboards, plot_corrcoeff
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

# Cambio la directory di esecuzione in quella dove si trova questo file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#data_folder = 'C:/Users/david/Documents/University/Borsa di Studio - REDCap/only_AC-80_patients/'
data_folder = '../../../Dati_RAW/'

min_mean_test_score = 0.75 #TODO: change to 0.85
window_size = [4800, 6400, 8000]  # 4800 ≃ 180s, 6400 ≃ 240s, 8000 ≃ 300s

metadata = pd.read_excel(data_folder + 'metadata2022_04.xlsx')
subjects_indexes = list(range(len(metadata)))

# Random state = 42
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
train_indexes, test_indexes = next(sss.split(metadata, metadata['hemi']))

# New
if not os.path.exists('Best_model/'):

    print(' ----- TRAINING CLASSIFIERS ----- ')

    train_select_classifiers(data_folder, save_folder='Best_model/', subjects_indexes=train_indexes, l_window_size = window_size)

if not os.path.exists('Best_model/Regressors/'):

    print(' ----- TRAINING REGRESSOR ----- ')
    
    train_regressor(data_folder, save_folder='Best_model/', train_indexes=train_indexes, min_mean_test_score=min_mean_test_score, window_size=window_size)

if not os.path.exists('Best_model/Week_stats/'):
    
    print(' ----- CREATING DASHBOARDS ----- ')
    
    plot_dashboards(data_folder, save_folder='Best_model/', subjects_indexes=test_indexes, min_mean_test_score=min_mean_test_score, window_size=window_size)
    plot_corrcoeff(iterations_folders=['Best_model/'], save_folder='Best_model/')