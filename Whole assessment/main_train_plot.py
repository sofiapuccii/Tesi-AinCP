import os
import numpy as np
import pandas as pd
from train_select_classifiers import train_select_classifiers
from train_regressor import train_regressor
from plotting import plot_dashboards, plot_corrcoeff

# Cambio la directory di esecuzione in quella dove si trova questo file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

data_folder = 'C:/Users/david/Documents/University/Borsa di Studio - REDCap/only_AC-80_patients/'
#data_folder = '../../only_AC-80_patients/'

min_mean_test_score = 0.85
window_size = 300

metadata = pd.read_excel(data_folder + 'metadata2023_08.xlsx')
subjects_indexes = list(range(len(metadata)))
np.random.shuffle(subjects_indexes)

# New
if not os.path.exists('Best_model/'):

    print(' ----- TRAINING CLASSIFIERS ----- ')

    train_select_classifiers(data_folder, save_folder='Best_model/', subjects_indexes=subjects_indexes, l_window_size = [window_size])
    
if not os.path.exists('Best_model/Regressors/'):

    print(' ----- TRAINING REGRESSOR ----- ')
    
    train_regressor(data_folder, save_folder='Best_model/', train_indexes=subjects_indexes, min_mean_test_score=min_mean_test_score, window_size=window_size)

if not os.path.exists('Best_model/Week_stats/'):
    
    print(' ----- CREATING DASHBOARDS ----- ')
    
    plot_dashboards(data_folder, save_folder='Best_model/', subjects_indexes=subjects_indexes, min_mean_test_score=min_mean_test_score, window_size=window_size)
    #plot_corrcoeff(stats_folder)