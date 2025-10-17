import json
import os
import hashlib
import itertools
import pandas as pd
from train_best_model import train_best_model

# Import compatto per miglioramenti
try:
    from simple_preprocessing import improved_preprocessing
    PREPROCESSING_AVAILABLE = True
except ImportError:
    PREPROCESSING_AVAILABLE = False

def train_select_classifiers(data_folder, save_folder, subjects_indexes, l_window_size=[300, 600, 900], l_method=['concat','difference', 'ai'], config=None):
    
    # Se c'è config migliorata, modifica leggermente i parametri
    if config and config.get('use_advanced_preprocessing') and PREPROCESSING_AVAILABLE:
        print("🚀 Training con preprocessing migliorato")
        # Usa finestre più grandi per sfruttare preprocessing migliore
        if 6400 not in l_window_size:
            l_window_size = [6400] + l_window_size
        
        # Aggiungi modelli più robusti
        extra_boss_params = {'feature_selection': ['chi2', 'none', 'relief'], 'max_ensemble_size': [250, 500]}
        extra_shapedtw_params = {'shape_descriptor_function': ['raw', 'paa', 'slope']}
    else:
        print("📊 Training con metodi tradizionali")
        extra_boss_params = {}
        extra_shapedtw_params = {}
    
    # Configurazioni modelli (leggermente migliorate)
    kmeans_type = 'sktime.clustering.k_means.TimeSeriesKMeans'
    kmeans_params = {'averaging_method': ['mean'], 'init_algorithm': ['kmeans++', 'forgy'], 'metric': ['euclidean', 'dtw'], 'n_clusters': [2]}
    kmeans = (kmeans_type, kmeans_params)

    kmedoids_type = 'sktime.clustering.k_medoids.TimeSeriesKMedoids'
    kmedoids_params = {'init_algorithm': ['forgy', 'random'], 'metric': ['euclidean', 'dtw'], 'n_clusters': [2]}
    kmedoids = (kmedoids_type, kmedoids_params)

    boss_type = 'sktime.classification.dictionary_based._boss.BOSSEnsemble'
    boss_params = {'feature_selection': ['chi2', 'none']}
    if extra_boss_params:
        boss_params.update(extra_boss_params)
    boss = (boss_type, boss_params)

    shapedtw_type = 'sktime.classification.distance_based._shape_dtw.ShapeDTW'
    shapedtw_params = {'shape_descriptor_function': ['raw', 'paa']}
    if extra_shapedtw_params:
        shapedtw_params.update(extra_shapedtw_params)
    shapedtw = (shapedtw_type, shapedtw_params)

    l_gridsearch_specs = [kmeans, kmedoids, boss, shapedtw]

    estimators_l = []
    best_estimators_l = []

    for method, window_size, gridsearch_specs in itertools.product(l_method, l_window_size, l_gridsearch_specs):

        model_type, model_params = gridsearch_specs
        
        # Hash che include anche config per evitare conflitti
        config_hash = ""
        if config and config.get('use_advanced_preprocessing'):
            config_str = f"_advanced_{config.get('bandpass_filter', False)}_{config.get('smart_windowing', False)}"
            config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:5]
        
        gridsearch_hash = hashlib.sha256(json.dumps(model_params, sort_keys=True).encode()).hexdigest()[:10] + config_hash

        print('Method:', method, 'Window size:', window_size, 'Model:', model_type.split(".")[-1])
        if config_hash:
            print('  🚀 Con preprocessing avanzato')

        gridsearch_folder = save_folder + "Trained_models/" + method + "/" + str(window_size) + "_points/" + model_type.split(".")[-1] + "/" + "gridsearch_" + gridsearch_hash + "/"

        if not(os.path.exists(gridsearch_folder + "best_estimator.zip")) or not(os.path.exists(gridsearch_folder + 'GridSearchCV_stats/cv_results.csv')):
            
            # Passa config a train_best_model per preprocessing
            train_best_model(data_folder, subjects_indexes, gridsearch_folder, model_type, model_params, method, window_size, config)

        cv_results = pd.read_csv(gridsearch_folder + 'GridSearchCV_stats/cv_results.csv', index_col=0)
        cv_results.columns = cv_results.columns.str.strip()
        cv_results['method'] = method
        cv_results['window_size'] = window_size
        cv_results['model_type'] = model_type.split(".")[-1]
        cv_results['gridsearch_hash'] = gridsearch_hash
        
        # Aggiungi info su preprocessing usato
        cv_results['advanced_preprocessing'] = config.get('use_advanced_preprocessing', False) if config else False

        estimators_l.append(cv_results)
        best_estimators_l.append(cv_results.iloc[[cv_results['rank_test_score'].argmin()]])

    estimators_df = pd.concat(estimators_l, ignore_index=True)
    best_estimators_df = pd.concat(best_estimators_l, ignore_index=True)

    # Salva con suffisso se avanzato
    suffix = "_advanced" if (config and config.get('use_advanced_preprocessing')) else ""
    
    estimators_df.sort_values(by=['mean_test_score', 'std_test_score'], ascending=False).to_csv(save_folder + f'estimators_results{suffix}.csv')
    best_estimators_df.sort_values(by=['mean_test_score', 'std_test_score'], ascending=False).to_csv(save_folder + f'best_estimators_results{suffix}.csv')
    
    # Salva config usata
    if config:
        with open(save_folder + 'training_config.json', 'w') as f:
            json.dump(config, f, indent=4)