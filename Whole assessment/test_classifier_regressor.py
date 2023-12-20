import os
import json
import hashlib
import numpy as np
import pandas as pd
import joblib as jl
from sktime.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold
from predict_samples import predict_samples
from sklearn.model_selection import cross_val_score

def test_classifier_regressor(data_folder, save_folder, train_indexes, test_indexes, min_mean_test_score, window_size):

    best_estimators_df = pd.read_csv(save_folder + 'best_estimators_results.csv', index_col=0).sort_values(by=['mean_test_score', 'std_test_score'], ascending=False)

    metadata = pd.read_excel(data_folder + 'metadata2023_08.xlsx').iloc[test_indexes]
    metadata.drop(['age_aha', 'gender', 'dom', 'date AHA', 'start AHA', 'stop AHA'], axis=1, inplace=True)

    estimators_specs_list = [row for index, row in best_estimators_df[(best_estimators_df['mean_test_score'] >= min_mean_test_score) & (best_estimators_df['window_size'] == window_size)].iterrows()]

    estimators_list = []
    model_params_concat = ''
    
    for estimators_specs in estimators_specs_list:
        estimator_dir = save_folder + "Trained_models/" + estimators_specs['method'] + "/" + str(estimators_specs['window_size']) + "_seconds/" + estimators_specs['model_type'].split(".")[-1] + "/gridsearch_" + estimators_specs['gridsearch_hash']  + "/"

        with open(estimator_dir + 'GridSearchCV_stats/best_estimator_stats.json', "r") as stats_f:
            grid_search_best_params = json.load(stats_f)

        estimator = BaseEstimator().load_from_path(estimator_dir + 'best_estimator.zip')
        estimators_list.append({'estimator': estimator, 'method': estimators_specs['method'], 'window_size': estimators_specs['window_size'], 'hemi_cluster': grid_search_best_params['Hemi cluster']})
        print('Loaded -> ', estimator_dir + 'best_estimator.zip')
        model_params_concat = model_params_concat + str(estimator.get_params())

    hp_tot_list_list = []
    
    print('qui')

    for index in test_indexes:
        _, hp_tot, _, _ = predict_samples(data_folder, estimators_list, index+1)
        print('quaas')
        hp_tot_list_list.append(hp_tot[0])

        #   hp_tot_list_list =                 y =
        #   [[ 95.0, 90.0, 80.0],              [56,
        #    [ 95.0, 90.0, 80.0],               70,
        #    [ 95.0, 90.0, 80.0],               80,
        #    [ 95.0, 90.0, 80.0]]               34]

    # Alternative
    #for i in range (metadata.shape[0]):
    #    _, hp_tot, _, _ = predict_samples(data_folder, estimators_list, metadata['subject'].iloc[i])
    #    x.append(hp_tot[0])
    #    y.append(metadata['AHA'].iloc[i])

    X = np.array(hp_tot_list_list)
    y = np.array(metadata['AHA'].values)

    #cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    #kf = KFold(n_splits=5, shuffle=True)

    model = LinearRegression()
    rkf = RepeatedKFold(n_splits=5, n_repeats=100)
    score = cross_val_score(model, X, y, cv=rkf)

    print(score)
    print(np.average(score))
    print(score.mean())

    # Organizing data into a dictionary
    data_corrcoef = {
        "Method": estimators_specs_list[0]['method'],
        "Window Size": estimators_specs_list[0]['window_size'],
        "Model Type": estimators_specs_list[0]['model_type'],
        "Gridsearch Hash": estimators_specs_list[0]['gridsearch_hash'],
        "Correlation Coefficient": np.corrcoef(np.array(hp_tot_list_list)[:, 0], y)[0, 1]
    }

    data_regression = {
        "Score": score,
        "Average score": np.average(score)
    }

    data_test = {
        "Best Classifier Stats": data_corrcoef,
        "Regressor Stats": data_regression
    }

    # Writing to a JSON file
    with open(save_folder + 'combined_test_stats.json.json', 'w') as file:
        json.dump(data_test, file, indent=4)

    final_model = LinearRegression()
    print('REGRESSOR: START FIT')
    final_model.fit(X, y)
    print('REGRESSOR: END FIT')
    reg_path = save_folder + 'Regressors/regressor_'+ (hashlib.sha256((model_params_concat).encode()).hexdigest()[:10])
    os.makedirs(reg_path, exist_ok = True)
    jl.dump(final_model, reg_path)