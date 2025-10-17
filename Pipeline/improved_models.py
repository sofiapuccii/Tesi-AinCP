import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import joblib as jl

def create_improved_classifiers():
    """
    Crea classificatori migliorati con ensemble methods.
    """
    # Classificatori base
    rf = RandomForestClassifier(
        n_estimators=200, # Aumentato da 100 a 200 alberi
        max_depth=10,     # Limitata profondità per ridurre overfitting
        min_samples_split=5,  #minimo 5 campione per split
        random_state=42, # Per riproducibilità
        n_jobs=-1 # Usa tutti i core disponibili
    )
    
    gb = GradientBoostingClassifier(
        n_estimators=100, #100 stimatori in sequenza
        learning_rate=0.1, # Tasso di apprendimento moderato
        max_depth=6, # Profondità massima dell'albero
        random_state=42 # Per riproducibilità
    )
    
    lr = LogisticRegression( 
        random_state=42,
        max_iter=1000
    )
    
    # Combina tutti e 3 i modelli e prende la decisione finale basata sulla media delle loro "opinioni"
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf), # Random Forest
            ('gb', gb), # Gradient Boosting
            ('lr', lr)  # Logistic Regression
        ],
        voting='soft' # Votazione basata sulle probabilità
    )
    '''Esempio
    Per paziente X:
    RF_prediction:[0.2,0.8] 20% dx, 80% sx
    GB_prediction:[0.3,0.7] 30% dx, 70% sx
    LR_prediction:[0.1,0.9] 10% dx, 90% sx
    Media:[(0.2+0.3+0.1)/3,(0.8+0.7+0.9)/3]=[0.2,0.8] -> 80% emiplegia sx'''

    # Pipeline con feature selection
    improved_pipeline = Pipeline([
        ('scaler', StandardScaler()), #normalizzazione
        ('feature_selection', SelectKBest(f_classif, k=50)),  # Seleziona top 50 features
        ('classifier', ensemble) # Classificatore ensemble
    ])
    
    return {
        'RandomForest': rf,
        'GradientBoosting': gb,
        'LogisticRegression': lr,
        'Ensemble': ensemble,
        'ImprovedPipeline': improved_pipeline
    }

#crea regressori per predire punteggi CPI-AHA continui
def create_improved_regressors():
    """
    Crea regressori migliorati.
    """
    # Regressori base
    rf_reg = RandomForestRegressor(
        n_estimators=200, #più alberi = predizioni più stabili
        max_depth=10, # profondità limitata per ridurre overfitting
        min_samples_split=5, #evita overfitting
        random_state=42,
        n_jobs=-1
    )
    
    gb_reg = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1, # Tasso di apprendimento moderato
        max_depth=6, # Profondità massima dell'albero
        random_state=42
    )
    
    #bilancia le predizioni troppo estreme 
    ridge = Ridge(alpha=1.0, random_state=42) #regressione lineare con regolarizzazione L2 per evitare overfitting
    
    # Ensemble voting
    ensemble_reg = VotingRegressor(
        estimators=[
            ('rf', rf_reg),
            ('gb', gb_reg),
            ('ridge', ridge)
        ] 
    )
    
    # Pipeline con feature selection
    improved_pipeline_reg = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(f_regression, k=50)),
        ('regressor', ensemble_reg)
    ])
    
    return {
        'RandomForest': rf_reg,
        'GradientBoosting': gb_reg,
        'Ridge': ridge,
        'Ensemble': ensemble_reg,
        'ImprovedPipeline': improved_pipeline_reg
    }

#Ottimizza automaticamente i parametri dei modelli per massimizzare le performance
def tune_hyperparameters(model, X, y, param_grid, cv=3): # X=features (dati in input), y=target (quello che si vuole predire), param_grid=dizionario con parametri da testare, cv=numero di fold per cross-validation
    """
    Ottimizza iperparametri con GridSearch.
    """
    grid_search = GridSearchCV(
        model, # modello da ottimizzare
        param_grid, #parametri da testare
        cv=cv, # cross-validation
        scoring='accuracy' if len(np.unique(y)) < 10 else 'r2', # accuracy per classificazione, r2 per regressione: se<10 classi uniche (classificazione) se >=10 usa R^2 (regressione)
        n_jobs=-1, # parallelizzazione, usa tutti i core
        verbose=1 # output base
    )
    
    grid_search.fit(X, y) # allena su tutti i dati
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_