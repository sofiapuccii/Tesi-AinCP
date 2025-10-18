import numpy as np

def elaborate_magnitude(operation_type, magnitude_D, magnitude_ND): # Funzione per elaborare la magnitudo in base al tipo di operazione

    if operation_type == 'concat': 
        elaborated_magnitude = np.concatenate((magnitude_D, magnitude_ND)) # Concatenazione delle due magnitudo per braccio domaninante e non domaninante
    elif operation_type == 'difference':
        elaborated_magnitude = magnitude_D - magnitude_ND # Differenza tra le due magnitudo
    elif operation_type == 'ai':
        mask = (magnitude_D + magnitude_ND) != 0 # Evita divisioni per zero creando una maschera booleana
        elaborated_magnitude = np.divide((magnitude_D - magnitude_ND), (magnitude_D + magnitude_ND), where=mask, out=np.zeros_like(magnitude_D)) * 100 # Calcolo dell'Asymmetry Index (AI) con gestione delle divisioni per zero
    else: 
        print('operation type non supportata.')
        exit(1) 

    return elaborated_magnitude 
