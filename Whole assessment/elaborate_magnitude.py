import numpy as np

def elaborate_magnitude(operation_type, magnitude_D, magnitude_ND):

    if operation_type == 'concat':
        elaborated_magnitude = np.concatenate((magnitude_D, magnitude_ND))
    elif operation_type == 'difference':
        elaborated_magnitude = magnitude_D - magnitude_ND
    elif operation_type == 'ai':
        mask = (magnitude_D + magnitude_ND) != 0
        elaborated_magnitude = np.divide((magnitude_D - magnitude_ND), (magnitude_D + magnitude_ND), where=mask, out=np.zeros_like(magnitude_D)) * 100
    else: 
        print('operation type non supportata.')
        exit(1)

    return elaborated_magnitude