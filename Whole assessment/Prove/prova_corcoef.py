import numpy as np

hp_tot_list_list = [[1.0, 20.0, 80.0], [2.0, 21.0, 78.0], [5.0, 51.0, 68.0], [10.0, 101.0, 55.0]]
y = [1, 2, 5, 10]

# Convert lists to numpy arrays
hp_tot_array = np.array(hp_tot_list_list)
y_array = np.array(y)

# Transpose the matrix and calculate the correlation coefficients
print(np.corrcoef(hp_tot_array, y_array, rowvar=False))

print("Correlation Coefficient ", np.corrcoef(np.array(hp_tot_list_list)[:, 0], y)[0, 1])